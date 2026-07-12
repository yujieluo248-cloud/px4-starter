#!/usr/bin/env python3
"""
Wide-gate traversal v3.

This node keeps the verified v2 OFFBOARD/ARM and estimator-safety logic, but
uses smooth position setpoints with velocity feedforward and avoids hovering at
the gate entry.
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLocalPosition,
    VehicleStatus,
)


Vector3 = Tuple[float, float, float]


class Phase(Enum):
    WAIT_FOR_POSITION = auto()
    PRESTREAM = auto()
    REQUEST_OFFBOARD_ARM = auto()
    WAIT_FOR_OFFBOARD_ARM = auto()
    SETTLE_HOME_AFTER_ARM = auto()
    TAKEOFF = auto()
    SMOOTH_APPROACH = auto()
    PRE_GATE_ALIGNMENT_CHECK = auto()
    SMOOTH_GATE_CROSS = auto()
    AFTER_GATE_HOLD = auto()
    LAND = auto()
    DONE = auto()


class OffboardWideGateTraversalV3(Node):
    TIMER_PERIOD_S = 0.05
    PRESTREAM_TICKS = 20

    # Gate geometry from test_world_gate_wide collision boxes:
    # world opening x=[0.05, 1.03], y=[3.077722, 3.377722], z=[0.20, 1.68].
    # The launch pose is world (1, 1, 0.1); verified PX4 local +X points toward
    # increasing world Y, and PX4 local -Y points toward lower world X.
    GATE_PLANE_DX_M = 2.23
    GATE_CENTER_DY_M = -0.45

    FLIGHT_UP_M = 0.55
    PRE_GATE_DX_M = 1.45
    PRE_GATE_DY_M = GATE_CENTER_DY_M
    AFTER_GATE_DX_M = 2.75
    AFTER_GATE_DY_M = GATE_CENTER_DY_M

    SMOOTH_APPROACH_SECONDS = 8.00
    SMOOTH_GATE_CROSS_SECONDS = 6.50
    AFTER_GATE_HOLD_SECONDS = 1.50

    MIN_TAKEOFF_REL_HEIGHT_M = 0.40
    TAKEOFF_Z_TOL_M = 0.15
    TAKEOFF_XY_TOL_M = 0.20
    TAKEOFF_STABLE_SECONDS = 0.60
    TAKEOFF_TIMEOUT_SECONDS = 15.00

    MAX_PRE_GATE_LATERAL_ERROR_M = 0.10
    MAX_PRE_GATE_VERTICAL_ERROR_M = 0.12
    ALIGNMENT_CHECK_TIMEOUT_S = 2.00

    MAX_FINAL_HORIZONTAL_ERROR_M = 0.22
    MAX_FINAL_VERTICAL_ERROR_M = 0.20
    TRAJECTORY_CONVERGE_TIMEOUT_S = 2.00

    OFFBOARD_ARM_TIMEOUT_SECONDS = 12.00
    COMMAND_RETRY_SECONDS = 1.00
    HORIZONTAL_RESPONSE_TIMEOUT_SECONDS = 8.00
    MIN_HORIZONTAL_RESPONSE_M = 0.12
    MIN_HORIZONTAL_TARGET_PROJECTION_M = 0.05
    CROSS_GATE_PROGRESS_TIMEOUT_SECONDS = 6.00
    MIN_CROSS_GATE_PROGRESS_M = 0.25

    HOME_SETTLE_SECONDS = 1.50
    HOME_SETTLE_XY_DRIFT_M = 0.08
    HOME_SETTLE_Z_DRIFT_M = 0.08
    HOME_SETTLE_TIMEOUT_SECONDS = 10.00

    MAX_REL_HEIGHT_M = 1.10
    MIN_REL_HEIGHT_M = -0.20

    POSITION_DISCONTINUITY_MAX_DT_S = 0.50
    POSITION_DISCONTINUITY_HORIZONTAL_M = 0.80
    POSITION_DISCONTINUITY_VERTICAL_M = 0.50

    def __init__(self) -> None:
        super().__init__("offboard_wide_gate_traversal_v3")

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10,
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            10,
        )
        self.command_pub = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            10,
        )

        px4_out_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self._on_local_position,
            px4_out_qos,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self._on_vehicle_status,
            px4_out_qos,
        )
        self.create_subscription(
            VehicleCommandAck,
            "/fmu/out/vehicle_command_ack",
            self._on_vehicle_command_ack,
            px4_out_qos,
        )

        self.local_position: Optional[VehicleLocalPosition] = None
        self.vehicle_status: Optional[VehicleStatus] = None
        self.last_command_ack: Optional[VehicleCommandAck] = None
        self.last_offboard_ack: Optional[VehicleCommandAck] = None
        self.last_arm_ack: Optional[VehicleCommandAck] = None

        self.home: Optional[Vector3] = None
        self.preliminary_home: Optional[Vector3] = None
        self.target: Optional[Vector3] = None
        self.target_velocity: Optional[Vector3] = None
        self.settle_anchor_position: Optional[Vector3] = None
        self.settle_stable_since: Optional[float] = None
        self.takeoff_reference_locked = False

        self.trajectory_start_position: Optional[Vector3] = None
        self.trajectory_end_position: Optional[Vector3] = None
        self.trajectory_start_time: Optional[float] = None
        self.trajectory_duration: Optional[float] = None
        self.trajectory_converge_started: Optional[float] = None

        self.move_started: Optional[float] = None
        self.move_start_offset: Optional[Vector3] = None
        self.cross_gate_started: Optional[float] = None
        self.cross_gate_start_offset: Optional[Vector3] = None
        self.gate_plane_passed = False

        self.last_position_sample: Optional[Vector3] = None
        self.last_position_sample_time: Optional[float] = None

        self.phase = Phase.WAIT_FOR_POSITION
        self.phase_started = time.monotonic()
        self.hold_started: Optional[float] = None
        self.takeoff_stable_since: Optional[float] = None
        self.last_log = 0.0
        self.last_command_request = 0.0
        self.prestream_count = 0

        self.offboard_requested = False
        self.arm_requested = False
        self.received_command_ack = False
        self.land_requested = False

        self.timer = self.create_timer(self.TIMER_PERIOD_S, self._tick)

        self.get_logger().info(
            "Wide-gate traversal v3 started. "
            "This run uses smooth takeoff, smooth approach, alignment check, "
            "and continuous gate crossing."
        )
        self.get_logger().info(
            f"FLIGHT_UP_M={self.FLIGHT_UP_M:.2f}, "
            f"GATE_PLANE_DX_M={self.GATE_PLANE_DX_M:.2f}, "
            f"GATE_CENTER_DY_M={self.GATE_CENTER_DY_M:.2f}, "
            f"PRE_GATE=({self.PRE_GATE_DX_M:.2f}, {self.PRE_GATE_DY_M:.2f}), "
            f"AFTER_GATE=({self.AFTER_GATE_DX_M:.2f}, {self.AFTER_GATE_DY_M:.2f})."
        )

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        self.local_position = msg

    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def _on_vehicle_command_ack(self, msg: VehicleCommandAck) -> None:
        self.received_command_ack = True
        self.last_command_ack = msg
        if msg.command == VehicleCommand.VEHICLE_CMD_DO_SET_MODE:
            self.last_offboard_ack = msg
        elif msg.command == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM:
            self.last_arm_ack = msg
        self.get_logger().info(
            "VehicleCommandAck: "
            f"command={msg.command}, "
            f"result={msg.result}({self._ack_result_name(msg.result)}), "
            f"result_param1={msg.result_param1}, "
            f"result_param2={msg.result_param2}"
        )

    def _has_valid_position(self) -> bool:
        if self.local_position is None:
            return False
        return all(
            math.isfinite(value)
            for value in (
                self.local_position.x,
                self.local_position.y,
                self.local_position.z,
            )
        ) and bool(
            self.local_position.xy_valid
        ) and bool(self.local_position.z_valid) and not bool(
            self.local_position.dead_reckoning
        )

    def _position(self) -> Vector3:
        assert self.local_position is not None
        return (
            float(self.local_position.x),
            float(self.local_position.y),
            float(self.local_position.z),
        )

    def _target_from_home(self, dx: float, dy: float) -> Vector3:
        assert self.home is not None
        return (
            self.home[0] + dx,
            self.home[1] + dy,
            self.home[2] - self.FLIGHT_UP_M,
        )

    def _publish_heartbeat(self) -> None:
        msg = OffboardControlMode()
        msg.timestamp = self._now_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def _publish_setpoint(self) -> None:
        if self.target is None:
            return

        nan = math.nan
        msg = TrajectorySetpoint()
        msg.timestamp = self._now_us()
        msg.position = [
            float(self.target[0]),
            float(self.target[1]),
            float(self.target[2]),
        ]
        if self.target_velocity is None:
            msg.velocity = [nan, nan, nan]
        else:
            msg.velocity = [
                float(self.target_velocity[0]),
                float(self.target_velocity[1]),
                float(self.target_velocity[2]),
            ]
        msg.acceleration = [nan, nan, nan]
        msg.jerk = [nan, nan, nan]
        msg.yaw = nan
        msg.yawspeed = nan
        self.setpoint_pub.publish(msg)

    def _command(self, command: int, **params: float) -> None:
        msg = VehicleCommand()
        msg.timestamp = self._now_us()
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        for i in range(1, 8):
            setattr(msg, f"param{i}", float(params.get(f"param{i}", 0.0)))
        self.command_pub.publish(msg)

    def _request_offboard_mode(self) -> None:
        self._command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )
        self.offboard_requested = True
        self.get_logger().info("Requested OFFBOARD mode.")

    def _request_arm(self) -> None:
        self._command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )
        self.arm_requested = True
        self.get_logger().info("Requested ARM.")

    def _request_offboard_and_arm(self) -> None:
        self._request_offboard_mode()
        self._request_arm()
        self.last_command_request = time.monotonic()

    def request_land(self) -> None:
        if self.land_requested:
            return
        self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.land_requested = True
        self.get_logger().info("Requested AUTO LAND.")

    def _set_phase(self, phase: Phase, message: str) -> None:
        self.phase = phase
        self.phase_started = time.monotonic()
        self.hold_started = None
        if phase == Phase.SETTLE_HOME_AFTER_ARM and self._has_valid_position():
            self.settle_anchor_position = self._position()
            self.settle_stable_since = time.monotonic()
            self.target = self.settle_anchor_position
            self.target_velocity = None
        if phase == Phase.PRE_GATE_ALIGNMENT_CHECK:
            self.target = self._target_from_home(
                self.PRE_GATE_DX_M,
                self.PRE_GATE_DY_M,
            )
            self.target_velocity = None
        if phase == Phase.AFTER_GATE_HOLD:
            self.target = self._target_from_home(
                self.AFTER_GATE_DX_M,
                self.AFTER_GATE_DY_M,
            )
            self.target_velocity = None
            self.hold_started = time.monotonic()
        if phase == Phase.LAND:
            self.target = None
            self.target_velocity = None
        if phase == Phase.DONE and hasattr(self, "timer"):
            self.timer.cancel()

        self.get_logger().info(message)
        self._log_target_for_phase(phase)

    def _target_offset(self) -> Vector3:
        if self.home is None or self.target is None:
            return (math.nan, math.nan, math.nan)
        return (
            self.target[0] - self.home[0],
            self.target[1] - self.home[1],
            self.home[2] - self.target[2],
        )

    def _current_offset(self) -> Vector3:
        if self.home is None or self.local_position is None:
            return (math.nan, math.nan, math.nan)
        return (
            self.local_position.x - self.home[0],
            self.local_position.y - self.home[1],
            self.home[2] - self.local_position.z,
        )

    def _settle_drift(self) -> Tuple[float, float]:
        if self.settle_anchor_position is None or self.local_position is None:
            return (math.nan, math.nan)
        x, y, z = self._position()
        anchor_x, anchor_y, anchor_z = self.settle_anchor_position
        xy_drift = math.sqrt((x - anchor_x) ** 2 + (y - anchor_y) ** 2)
        z_drift = abs(z - anchor_z)
        return (xy_drift, z_drift)

    def _z_error_to_target(self) -> float:
        if self.target is None or self.local_position is None:
            return math.nan
        return abs(self.local_position.z - self.target[2])

    def _horizontal_error_to_target(self) -> float:
        if self.target is None or self.local_position is None:
            return math.nan
        x, y, _ = self._position()
        return math.sqrt((x - self.target[0]) ** 2 + (y - self.target[1]) ** 2)

    def _log_target_for_phase(self, phase: Phase) -> None:
        if self.target is None:
            return
        dx, dy, up = self._target_offset()
        self.get_logger().info(
            f"{phase.name} target=({self.target[0]:.3f}, "
            f"{self.target[1]:.3f}, {self.target[2]:.3f}), "
            f"target_offset=({dx:.3f}, {dy:.3f}, {up:.3f})"
        )

    def _distance_to_target(self) -> float:
        if self.target is None or self.local_position is None:
            return math.nan
        x, y, z = self._position()
        return math.sqrt(
            (x - self.target[0]) ** 2
            + (y - self.target[1]) ** 2
            + (z - self.target[2]) ** 2
        )

    def _phase_timeout_seconds(self) -> float:
        if self.phase == Phase.WAIT_FOR_OFFBOARD_ARM:
            return self.OFFBOARD_ARM_TIMEOUT_SECONDS
        if self.phase == Phase.SETTLE_HOME_AFTER_ARM:
            return self.HOME_SETTLE_TIMEOUT_SECONDS
        if self.phase == Phase.PRE_GATE_ALIGNMENT_CHECK:
            return self.ALIGNMENT_CHECK_TIMEOUT_S
        if self.phase == Phase.AFTER_GATE_HOLD:
            return self.AFTER_GATE_HOLD_SECONDS + 1.00
        if self.phase == Phase.TAKEOFF:
            return self.TAKEOFF_TIMEOUT_SECONDS
        if self.phase in (
            Phase.SMOOTH_APPROACH,
            Phase.SMOOTH_GATE_CROSS,
        ):
            duration = self.trajectory_duration or 0.0
            return duration + self.TRAJECTORY_CONVERGE_TIMEOUT_S + 2.00
        return 40.00

    def _phase_timed_out(self) -> bool:
        return time.monotonic() - self.phase_started > self._phase_timeout_seconds()

    def _arming_state(self) -> int:
        if self.vehicle_status is None:
            return -1
        return int(self.vehicle_status.arming_state)

    def _nav_state(self) -> int:
        if self.vehicle_status is None:
            return -1
        return int(self.vehicle_status.nav_state)

    def _failsafe_state(self) -> str:
        if self.vehicle_status is None:
            return "unknown"
        return str(bool(self.vehicle_status.failsafe))

    def _preflight_checks_pass(self) -> str:
        if self.vehicle_status is None:
            return "unknown"
        return str(bool(self.vehicle_status.pre_flight_checks_pass))

    def _position_validity_summary(self) -> str:
        if self.local_position is None:
            return "local_position=none"
        return (
            f"xy_valid={bool(self.local_position.xy_valid)}, "
            f"z_valid={bool(self.local_position.z_valid)}, "
            f"dead_reckoning={bool(self.local_position.dead_reckoning)}"
        )

    def _is_armed(self) -> bool:
        return self._arming_state() == VehicleStatus.ARMING_STATE_ARMED

    def _is_offboard(self) -> bool:
        return self._nav_state() == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def _has_offboard_and_arm(self) -> bool:
        return self._is_armed() and self._is_offboard()

    def _relative_height(self) -> float:
        _, _, up = self._current_offset()
        return up

    def _log_status(self) -> None:
        now = time.monotonic()
        if now - self.last_log < 1.0:
            return

        if self.local_position is None:
            self.get_logger().info(f"{self.phase.name}: waiting for local position.")
            self.last_log = now
            return

        x, y, z = self._position()
        cur_dx, cur_dy, cur_up = self._current_offset()
        tgt_dx, tgt_dy, tgt_up = self._target_offset()

        if self.phase == Phase.SETTLE_HOME_AFTER_ARM:
            xy_drift, z_drift = self._settle_drift()
            anchor = self.settle_anchor_position
            stable_duration = 0.0
            if self.settle_stable_since is not None:
                stable_duration = now - self.settle_stable_since
            anchor_text = "none"
            if anchor is not None:
                anchor_text = f"({anchor[0]:.3f}, {anchor[1]:.3f}, {anchor[2]:.3f})"
            target_text = "target=(none)"
            if self.target is not None:
                target_text = (
                    f"target=({self.target[0]:.3f}, {self.target[1]:.3f}, "
                    f"{self.target[2]:.3f})"
                )
            self.get_logger().info(
                f"{self.phase.name}: current=({x:.3f}, {y:.3f}, {z:.3f}), "
                f"{target_text}, "
                f"current_offset=({cur_dx:.3f}, {cur_dy:.3f}, {cur_up:.3f}), "
                f"target_offset=({tgt_dx:.3f}, {tgt_dy:.3f}, {tgt_up:.3f}), "
                f"rel_height={cur_up:.3f} m, "
                f"z_error={self._z_error_to_target():.3f} m, "
                f"position_error={self._distance_to_target():.3f} m, "
                f"settle_anchor={anchor_text}, xy_drift={xy_drift:.3f} m, "
                f"z_drift={z_drift:.3f} m, stable_duration={stable_duration:.2f} s, "
                f"arming_state={self._arming_state()}, nav_state={self._nav_state()}, "
                f"failsafe={self._failsafe_state()}, "
                f"pre_flight_checks_pass={self._preflight_checks_pass()}, "
                f"{self._position_validity_summary()}"
            )
            self.last_log = now
            return

        if self.target is None:
            target_text = "target=(none)"
        else:
            target_text = (
                f"target=({self.target[0]:.3f}, {self.target[1]:.3f}, "
                f"{self.target[2]:.3f})"
            )

        self.get_logger().info(
            f"{self.phase.name}: current=({x:.3f}, {y:.3f}, {z:.3f}), "
            f"{target_text}, position_error={self._distance_to_target():.3f} m, "
            f"horizontal_error={self._horizontal_error_to_target():.3f} m, "
            f"current_offset=({cur_dx:.3f}, {cur_dy:.3f}, {cur_up:.3f}), "
            f"target_offset=({tgt_dx:.3f}, {tgt_dy:.3f}, {tgt_up:.3f}), "
            f"rel_height={cur_up:.3f} m, "
            f"z_error={self._z_error_to_target():.3f} m, "
            f"arming_state={self._arming_state()}, "
            f"nav_state={self._nav_state()}, "
            f"failsafe={self._failsafe_state()}, "
            f"pre_flight_checks_pass={self._preflight_checks_pass()}, "
            f"{self._position_validity_summary()}"
        )
        self.last_log = now

    def _timeout_to_land(self) -> None:
        self.get_logger().error(
            f"{self.phase.name} timed out after "
            f"{self._phase_timeout_seconds():.1f} s. Requesting AUTO LAND."
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: timeout failsafe.")

    def _ack_result_name(self, result: int) -> str:
        names = {
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED: "ACCEPTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED:
                "TEMPORARILY_REJECTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED: "DENIED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_UNSUPPORTED: "UNSUPPORTED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_FAILED: "FAILED",
            VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS: "IN_PROGRESS",
            VehicleCommandAck.VEHICLE_CMD_RESULT_CANCELLED: "CANCELLED",
        }
        return names.get(int(result), "UNKNOWN")

    def _ack_summary(self, ack: Optional[VehicleCommandAck]) -> str:
        if ack is None:
            return "none"
        return (
            f"command={ack.command}, result={ack.result}"
            f"({self._ack_result_name(ack.result)}), "
            f"result_param1={ack.result_param1}, "
            f"result_param2={ack.result_param2}"
        )

    def _offboard_arm_timeout_to_land(self) -> None:
        self.get_logger().error(
            "WAIT_FOR_OFFBOARD_ARM timed out. "
            f"arming_state={self._arming_state()}, "
            f"nav_state={self._nav_state()}, "
            f"failsafe={self._failsafe_state()}, "
            f"pre_flight_checks_pass={self._preflight_checks_pass()}, "
            f"received_command_ack={self.received_command_ack}, "
            f"last_offboard_ack={self._ack_summary(self.last_offboard_ack)}, "
            f"last_arm_ack={self._ack_summary(self.last_arm_ack)}"
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: OFFBOARD/ARM confirmation failed.")

    def _flight_mode_lost_to_land(self) -> bool:
        if self.phase not in (
            Phase.SETTLE_HOME_AFTER_ARM,
            Phase.TAKEOFF,
            Phase.SMOOTH_APPROACH,
            Phase.PRE_GATE_ALIGNMENT_CHECK,
            Phase.SMOOTH_GATE_CROSS,
            Phase.AFTER_GATE_HOLD,
        ):
            return False
        if self._has_offboard_and_arm():
            return False
        self.get_logger().error(
            "OFFBOARD_OR_ARM_LOST: stopping state machine. "
            f"arming_state={self._arming_state()}, nav_state={self._nav_state()}."
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: OFFBOARD/ARM lost.")
        return True

    def _position_validity_lost_to_land(self) -> bool:
        if self.phase not in (
            Phase.SETTLE_HOME_AFTER_ARM,
            Phase.TAKEOFF,
            Phase.SMOOTH_APPROACH,
            Phase.PRE_GATE_ALIGNMENT_CHECK,
            Phase.SMOOTH_GATE_CROSS,
            Phase.AFTER_GATE_HOLD,
        ):
            return False
        if self._has_valid_position():
            return False

        self.get_logger().error(
            "LOCAL_POSITION_INVALID: requesting AUTO LAND. "
            f"{self._position_validity_summary()}, "
            f"arming_state={self._arming_state()}, nav_state={self._nav_state()}, "
            f"failsafe={self._failsafe_state()}, "
            f"pre_flight_checks_pass={self._preflight_checks_pass()}"
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: local position validity lost.")
        return True

    def _position_discontinuity_to_land(self) -> bool:
        if not self._has_valid_position():
            return False

        now = time.monotonic()
        current = self._position()
        previous = self.last_position_sample
        previous_time = self.last_position_sample_time
        self.last_position_sample = current
        self.last_position_sample_time = now

        if not self.takeoff_reference_locked or previous is None or previous_time is None:
            return False
        if self.phase not in (
            Phase.TAKEOFF,
            Phase.SMOOTH_APPROACH,
            Phase.PRE_GATE_ALIGNMENT_CHECK,
            Phase.SMOOTH_GATE_CROSS,
            Phase.AFTER_GATE_HOLD,
        ):
            return False

        dt = now - previous_time
        if dt <= 0.0 or dt > self.POSITION_DISCONTINUITY_MAX_DT_S:
            return False

        horizontal_jump = math.sqrt(
            (current[0] - previous[0]) ** 2 + (current[1] - previous[1]) ** 2
        )
        vertical_jump = abs(current[2] - previous[2])
        if (
            horizontal_jump <= self.POSITION_DISCONTINUITY_HORIZONTAL_M
            and vertical_jump <= self.POSITION_DISCONTINUITY_VERTICAL_M
        ):
            return False

        cur_dx, cur_dy, cur_up = self._current_offset()
        tgt_dx, tgt_dy, tgt_up = self._target_offset()
        self.get_logger().error(
            "POSITION_DISCONTINUITY_OR_COLLISION: requesting AUTO LAND. "
            f"previous=({previous[0]:.3f}, {previous[1]:.3f}, {previous[2]:.3f}), "
            f"current=({current[0]:.3f}, {current[1]:.3f}, {current[2]:.3f}), "
            f"dt={dt:.3f} s, horizontal_jump={horizontal_jump:.3f} m, "
            f"vertical_jump={vertical_jump:.3f} m, phase={self.phase.name}, "
            f"current_offset=({cur_dx:.3f}, {cur_dy:.3f}, {cur_up:.3f}), "
            f"target_offset=({tgt_dx:.3f}, {tgt_dy:.3f}, {tgt_up:.3f})"
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: position discontinuity or collision.")
        return True

    def _height_failsafe_to_land(self) -> bool:
        if not self.takeoff_reference_locked:
            return False
        if self.home is None or self.local_position is None:
            return False

        rel_height = self._relative_height()
        if rel_height > self.MAX_REL_HEIGHT_M:
            self.get_logger().error(
                f"Relative height {rel_height:.3f} m exceeds "
                f"{self.MAX_REL_HEIGHT_M:.3f} m. Requesting AUTO LAND."
            )
        elif rel_height < self.MIN_REL_HEIGHT_M:
            self.get_logger().error(
                f"Relative height {rel_height:.3f} m is below "
                f"{self.MIN_REL_HEIGHT_M:.3f} m. Requesting AUTO LAND."
            )
        else:
            return False

        self.request_land()
        self._set_phase(Phase.LAND, "LAND: height failsafe.")
        return True

    def _horizontal_progress(self) -> Tuple[float, float]:
        if self.move_start_offset is None or self.target is None:
            return (0.0, 0.0)

        start_dx, start_dy, _ = self.move_start_offset
        cur_dx, cur_dy, _ = self._current_offset()
        tgt_dx, tgt_dy, _ = self._target_offset()

        moved_dx = cur_dx - start_dx
        moved_dy = cur_dy - start_dy
        horizontal_move = math.sqrt(moved_dx**2 + moved_dy**2)

        desired_dx = tgt_dx - start_dx
        desired_dy = tgt_dy - start_dy
        desired_norm = math.sqrt(desired_dx**2 + desired_dy**2)
        target_projection = 0.0
        if desired_norm > 1e-6:
            target_projection = (
                moved_dx * desired_dx + moved_dy * desired_dy
            ) / desired_norm

        return (horizontal_move, target_projection)

    def _horizontal_response_watchdog_to_land(self) -> bool:
        if (
            self.phase != Phase.SMOOTH_APPROACH
            or self.move_started is None
            or self.move_start_offset is None
            or self.target is None
        ):
            return False
        if time.monotonic() - self.move_started < self.HORIZONTAL_RESPONSE_TIMEOUT_SECONDS:
            return False

        horizontal_move, target_projection = self._horizontal_progress()
        if (
            horizontal_move >= self.MIN_HORIZONTAL_RESPONSE_M
            and target_projection >= self.MIN_HORIZONTAL_TARGET_PROJECTION_M
        ):
            return False

        start_dx, start_dy, _ = self.move_start_offset
        cur_dx, cur_dy, _ = self._current_offset()
        tgt_dx, tgt_dy, _ = self._target_offset()
        self.get_logger().error(
            "NO_HORIZONTAL_RESPONSE: requesting AUTO LAND. "
            f"arming_state={self._arming_state()}, nav_state={self._nav_state()}, "
            f"start_offset=({start_dx:.3f}, {start_dy:.3f}), "
            f"current_offset=({cur_dx:.3f}, {cur_dy:.3f}), "
            f"target_offset=({tgt_dx:.3f}, {tgt_dy:.3f}), "
            f"horizontal_move={horizontal_move:.3f}, "
            f"target_projection={target_projection:.3f}"
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: no horizontal response.")
        return True

    def _cross_gate_progress_watchdog_to_land(self) -> bool:
        if (
            self.phase != Phase.SMOOTH_GATE_CROSS
            or self.cross_gate_started is None
            or self.cross_gate_start_offset is None
            or self.target is None
        ):
            return False
        if time.monotonic() - self.cross_gate_started < self.CROSS_GATE_PROGRESS_TIMEOUT_SECONDS:
            return False

        start_dx, start_dy, _ = self.cross_gate_start_offset
        cur_dx, cur_dy, _ = self._current_offset()
        tgt_dx, tgt_dy, _ = self._target_offset()

        moved_dx = cur_dx - start_dx
        moved_dy = cur_dy - start_dy
        desired_dx = tgt_dx - start_dx
        desired_dy = tgt_dy - start_dy
        desired_norm = math.sqrt(desired_dx**2 + desired_dy**2)
        target_projection = 0.0
        if desired_norm > 1e-6:
            target_projection = (
                moved_dx * desired_dx + moved_dy * desired_dy
            ) / desired_norm

        if target_projection >= self.MIN_CROSS_GATE_PROGRESS_M:
            return False

        self.get_logger().error(
            "CROSS_GATE_NO_PROGRESS: requesting AUTO LAND. "
            f"start_offset=({start_dx:.3f}, {start_dy:.3f}), "
            f"current_offset=({cur_dx:.3f}, {cur_dy:.3f}), "
            f"target_offset=({tgt_dx:.3f}, {tgt_dy:.3f}), "
            f"target_projection={target_projection:.3f}, "
            f"arming_state={self._arming_state()}, nav_state={self._nav_state()}"
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: cross-gate no progress.")
        return True

    def _settle_home_ready(self) -> bool:
        if not self._has_valid_position():
            return False
        now = time.monotonic()
        if self.settle_anchor_position is None:
            self.settle_anchor_position = self._position()
            self.settle_stable_since = now
            self.target = self.settle_anchor_position
            self.target_velocity = None
            return False

        xy_drift, z_drift = self._settle_drift()
        if (
            xy_drift <= self.HOME_SETTLE_XY_DRIFT_M
            and z_drift <= self.HOME_SETTLE_Z_DRIFT_M
        ):
            if self.settle_stable_since is None:
                self.settle_stable_since = now
            return now - self.settle_stable_since >= self.HOME_SETTLE_SECONDS

        self.settle_anchor_position = self._position()
        self.settle_stable_since = now
        self.target = self.settle_anchor_position
        self.target_velocity = None
        self.get_logger().info(
            "SETTLE_HOME_AFTER_ARM: local position drift exceeded threshold; "
            "resetting settle anchor and hold target."
        )
        return False

    def _settle_home_timeout_to_land(self) -> None:
        xy_drift, z_drift = self._settle_drift()
        self.get_logger().error(
            "SETTLE_HOME_AFTER_ARM timed out before local position stabilized. "
            f"xy_drift={xy_drift:.3f} m, z_drift={z_drift:.3f} m, "
            f"arming_state={self._arming_state()}, nav_state={self._nav_state()}, "
            f"failsafe={self._failsafe_state()}, "
            f"pre_flight_checks_pass={self._preflight_checks_pass()}"
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: takeoff home did not stabilize.")

    def _lock_stable_takeoff_home(self) -> None:
        previous_home = self.home
        stable_home = self._position()
        if previous_home is None:
            previous_home = stable_home

        self.home = stable_home
        self.target = self.home
        self.target_velocity = None
        self.takeoff_reference_locked = True
        self.last_position_sample = stable_home
        self.last_position_sample_time = time.monotonic()

        shift = (
            stable_home[0] - previous_home[0],
            stable_home[1] - previous_home[1],
            stable_home[2] - previous_home[2],
        )
        self.get_logger().info(
            f"Previous preliminary home=({previous_home[0]:.3f}, "
            f"{previous_home[1]:.3f}, {previous_home[2]:.3f})"
        )
        self.get_logger().info(
            f"Stable takeoff home=({stable_home[0]:.3f}, "
            f"{stable_home[1]:.3f}, {stable_home[2]:.3f})"
        )
        self.get_logger().info(
            f"Reference shift=({shift[0]:.3f}, {shift[1]:.3f}, {shift[2]:.3f})"
        )
        self.get_logger().info("Stable takeoff reference locked.")

    def _capture_home(self) -> None:
        self.home = self._position()
        self.preliminary_home = self.home
        self.takeoff_reference_locked = False
        self.target = self.home
        self.target_velocity = None
        self._set_phase(
            Phase.PRESTREAM,
            "PRESTREAM: sending home setpoints before OFFBOARD / ARM.",
        )
        self.get_logger().info(
            f"Captured preliminary home=({self.home[0]:.3f}, {self.home[1]:.3f}, "
            f"{self.home[2]:.3f})."
        )

    def _smoothstep(self, u: float) -> Tuple[float, float]:
        u = max(0.0, min(1.0, u))
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        ds_du = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
        return (s, ds_du)

    def _start_smooth_phase(
        self,
        phase: Phase,
        end_position: Vector3,
        duration: float,
        message: str,
    ) -> None:
        start_position = self._position() if self._has_valid_position() else (
            self.target if self.target is not None else end_position
        )
        assert start_position is not None
        self.trajectory_start_position = start_position
        self.trajectory_end_position = end_position
        self.trajectory_start_time = time.monotonic()
        self.trajectory_duration = duration
        self.trajectory_converge_started = None
        self.target = start_position
        self.target_velocity = (0.0, 0.0, 0.0)

        if phase == Phase.SMOOTH_APPROACH:
            self.move_started = time.monotonic()
            self.move_start_offset = self._current_offset()
        if phase == Phase.SMOOTH_GATE_CROSS:
            self.cross_gate_started = time.monotonic()
            self.cross_gate_start_offset = self._current_offset()
            self.gate_plane_passed = False
            self.get_logger().info("CROSS_GATE_STARTED")

        self._set_phase(phase, message)
        end_dx = end_position[0] - self.home[0] if self.home is not None else math.nan
        end_dy = end_position[1] - self.home[1] if self.home is not None else math.nan
        end_up = self.home[2] - end_position[2] if self.home is not None else math.nan
        self.get_logger().info(
            f"{phase.name} end=({end_position[0]:.3f}, {end_position[1]:.3f}, "
            f"{end_position[2]:.3f}), end_offset=({end_dx:.3f}, "
            f"{end_dy:.3f}, {end_up:.3f}), duration={duration:.2f} s"
        )

    def _update_smooth_target(self) -> None:
        if (
            self.phase not in (
                Phase.SMOOTH_APPROACH,
                Phase.SMOOTH_GATE_CROSS,
            )
            or self.trajectory_start_position is None
            or self.trajectory_end_position is None
            or self.trajectory_start_time is None
            or self.trajectory_duration is None
        ):
            return

        now = time.monotonic()
        elapsed = max(0.0, now - self.trajectory_start_time)
        u = elapsed / self.trajectory_duration
        s, ds_du = self._smoothstep(u)
        p0 = self.trajectory_start_position
        p1 = self.trajectory_end_position
        self.target = (
            p0[0] + (p1[0] - p0[0]) * s,
            p0[1] + (p1[1] - p0[1]) * s,
            p0[2] + (p1[2] - p0[2]) * s,
        )
        ds_dt = ds_du / self.trajectory_duration
        self.target_velocity = (
            (p1[0] - p0[0]) * ds_dt,
            (p1[1] - p0[1]) * ds_dt,
            (p1[2] - p0[2]) * ds_dt,
        )
        if u >= 1.0:
            self.target = p1
            self.target_velocity = (0.0, 0.0, 0.0)

    def _smooth_elapsed(self) -> float:
        if self.trajectory_start_time is None:
            return 0.0
        return time.monotonic() - self.trajectory_start_time

    def _smooth_time_complete(self) -> bool:
        if self.trajectory_duration is None:
            return False
        return self._smooth_elapsed() >= self.trajectory_duration

    def _final_converged(self) -> bool:
        return (
            self._horizontal_error_to_target() <= self.MAX_FINAL_HORIZONTAL_ERROR_M
            and self._z_error_to_target() <= self.MAX_FINAL_VERTICAL_ERROR_M
        )

    def _convergence_timed_out(self) -> bool:
        if not self._smooth_time_complete():
            return False
        now = time.monotonic()
        if self.trajectory_converge_started is None:
            self.trajectory_converge_started = now
            return False
        return now - self.trajectory_converge_started > self.TRAJECTORY_CONVERGE_TIMEOUT_S

    def _smooth_phase_failed_to_land(self) -> None:
        self.get_logger().error(
            f"{self.phase.name} failed to converge after smooth trajectory. "
            f"horizontal_error={self._horizontal_error_to_target():.3f} m, "
            f"z_error={self._z_error_to_target():.3f} m, "
            f"current_offset=({self._current_offset()[0]:.3f}, "
            f"{self._current_offset()[1]:.3f}, {self._current_offset()[2]:.3f}), "
            f"target_offset=({self._target_offset()[0]:.3f}, "
            f"{self._target_offset()[1]:.3f}, {self._target_offset()[2]:.3f})"
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: smooth trajectory convergence failed.")

    def _takeoff_ready(self) -> bool:
        if (
            not self._has_valid_position()
            or self.target is None
            or not self._has_offboard_and_arm()
        ):
            self.takeoff_stable_since = None
            return False

        cur_dx, cur_dy, cur_up = self._current_offset()
        z_error = self._z_error_to_target()
        horizontal_error = math.sqrt(cur_dx**2 + cur_dy**2)
        ready = (
            cur_up >= self.MIN_TAKEOFF_REL_HEIGHT_M
            and z_error <= self.TAKEOFF_Z_TOL_M
            and horizontal_error <= self.TAKEOFF_XY_TOL_M
        )
        if not ready:
            self.takeoff_stable_since = None
            return False

        now = time.monotonic()
        if self.takeoff_stable_since is None:
            self.takeoff_stable_since = now
        return now - self.takeoff_stable_since >= self.TAKEOFF_STABLE_SECONDS

    def _takeoff_failed_to_land(self) -> None:
        if self.local_position is None:
            current_text = "none"
        else:
            x, y, z = self._position()
            current_text = f"({x:.3f}, {y:.3f}, {z:.3f})"
        cur_dx, cur_dy, cur_up = self._current_offset()
        tgt_dx, tgt_dy, tgt_up = self._target_offset()
        self.get_logger().error(
            "TAKEOFF_FAILED_TO_REACH_SAFE_HEIGHT: requesting AUTO LAND. "
            f"current={current_text}, "
            f"current_offset=({cur_dx:.3f}, {cur_dy:.3f}, {cur_up:.3f}), "
            f"target_offset=({tgt_dx:.3f}, {tgt_dy:.3f}, {tgt_up:.3f}), "
            f"rel_height={cur_up:.3f}, "
            f"z_error={self._z_error_to_target():.3f}, "
            f"horizontal_error={self._horizontal_error_to_target():.3f}, "
            f"arming_state={self._arming_state()}, "
            f"nav_state={self._nav_state()}"
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: takeoff failed to reach safe height.")

    def _pre_gate_alignment_ready(self) -> bool:
        cur_dx, cur_dy, cur_up = self._current_offset()
        tgt_dx, tgt_dy, tgt_up = self._target_offset()
        lateral_error = abs(cur_dy - tgt_dy)
        vertical_error = abs(cur_up - tgt_up)
        if (
            lateral_error <= self.MAX_PRE_GATE_LATERAL_ERROR_M
            and vertical_error <= self.MAX_PRE_GATE_VERTICAL_ERROR_M
        ):
            self.get_logger().info(
                "PRE_GATE_ALIGNMENT_PASSED: "
                f"current_offset=({cur_dx:.3f}, {cur_dy:.3f}, {cur_up:.3f}), "
                f"target_offset=({tgt_dx:.3f}, {tgt_dy:.3f}, {tgt_up:.3f}), "
                f"lateral_error={lateral_error:.3f}, "
                f"vertical_error={vertical_error:.3f}"
            )
            return True
        return False

    def _pre_gate_alignment_failed_to_land(self) -> None:
        cur_dx, cur_dy, cur_up = self._current_offset()
        tgt_dx, tgt_dy, tgt_up = self._target_offset()
        self.get_logger().error(
            "PRE_GATE_ALIGNMENT_FAILED: requesting AUTO LAND. "
            f"current_offset=({cur_dx:.3f}, {cur_dy:.3f}, {cur_up:.3f}), "
            f"target_offset=({tgt_dx:.3f}, {tgt_dy:.3f}, {tgt_up:.3f}), "
            f"lateral_error={abs(cur_dy - tgt_dy):.3f}, "
            f"vertical_error={abs(cur_up - tgt_up):.3f}, "
            f"arming_state={self._arming_state()}, nav_state={self._nav_state()}, "
            f"{self._position_validity_summary()}"
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: pre-gate alignment failed.")

    def _check_gate_plane_passed(self) -> None:
        if self.phase != Phase.SMOOTH_GATE_CROSS or self.gate_plane_passed:
            return
        cur_dx, cur_dy, cur_up = self._current_offset()
        if cur_dx >= self.GATE_PLANE_DX_M:
            self.gate_plane_passed = True
            self.get_logger().info(
                "GATE_PLANE_PASSED: "
                f"current_offset=({cur_dx:.3f}, {cur_dy:.3f}, {cur_up:.3f}), "
                f"gate_plane_dx={self.GATE_PLANE_DX_M:.3f}"
            )

    def _tick(self) -> None:
        self._log_status()

        if self.phase == Phase.DONE:
            return

        if self.phase == Phase.WAIT_FOR_POSITION:
            if not self._has_valid_position():
                if self._phase_timed_out():
                    self._timeout_to_land()
                return
            self._capture_home()

        if self.phase == Phase.LAND:
            if (
                self.vehicle_status is not None
                and self.vehicle_status.arming_state
                == VehicleStatus.ARMING_STATE_DISARMED
            ):
                self._set_phase(Phase.DONE, "DONE: vehicle disarmed.")
            return

        if (
            self.phase in (Phase.REQUEST_OFFBOARD_ARM, Phase.WAIT_FOR_OFFBOARD_ARM)
            and self._has_valid_position()
        ):
            self.target = self._position()
            self.target_velocity = None

        if self._position_validity_lost_to_land():
            return
        if self._position_discontinuity_to_land():
            return

        self._update_smooth_target()

        if self.phase not in (Phase.LAND, Phase.DONE):
            if self._height_failsafe_to_land():
                return
            if self._flight_mode_lost_to_land():
                return
            if self._phase_timed_out():
                if self.phase == Phase.WAIT_FOR_OFFBOARD_ARM:
                    self._offboard_arm_timeout_to_land()
                    return
                if self.phase == Phase.SETTLE_HOME_AFTER_ARM:
                    self._settle_home_timeout_to_land()
                    return
                if self.phase == Phase.PRE_GATE_ALIGNMENT_CHECK:
                    self._pre_gate_alignment_failed_to_land()
                    return
                if self.phase == Phase.TAKEOFF:
                    self._takeoff_failed_to_land()
                    return
                self._timeout_to_land()
                return

        self._publish_heartbeat()
        self._publish_setpoint()

        if self.phase == Phase.PRESTREAM:
            self.prestream_count += 1
            if self.prestream_count >= self.PRESTREAM_TICKS:
                self._set_phase(
                    Phase.REQUEST_OFFBOARD_ARM,
                    "REQUEST_OFFBOARD_ARM: requesting OFFBOARD and ARM.",
                )
            return

        if self.phase == Phase.REQUEST_OFFBOARD_ARM:
            self._request_offboard_and_arm()
            self._set_phase(
                Phase.WAIT_FOR_OFFBOARD_ARM,
                "WAIT_FOR_OFFBOARD_ARM: waiting for confirmed OFFBOARD and ARMED.",
            )
            return

        if self.phase == Phase.WAIT_FOR_OFFBOARD_ARM:
            if self._has_offboard_and_arm():
                self._set_phase(
                    Phase.SETTLE_HOME_AFTER_ARM,
                    "SETTLE_HOME_AFTER_ARM: confirmed OFFBOARD/ARMED, holding "
                    "current position before locking takeoff home.",
                )
                return
            if time.monotonic() - self.last_command_request >= self.COMMAND_RETRY_SECONDS:
                self._request_offboard_and_arm()
            return

        if self.phase == Phase.SETTLE_HOME_AFTER_ARM:
            if self._settle_home_ready():
                self._lock_stable_takeoff_home()
                self.target = self._target_from_home(0.0, 0.0)
                self.target_velocity = None
                self.takeoff_stable_since = None
                self._set_phase(
                    Phase.TAKEOFF,
                    "TAKEOFF: climbing with fixed position setpoint.",
                )
            return

        if self.phase == Phase.TAKEOFF:
            if self._takeoff_ready():
                self._start_smooth_phase(
                    Phase.SMOOTH_APPROACH,
                    self._target_from_home(self.PRE_GATE_DX_M, self.PRE_GATE_DY_M),
                    self.SMOOTH_APPROACH_SECONDS,
                    "SMOOTH_APPROACH: moving to safe pre-gate alignment point.",
                )
                return
            return

        if self.phase == Phase.SMOOTH_APPROACH:
            if self._horizontal_response_watchdog_to_land():
                return
            if self._smooth_time_complete():
                self._set_phase(
                    Phase.PRE_GATE_ALIGNMENT_CHECK,
                    "PRE_GATE_ALIGNMENT_CHECK: checking lateral and vertical "
                    "alignment before continuous crossing.",
                )
            return

        if self.phase == Phase.PRE_GATE_ALIGNMENT_CHECK:
            if self._pre_gate_alignment_ready():
                self._start_smooth_phase(
                    Phase.SMOOTH_GATE_CROSS,
                    self._target_from_home(self.AFTER_GATE_DX_M, self.AFTER_GATE_DY_M),
                    self.SMOOTH_GATE_CROSS_SECONDS,
                    "SMOOTH_GATE_CROSS: crossing continuously to after-gate point.",
                )
            return

        if self.phase == Phase.SMOOTH_GATE_CROSS:
            if self._cross_gate_progress_watchdog_to_land():
                return
            self._check_gate_plane_passed()
            if self._smooth_time_complete() and self._final_converged():
                self.get_logger().info("GATE_CROSSING_COMPLETED")
                self._set_phase(
                    Phase.AFTER_GATE_HOLD,
                    "AFTER_GATE_HOLD: holding after gate briefly before landing.",
                )
                return
            if self._convergence_timed_out():
                self._smooth_phase_failed_to_land()
            return

        if self.phase == Phase.AFTER_GATE_HOLD:
            if self.hold_started is None:
                self.hold_started = time.monotonic()
            if time.monotonic() - self.hold_started >= self.AFTER_GATE_HOLD_SECONDS:
                self.request_land()
                self._set_phase(Phase.LAND, "LAND: wide-gate traversal complete.")
            return


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OffboardWideGateTraversalV3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C received. Requesting AUTO LAND.")
        node.request_land()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
