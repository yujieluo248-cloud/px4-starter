#!/usr/bin/env python3
"""
Wide-gate traversal v1: pre-gate direction verification only.

This script intentionally does not cross the gate. It only verifies the PX4
local offset direction near the start of the wide-gate route.
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


class Phase(Enum):
    WAIT_FOR_POSITION = auto()
    PRESTREAM = auto()
    REQUEST_OFFBOARD_ARM = auto()
    WAIT_FOR_OFFBOARD_ARM = auto()
    SETTLE_HOME_AFTER_ARM = auto()
    TAKEOFF = auto()
    MOVE_LOCAL_XY_TEST_POINT = auto()
    HOLD = auto()
    LAND = auto()
    DONE = auto()


class OffboardWideGateTraversalV1(Node):
    TIMER_PERIOD_S = 0.10
    PRESTREAM_TICKS = 20

    STOP_AT_PRE_GATE = True

    TAKEOFF_UP_M = 0.45
    TEST_DX_M = 0.80
    TEST_DY_M = -0.20

    POSITION_TOL_M = 0.22
    STABLE_SECONDS = 0.60
    TAKEOFF_Z_TOL_M = 0.18
    MIN_TAKEOFF_REL_HEIGHT_M = 0.25
    HOLD_SECONDS = 5.00
    PHASE_TIMEOUT_SECONDS = 40.00
    OFFBOARD_ARM_TIMEOUT_SECONDS = 12.00
    COMMAND_RETRY_SECONDS = 1.00
    HORIZONTAL_RESPONSE_TIMEOUT_SECONDS = 8.00
    MIN_HORIZONTAL_RESPONSE_M = 0.12
    MIN_HORIZONTAL_TARGET_PROJECTION_M = 0.05

    HOME_SETTLE_SECONDS = 1.50
    HOME_SETTLE_XY_DRIFT_M = 0.08
    HOME_SETTLE_Z_DRIFT_M = 0.08
    HOME_SETTLE_TIMEOUT_SECONDS = 10.00

    MAX_REL_HEIGHT_M = 0.90
    MIN_REL_HEIGHT_M = -0.20

    def __init__(self) -> None:
        super().__init__("offboard_wide_gate_traversal_v1")

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
        self.home: Optional[Tuple[float, float, float]] = None
        self.preliminary_home: Optional[Tuple[float, float, float]] = None
        self.target: Optional[Tuple[float, float, float]] = None
        self.move_start_offset: Optional[Tuple[float, float, float]] = None
        self.settle_anchor_position: Optional[Tuple[float, float, float]] = None
        self.settle_stable_since: Optional[float] = None
        self.takeoff_reference_locked = False

        self.phase = Phase.WAIT_FOR_POSITION
        self.phase_started = time.monotonic()
        self.stable_since: Optional[float] = None
        self.hold_started: Optional[float] = None
        self.move_started: Optional[float] = None
        self.last_log = 0.0
        self.last_command_request = 0.0
        self.prestream_count = 0

        self.offboard_requested = False
        self.arm_requested = False
        self.received_command_ack = False
        self.land_requested = False

        self.timer = self.create_timer(self.TIMER_PERIOD_S, self._tick)

        self.get_logger().info(
            "Wide-gate v1 direction verification started. "
            "This run stops at the local XY test point and lands."
        )
        self.get_logger().info(
            f"STOP_AT_PRE_GATE={self.STOP_AT_PRE_GATE}, "
            f"TEST_DX_M={self.TEST_DX_M:.2f}, TEST_DY_M={self.TEST_DY_M:.2f}."
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

    def _position(self) -> Tuple[float, float, float]:
        assert self.local_position is not None
        return (
            float(self.local_position.x),
            float(self.local_position.y),
            float(self.local_position.z),
        )

    def _target_from_home(self, dx: float, dy: float) -> Tuple[float, float, float]:
        assert self.home is not None
        return (
            self.home[0] + dx,
            self.home[1] + dy,
            self.home[2] - self.TAKEOFF_UP_M,
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
        msg.velocity = [nan, nan, nan]
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
        self.stable_since = None
        self.hold_started = None
        if phase == Phase.SETTLE_HOME_AFTER_ARM and self._has_valid_position():
            self.settle_anchor_position = self._position()
            self.settle_stable_since = time.monotonic()
            self.target = self.settle_anchor_position
        if phase == Phase.LAND:
            self.target = None
        self.get_logger().info(message)
        self._log_target_for_phase(phase)

        if phase == Phase.MOVE_LOCAL_XY_TEST_POINT:
            self.move_started = time.monotonic()
            self.move_start_offset = self._current_offset()
        if phase == Phase.DONE and hasattr(self, "timer"):
            self.timer.cancel()

    def _target_offset(self) -> Tuple[float, float, float]:
        if self.home is None or self.target is None:
            return (math.nan, math.nan, math.nan)
        return (
            self.target[0] - self.home[0],
            self.target[1] - self.home[1],
            self.home[2] - self.target[2],
        )

    def _current_offset(self) -> Tuple[float, float, float]:
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

    def _stable_at_target(self) -> bool:
        if self._distance_to_target() <= self.POSITION_TOL_M:
            if self.stable_since is None:
                self.stable_since = time.monotonic()
            return time.monotonic() - self.stable_since >= self.STABLE_SECONDS
        self.stable_since = None
        self.hold_started = None
        return False

    def _hold_complete_after_reach(self) -> bool:
        if not self._stable_at_target():
            return False
        if self.hold_started is None:
            self.hold_started = time.monotonic()
        return time.monotonic() - self.hold_started >= self.HOLD_SECONDS

    def _takeoff_ready(self) -> bool:
        if self.local_position is None or self.target is None:
            return False
        cur_dx, cur_dy, cur_up = self._current_offset()
        z_error = abs(self.local_position.z - self.target[2])
        xy_error = math.sqrt(cur_dx**2 + cur_dy**2)
        ready = (
            cur_up >= self.MIN_TAKEOFF_REL_HEIGHT_M
            and z_error <= self.TAKEOFF_Z_TOL_M
            and xy_error <= self.POSITION_TOL_M
        )
        if ready:
            if self.stable_since is None:
                self.stable_since = time.monotonic()
            return time.monotonic() - self.stable_since >= self.STABLE_SECONDS
        self.stable_since = None
        return False

    def _phase_timeout_seconds(self) -> float:
        if self.phase == Phase.WAIT_FOR_OFFBOARD_ARM:
            return self.OFFBOARD_ARM_TIMEOUT_SECONDS
        if self.phase == Phase.SETTLE_HOME_AFTER_ARM:
            return self.HOME_SETTLE_TIMEOUT_SECONDS
        return self.PHASE_TIMEOUT_SECONDS

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
        if self.phase == Phase.SETTLE_HOME_AFTER_ARM:
            xy_drift, z_drift = self._settle_drift()
            anchor = self.settle_anchor_position
            stable_duration = 0.0
            if self.settle_stable_since is not None:
                stable_duration = now - self.settle_stable_since
            anchor_text = "none"
            if anchor is not None:
                anchor_text = f"({anchor[0]:.3f}, {anchor[1]:.3f}, {anchor[2]:.3f})"
            self.get_logger().info(
                f"{self.phase.name}: current=({x:.3f}, {y:.3f}, {z:.3f}), "
                f"settle_anchor={anchor_text}, xy_drift={xy_drift:.3f} m, "
                f"z_drift={z_drift:.3f} m, stable_duration={stable_duration:.2f} s, "
                f"arming_state={self._arming_state()}, nav_state={self._nav_state()}, "
                f"failsafe={self._failsafe_state()}, "
                f"pre_flight_checks_pass={self._preflight_checks_pass()}"
            )
            self.last_log = now
            return

        cur_dx, cur_dy, cur_up = self._current_offset()
        tgt_dx, tgt_dy, tgt_up = self._target_offset()

        if self.target is None:
            target_text = "target=(none)"
        else:
            target_text = (
                f"target=({self.target[0]:.3f}, {self.target[1]:.3f}, "
                f"{self.target[2]:.3f})"
            )

        self.get_logger().info(
            f"{self.phase.name}: current=({x:.3f}, {y:.3f}, {z:.3f}), "
            f"{target_text}, error={self._distance_to_target():.3f} m, "
            f"current_offset=({cur_dx:.3f}, {cur_dy:.3f}, {cur_up:.3f}), "
            f"rel_height={cur_up:.3f} m, "
            f"z_error={self._z_error_to_target():.3f} m, "
            f"target_offset=({tgt_dx:.3f}, {tgt_dy:.3f}, {tgt_up:.3f}), "
            f"arming_state={self._arming_state()}, "
            f"nav_state={self._nav_state()}, "
            f"failsafe={self._failsafe_state()}, "
            f"pre_flight_checks_pass={self._preflight_checks_pass()}"
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
            Phase.MOVE_LOCAL_XY_TEST_POINT,
            Phase.HOLD,
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

    def _position_validity_summary(self) -> str:
        if self.local_position is None:
            return "local_position=none"
        return (
            f"xy_valid={bool(self.local_position.xy_valid)}, "
            f"z_valid={bool(self.local_position.z_valid)}, "
            f"dead_reckoning={bool(self.local_position.dead_reckoning)}"
        )

    def _position_validity_lost_to_land(self) -> bool:
        if self.phase not in (
            Phase.SETTLE_HOME_AFTER_ARM,
            Phase.TAKEOFF,
            Phase.MOVE_LOCAL_XY_TEST_POINT,
            Phase.HOLD,
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

    def _horizontal_response_watchdog_to_land(self) -> bool:
        if (
            self.phase != Phase.MOVE_LOCAL_XY_TEST_POINT
            or self.move_started is None
            or self.move_start_offset is None
            or self.target is None
        ):
            return False
        if time.monotonic() - self.move_started < self.HORIZONTAL_RESPONSE_TIMEOUT_SECONDS:
            return False

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

        if (
            horizontal_move >= self.MIN_HORIZONTAL_RESPONSE_M
            and target_projection >= self.MIN_HORIZONTAL_TARGET_PROJECTION_M
        ):
            return False

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

    def _settle_home_ready(self) -> bool:
        if not self._has_valid_position():
            return False
        now = time.monotonic()
        if self.settle_anchor_position is None:
            self.settle_anchor_position = self._position()
            self.settle_stable_since = now
            self.target = self.settle_anchor_position
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
        self.takeoff_reference_locked = True

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
        self._set_phase(
            Phase.PRESTREAM,
            "PRESTREAM: sending home setpoints before OFFBOARD / ARM.",
        )
        self.get_logger().info(
            f"Captured preliminary home=({self.home[0]:.3f}, {self.home[1]:.3f}, "
            f"{self.home[2]:.3f})."
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

        if self._position_validity_lost_to_land():
            return

        self._publish_heartbeat()
        self._publish_setpoint()

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
                self._timeout_to_land()
                return

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
                self._set_phase(
                    Phase.TAKEOFF,
                    "TAKEOFF: stable takeoff reference locked, climbing "
                    "0.45 m above home.",
                )
            return

        if self.phase == Phase.TAKEOFF:
            if self._takeoff_ready():
                self.target = self._target_from_home(self.TEST_DX_M, self.TEST_DY_M)
                self._set_phase(
                    Phase.MOVE_LOCAL_XY_TEST_POINT,
                    "MOVE_LOCAL_XY_TEST_POINT: moving to local direction test point.",
                )
            return

        if self.phase == Phase.MOVE_LOCAL_XY_TEST_POINT:
            if self._horizontal_response_watchdog_to_land():
                return
            if self._stable_at_target():
                self._set_phase(Phase.HOLD, "HOLD: holding test point for 5 seconds.")
            return

        if self.phase == Phase.HOLD:
            if self._hold_complete_after_reach():
                self.request_land()
                self._set_phase(Phase.LAND, "LAND: direction verification complete.")
            return


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OffboardWideGateTraversalV1()
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
