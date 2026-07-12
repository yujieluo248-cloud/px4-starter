#!/usr/bin/env python3
"""
Conservative PX4 local +X direction calibration.

This node only uses PX4 local relative coordinates. It does not use Gazebo
world coordinates, gate coordinates, or velocity-only control.
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
    VehicleLocalPosition,
    VehicleStatus,
)


class Phase(Enum):
    WAIT_FOR_POSITION = auto()
    PRESTREAM = auto()
    TAKEOFF_LOW = auto()
    MOVE_POS_X_SMALL = auto()
    HOLD_AFTER_X = auto()
    LAND = auto()
    DONE = auto()


class OffboardLocalXOnlyCalibration(Node):
    TIMER_PERIOD_S = 0.10
    PRESTREAM_TICKS = 20
    TAKEOFF_UP_M = 0.45
    X_STEP_M = 0.25
    POSITION_TOL_M = 0.20
    STABLE_SECONDS = 0.60
    HOLD_AFTER_X_SECONDS = 2.00
    PHASE_TIMEOUT_SECONDS = 35.00

    def __init__(self) -> None:
        super().__init__("offboard_local_x_only_calibration")

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

        self.local_position: Optional[VehicleLocalPosition] = None
        self.vehicle_status: Optional[VehicleStatus] = None
        self.home: Optional[Tuple[float, float, float]] = None
        self.target: Optional[Tuple[float, float, float]] = None

        self.phase = Phase.WAIT_FOR_POSITION
        self.phase_started = time.monotonic()
        self.stable_since: Optional[float] = None
        self.last_log = 0.0
        self.prestream_count = 0

        self.offboard_requested = False
        self.arm_requested = False
        self.land_requested = False

        self.timer = self.create_timer(self.TIMER_PERIOD_S, self._tick)

        self.get_logger().info(
            "Local +X-only calibration started. Low altitude, small X step only."
        )

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        self.local_position = msg

    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

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
        )

    def _position(self) -> Tuple[float, float, float]:
        assert self.local_position is not None
        return (
            float(self.local_position.x),
            float(self.local_position.y),
            float(self.local_position.z),
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

    def _request_offboard_and_arm(self) -> None:
        if not self.offboard_requested:
            self._command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1.0,
                param2=6.0,
            )
            self.offboard_requested = True
            self.get_logger().info("Requested OFFBOARD mode.")

        if not self.arm_requested:
            self._command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=1.0,
            )
            self.arm_requested = True
            self.get_logger().info("Requested ARM.")

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
        self.get_logger().info(message)

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
        return False

    def _phase_timed_out(self) -> bool:
        return time.monotonic() - self.phase_started > self.PHASE_TIMEOUT_SECONDS

    def _relative_height(self) -> float:
        if self.home is None or self.local_position is None:
            return math.nan
        return self.home[2] - self.local_position.z

    def _log_status(self) -> None:
        now = time.monotonic()
        if now - self.last_log < 1.0:
            return

        if self.local_position is None:
            self.get_logger().info(f"{self.phase.name}: waiting for local position.")
            self.last_log = now
            return

        x, y, z = self._position()
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
            f"rel_height={self._relative_height():.3f} m"
        )
        self.last_log = now

    def _timeout_to_land(self) -> None:
        self.get_logger().error(
            f"{self.phase.name} timed out after "
            f"{self.PHASE_TIMEOUT_SECONDS:.1f} s. Requesting AUTO LAND."
        )
        self.request_land()
        self._set_phase(Phase.LAND, "LAND: timeout failsafe.")

    def _capture_home(self) -> None:
        self.home = self._position()
        self.target = self.home
        self._set_phase(
            Phase.PRESTREAM,
            "PRESTREAM: sending home setpoints before OFFBOARD / ARM.",
        )
        self.get_logger().info(
            f"Captured home=({self.home[0]:.3f}, {self.home[1]:.3f}, "
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

        self._publish_heartbeat()
        self._publish_setpoint()

        if self.phase not in (Phase.LAND, Phase.DONE) and self._phase_timed_out():
            self._timeout_to_land()
            return

        if self.phase == Phase.PRESTREAM:
            self.prestream_count += 1
            if self.prestream_count >= self.PRESTREAM_TICKS:
                assert self.home is not None
                self._request_offboard_and_arm()
                self.target = (
                    self.home[0],
                    self.home[1],
                    self.home[2] - self.TAKEOFF_UP_M,
                )
                self._set_phase(
                    Phase.TAKEOFF_LOW,
                    "TAKEOFF_LOW: climbing 0.45 m above home.",
                )
            return

        if self.phase == Phase.TAKEOFF_LOW:
            if self._stable_at_target():
                assert self.home is not None
                self.target = (
                    self.home[0] + self.X_STEP_M,
                    self.home[1],
                    self.home[2] - self.TAKEOFF_UP_M,
                )
                self._set_phase(
                    Phase.MOVE_POS_X_SMALL,
                    "MOVE_POS_X_SMALL: moving local +X by 0.25 m.",
                )
            return

        if self.phase == Phase.MOVE_POS_X_SMALL:
            if self._stable_at_target():
                self._set_phase(
                    Phase.HOLD_AFTER_X,
                    "HOLD_AFTER_X: holding at +X target for 2 seconds.",
                )
            return

        if self.phase == Phase.HOLD_AFTER_X:
            if time.monotonic() - self.phase_started >= self.HOLD_AFTER_X_SECONDS:
                self.request_land()
                self._set_phase(Phase.LAND, "LAND: X-only calibration complete.")
            return

        if self.phase == Phase.LAND:
            if (
                self.vehicle_status is not None
                and self.vehicle_status.arming_state
                == VehicleStatus.ARMING_STATE_DISARMED
            ):
                self._set_phase(Phase.DONE, "DONE: vehicle disarmed.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OffboardLocalXOnlyCalibration()
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
