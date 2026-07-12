#!/usr/bin/env python3
"""
PX4 local-axis calibration node.

Purpose:
  Confirm which directions Gazebo shows for PX4 local +X and +Y.

Route:
  1. Capture the current PX4 local position as home.
  2. Pre-stream at least 20 position setpoints before OFFBOARD / ARM.
  3. Take off 0.80 m above home.
  4. Move local +X by 0.40 m, then return.
  5. Move local +Y by 0.40 m, then return.
  6. Hold for 4 seconds, then AUTO LAND.
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
    TAKEOFF = auto()
    MOVE_POS_X = auto()
    RETURN_FROM_X = auto()
    MOVE_POS_Y = auto()
    RETURN_FROM_Y = auto()
    HOLD = auto()
    LAND = auto()
    DONE = auto()


class OffboardLocalAxisCalibration(Node):
    TIMER_PERIOD_S = 0.10
    PRESTREAM_TICKS = 20
    TAKEOFF_UP_M = 0.80
    AXIS_STEP_M = 0.40
    POSITION_TOL_M = 0.18
    STABLE_SECONDS = 0.80
    HOLD_SECONDS = 4.00
    PHASE_TIMEOUT_SECONDS = 45.00

    def __init__(self) -> None:
        super().__init__("offboard_local_axis_calibration")

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
        self.hover_origin: Optional[Tuple[float, float, float]] = None
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
            "Local-axis calibration started. Waiting for "
            "/fmu/out/vehicle_local_position."
        )

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        self.local_position = msg

    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

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
        msg = TrajectorySetpoint()
        msg.timestamp = self._now_us()
        msg.position = [
            float(self.target[0]),
            float(self.target[1]),
            float(self.target[2]),
        ]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        msg.yaw = math.nan
        msg.yawspeed = math.nan
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
            self.get_logger().info(
                f"{self.phase.name}: current=({x:.3f}, {y:.3f}, {z:.3f}), "
                "target=(none), error=nan"
            )
        else:
            self.get_logger().info(
                f"{self.phase.name}: current=({x:.3f}, {y:.3f}, {z:.3f}), "
                f"target=({self.target[0]:.3f}, {self.target[1]:.3f}, "
                f"{self.target[2]:.3f}), error={self._distance_to_target():.3f} m"
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
        self.hover_origin = (
            self.home[0],
            self.home[1],
            self.home[2] - self.TAKEOFF_UP_M,
        )
        self.target = self.home
        self._set_phase(
            Phase.PRESTREAM,
            "PRESTREAM: sending home position setpoints before OFFBOARD / ARM.",
        )
        self.get_logger().info(
            f"Captured home=({self.home[0]:.3f}, {self.home[1]:.3f}, "
            f"{self.home[2]:.3f}); hover_origin=({self.hover_origin[0]:.3f}, "
            f"{self.hover_origin[1]:.3f}, {self.hover_origin[2]:.3f})."
        )

    def _tick(self) -> None:
        self._log_status()
        if self.phase == Phase.DONE:
            return
        if self.phase == Phase.WAIT_FOR_POSITION:
            if self.local_position is None:
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
                assert self.hover_origin is not None
                self._request_offboard_and_arm()
                self.target = self.hover_origin
                self._set_phase(Phase.TAKEOFF, "TAKEOFF: climbing 0.80 m above home.")
            return

        if self.phase == Phase.TAKEOFF:
            if self._stable_at_target():
                assert self.hover_origin is not None
                self.target = (
                    self.hover_origin[0] + self.AXIS_STEP_M,
                    self.hover_origin[1],
                    self.hover_origin[2],
                )
                self._set_phase(Phase.MOVE_POS_X, "MOVE_POS_X: moving +0.40 m X.")
            return

        if self.phase == Phase.MOVE_POS_X:
            if self._stable_at_target():
                assert self.hover_origin is not None
                self.target = self.hover_origin
                self._set_phase(Phase.RETURN_FROM_X, "RETURN_FROM_X: hover origin.")
            return

        if self.phase == Phase.RETURN_FROM_X:
            if self._stable_at_target():
                assert self.hover_origin is not None
                self.target = (
                    self.hover_origin[0],
                    self.hover_origin[1] + self.AXIS_STEP_M,
                    self.hover_origin[2],
                )
                self._set_phase(Phase.MOVE_POS_Y, "MOVE_POS_Y: moving +0.40 m Y.")
            return

        if self.phase == Phase.MOVE_POS_Y:
            if self._stable_at_target():
                assert self.hover_origin is not None
                self.target = self.hover_origin
                self._set_phase(Phase.RETURN_FROM_Y, "RETURN_FROM_Y: hover origin.")
            return

        if self.phase == Phase.RETURN_FROM_Y:
            if self._stable_at_target():
                self._set_phase(Phase.HOLD, "HOLD: holding for 4 seconds.")
            return

        if self.phase == Phase.HOLD:
            if time.monotonic() - self.phase_started >= self.HOLD_SECONDS:
                self.request_land()
                self._set_phase(Phase.LAND, "LAND: calibration complete.")
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
    node = OffboardLocalAxisCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C received. Requesting AUTO LAND.")
        node.request_land()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
