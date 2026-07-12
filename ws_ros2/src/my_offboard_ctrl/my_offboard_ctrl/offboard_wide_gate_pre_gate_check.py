#!/usr/bin/env python3
"""
Stage 1 for the retained wide-gate scene.

Purpose:
  Confirm the coordinate direction and safe pre-gate position only.
  It DOES NOT attempt to pass through the gate.

Route (all relative to the PX4 local position captured before arming):
  1. Take off vertically by 1.0 m.
  2. Move left by 0.46 m in local Y to align with the enlarged left opening.
  3. Move forward by 1.35 m in local X and stop well before the gate.
  4. Hold for 8 seconds, then AUTO LAND.

Important:
  - No fixed yaw command is sent. PX4 keeps its own yaw, avoiding the
    previous yaw-estimate / forced-yaw problem.
  - The test uses only relative local displacements, not Gazebo absolute Z.
  - It is intentionally a "go to pre-gate and stop" test, not a traversal.
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)


class Phase(Enum):
    WAIT_FOR_POSE = auto()
    PRESTREAM = auto()
    TAKEOFF = auto()
    LATERAL_ALIGN = auto()
    PRE_GATE = auto()
    HOLD = auto()
    LAND = auto()
    DONE = auto()


class WideGatePreGateCheck(Node):
    TIMER_PERIOD = 0.10
    PRESTREAM_TICKS = 20

    # Conservative relative route.
    TAKEOFF_UP_M = 1.00
    LEFT_LOCAL_Y_M = -0.46
    FORWARD_LOCAL_X_M = 1.35

    POSITION_TOL_M = 0.12
    STABLE_SECONDS = 2.0
    HOLD_SECONDS = 8.0
    PHASE_TIMEOUT_SECONDS = 35.0

    def __init__(self) -> None:
        super().__init__("offboard_wide_gate_pre_gate_check")

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10
        )

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self._on_local_position,
            px4_qos,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self._on_vehicle_status,
            px4_qos,
        )

        self.timer = self.create_timer(self.TIMER_PERIOD, self._tick)

        self.local_position: Optional[VehicleLocalPosition] = None
        self.vehicle_status: Optional[VehicleStatus] = None
        self.home: Optional[Tuple[float, float, float]] = None
        self.target: Optional[Tuple[float, float, float]] = None

        self.phase = Phase.WAIT_FOR_POSE
        self.phase_started = time.monotonic()
        self.stable_since: Optional[float] = None
        self.last_log = 0.0
        self.prestream_count = 0
        self.offboard_requested = False
        self.arm_requested = False
        self.land_requested = False

        self.get_logger().info(
            "Wide-gate Stage 1 started: takeoff -> left alignment -> pre-gate hold. "
            "No gate crossing will be attempted."
        )

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        self.local_position = msg

    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def _pos(self) -> Tuple[float, float, float]:
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

    def _publish_setpoint(self, target: Tuple[float, float, float]) -> None:
        msg = TrajectorySetpoint()
        msg.timestamp = self._now_us()
        msg.position = [float(target[0]), float(target[1]), float(target[2])]
        msg.velocity = [math.nan, math.nan, math.nan]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        # Keep yaw uncontrolled: no forced heading in this stage.
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

    def _request_land(self) -> None:
        if not self.land_requested:
            self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.land_requested = True
            self.get_logger().info("Requested AUTO LAND.")

    def _set_phase(self, phase: Phase, message: str) -> None:
        self.phase = phase
        self.phase_started = time.monotonic()
        self.stable_since = None
        self.get_logger().info(message)

    def _distance_to_target(self) -> float:
        assert self.target is not None
        x, y, z = self._pos()
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

    def _timed_out(self) -> bool:
        return time.monotonic() - self.phase_started > self.PHASE_TIMEOUT_SECONDS

    def _log(self, label: str) -> None:
        if self.target is None or time.monotonic() - self.last_log < 1.0:
            return
        x, y, z = self._pos()
        self.get_logger().info(
            f"{label} | current=({x:.3f}, {y:.3f}, {z:.3f}) "
            f"| goal=({self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f}) "
            f"| d={self._distance_to_target():.3f}m"
        )
        self.last_log = time.monotonic()

    def _tick(self) -> None:
        if self.local_position is None:
            return

        if self.home is None:
            self.home = self._pos()
            self.target = self.home
            self.get_logger().info(
                f"Captured local home: ({self.home[0]:.3f}, "
                f"{self.home[1]:.3f}, {self.home[2]:.3f})"
            )

        assert self.home is not None
        assert self.target is not None

        self._publish_heartbeat()
        self._publish_setpoint(self.target)

        if self.phase == Phase.WAIT_FOR_POSE:
            self._set_phase(Phase.PRESTREAM, "PRESTREAM: sending position setpoints.")
            return

        if self.phase == Phase.PRESTREAM:
            self.prestream_count += 1
            self._log("PRESTREAM")
            if self.prestream_count >= self.PRESTREAM_TICKS:
                self._request_offboard_and_arm()
                self.target = (
                    self.home[0],
                    self.home[1],
                    self.home[2] - self.TAKEOFF_UP_M,
                )
                self._set_phase(
                    Phase.TAKEOFF,
                    "TAKEOFF: rising 1.0 m at the start point.",
                )
            return

        if self.phase == Phase.TAKEOFF:
            self._log("TAKEOFF")
            if self._stable_at_target():
                self.target = (
                    self.home[0],
                    self.home[1] + self.LEFT_LOCAL_Y_M,
                    self.home[2] - self.TAKEOFF_UP_M,
                )
                self._set_phase(
                    Phase.LATERAL_ALIGN,
                    "LATERAL_ALIGN: moving left 0.46 m in open space.",
                )
            elif self._timed_out():
                self._request_land()
                self._set_phase(Phase.LAND, "TAKEOFF timeout: landing safely.")
            return

        if self.phase == Phase.LATERAL_ALIGN:
            self._log("LATERAL_ALIGN")
            if self._stable_at_target():
                self.target = (
                    self.home[0] + self.FORWARD_LOCAL_X_M,
                    self.home[1] + self.LEFT_LOCAL_Y_M,
                    self.home[2] - self.TAKEOFF_UP_M,
                )
                self._set_phase(
                    Phase.PRE_GATE,
                    "PRE_GATE: moving forward 1.35 m, stopping well before the frame.",
                )
            elif self._timed_out():
                self._request_land()
                self._set_phase(Phase.LAND, "LATERAL_ALIGN timeout: landing safely.")
            return

        if self.phase == Phase.PRE_GATE:
            self._log("PRE_GATE")
            if self._stable_at_target():
                self._set_phase(
                    Phase.HOLD,
                    "PRE_GATE HOLD: Stage 1 succeeded. Holding for 8 seconds.",
                )
            elif self._timed_out():
                self._request_land()
                self._set_phase(Phase.LAND, "PRE_GATE timeout: landing safely.")
            return

        if self.phase == Phase.HOLD:
            self._log("HOLD")
            if time.monotonic() - self.phase_started >= self.HOLD_SECONDS:
                self._request_land()
                self._set_phase(Phase.LAND, "LAND: Stage 1 complete, landing.")
            return

        if self.phase == Phase.LAND:
            self._log("LAND")
            if (
                self.vehicle_status is not None
                and self.vehicle_status.arming_state
                == VehicleStatus.ARMING_STATE_DISARMED
            ):
                self._set_phase(Phase.DONE, "DONE: vehicle disarmed.")
            return

    def request_land_on_exit(self) -> None:
        if self.local_position is not None and not self.land_requested:
            self._request_land()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WideGatePreGateCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.request_land_on_exit()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
