#!/usr/bin/env python3
"""
Geometry-based, position-controlled single traversal through the LEFT opening.

IMPORTANT SAFETY / ASSUMPTION
-----------------------------
This program is only valid when Gazebo/PX4 are launched with the existing
known spawn pose used in this project:

PX4_GZ_MODEL_POSE="1,1,0.1,0,0,0.9"

The route is calculated from the extracted left-frame opening geometry:

  opening center in Gazebo world (ENU):
      X = 0.5395 m
      Y = 3.2277 m
      Z = 1.3527 m

  opening clear size:
      width = 0.8014 m
      height = 0.8014 m

Gazebo world ENU -> PX4 local NED:
      PX4 local X (North) = Gazebo world Y displacement
      PX4 local Y (East)  = Gazebo world X displacement
      PX4 local Z (Down)  = - Gazebo world Z displacement

The program DOES NOT use velocity-only crossing. It continuously sends
position setpoints for all three axes, so lateral and vertical alignment stay
locked throughout every crossing step.

Route:
  WAIT_POSITION
  -> TAKEOFF_TO_GATE_HEIGHT
  -> ALIGN_TO_GATE_CENTERLINE
  -> PRE_GATE_STABLE
  -> CROSS_STEP_1 ... CROSS_STEP_6
  -> POST_GATE_HOLD
  -> LANDING
  -> FINISHED

Safety behaviour:
  * It will not begin crossing until it is stably aligned at the pre-gate point.
  * During crossing, if lateral (Y) or vertical (Z) error exceeds a strict
    corridor limit, it stops at the current point and lands rather than
    continuing toward the frame.
  * Ctrl+C requests an AUTO LAND command.

This file is intentionally independent from prior experimental scripts.
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

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
    WAIT_POSITION = auto()
    TAKEOFF_TO_GATE_HEIGHT = auto()
    ALIGN_TO_GATE_CENTERLINE = auto()
    PRE_GATE_STABLE = auto()
    CROSSING = auto()
    POST_GATE_HOLD = auto()
    LANDING = auto()
    FINISHED = auto()
    ABORT_HOLD = auto()


class GeometryLeftGateTraversal(Node):
    # -----------------------------------------------------------------
    # Extracted static geometry: do not change these without re-running
    # the mesh analysis.
    # -----------------------------------------------------------------
    SPAWN_WORLD_X = 1.0000
    SPAWN_WORLD_Y = 1.0000
    GATE_CENTER_WORLD_X = 0.5395
    GATE_CENTER_WORLD_Y = 3.2277
    GATE_CENTER_WORLD_Z = 1.3527
    GATE_MIN_WORLD_Y = 3.0777
    GATE_MAX_WORLD_Y = 3.3777
    OPENING_WIDTH = 0.8014
    OPENING_HEIGHT = 0.8014

    # Route geometry in Gazebo world coordinates.
    # World Y is PX4 local X after ENU -> NED conversion.
    PRE_GATE_WORLD_Y = 2.6500
    POST_GATE_WORLD_Y = 3.8500

    # These tolerances are deliberately conservative relative to 0.8014 m
    # opening dimensions and the x500 rotor span (~0.35 m).
    POSITION_REACHED_TOL = 0.08
    PRE_GATE_STABLE_SECONDS = 2.5
    STEP_STABLE_SECONDS = 0.8
    POST_GATE_HOLD_SECONDS = 5.0

    # Only lateral / vertical axes are safety-critical during a pass.
    CROSSING_LATERAL_LIMIT = 0.12
    CROSSING_VERTICAL_LIMIT = 0.12
    CROSSING_ERROR_TIMEOUT = 0.50

    TIMER_PERIOD = 0.10
    PRESTREAM_TICKS = 15

    def __init__(self) -> None:
        super().__init__("offboard_left_gate_single_traversal")

        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10
        )

        # PX4 uXRCE-DDS publishes FMU output topics with BEST_EFFORT reliability.
        # A default ROS 2 subscription requests RELIABLE, which is incompatible
        # and causes the node to receive no telemetry.
        px4_out_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self._local_position_callback,
            px4_out_qos,
        )
        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self._vehicle_status_callback,
            px4_out_qos,
        )

        self.timer = self.create_timer(self.TIMER_PERIOD, self._timer_callback)

        self.local_position: Optional[VehicleLocalPosition] = None
        self.vehicle_status: Optional[VehicleStatus] = None

        self.home: Optional[Tuple[float, float, float]] = None
        self.target: Optional[Tuple[float, float, float]] = None
        self.abort_hold_target: Optional[Tuple[float, float, float]] = None

        self.phase = Phase.WAIT_POSITION
        self.phase_started = time.monotonic()
        self.last_progress_log = 0.0
        self.offboard_setpoint_counter = 0
        self.offboard_requested = False
        self.arm_requested = False

        self.crossing_steps: List[float] = []
        self.crossing_index = 0
        self.stable_since: Optional[float] = None
        self.corridor_error_since: Optional[float] = None
        self.land_requested = False

        self.get_logger().info(
            "Geometry-based LEFT-GATE traversal node started. "
            "No crossing will occur until pre-gate alignment is stable."
        )

    # ------------------------------- PX4 I/O -------------------------------

    def _timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _local_position_callback(self, msg: VehicleLocalPosition) -> None:
        self.local_position = msg

    def _vehicle_status_callback(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def _publish_offboard_heartbeat(self) -> None:
        msg = OffboardControlMode()
        msg.timestamp = self._timestamp_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_pub.publish(msg)

    def _publish_position_setpoint(self, target: Tuple[float, float, float]) -> None:
        msg = TrajectorySetpoint()
        msg.timestamp = self._timestamp_us()
        msg.position = [float(target[0]), float(target[1]), float(target[2])]
        msg.velocity = [float("nan"), float("nan"), float("nan")]
        msg.acceleration = [float("nan"), float("nan"), float("nan")]
        msg.yaw = float("nan")
        msg.yawspeed = float("nan")
        self.trajectory_setpoint_pub.publish(msg)

    def _send_vehicle_command(self, command: int, **params: float) -> None:
        msg = VehicleCommand()
        msg.timestamp = self._timestamp_us()
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True

        for index in range(1, 8):
            setattr(msg, f"param{index}", float(params.get(f"param{index}", 0.0)))

        self.vehicle_command_pub.publish(msg)

    def _request_offboard(self) -> None:
        self._send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )
        self.offboard_requested = True
        self.get_logger().info("Requested OFFBOARD mode.")

    def _request_arm(self) -> None:
        self._send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )
        self.arm_requested = True
        self.get_logger().info("Requested ARM.")

    def _request_land(self) -> None:
        if self.land_requested:
            return
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.land_requested = True
        self.get_logger().info("Requested AUTO LAND.")

    # ------------------------- Geometry / route maths -----------------------

    def _gate_center_local_target(self) -> Tuple[float, float, float]:
        """
        Convert extracted Gazebo world opening center to PX4 local NED,
        anchored at the captured PX4 local home position.

        x_local = home_x + (world_y - spawn_world_y)
        y_local = home_y + (world_x - spawn_world_x)
        z_local = home_z - world_z
        """
        assert self.home is not None
        hx, hy, hz = self.home

        return (
            hx + (self.GATE_CENTER_WORLD_Y - self.SPAWN_WORLD_Y),
            hy + (self.GATE_CENTER_WORLD_X - self.SPAWN_WORLD_X),
            hz - self.GATE_CENTER_WORLD_Z,
        )

    def _local_x_for_world_y(self, world_y: float) -> float:
        assert self.home is not None
        return self.home[0] + (world_y - self.SPAWN_WORLD_Y)

    def _build_route(self) -> None:
        """
        Every route waypoint keeps gate-centre lateral and height coordinates
        fixed. Only PX4 local X changes during the pass through the frame.
        """
        gate_x, gate_y, gate_z = self._gate_center_local_target()

        pre_x = self._local_x_for_world_y(self.PRE_GATE_WORLD_Y)
        post_x = self._local_x_for_world_y(self.POST_GATE_WORLD_Y)

        # Small position-controlled increments across the gate plane.
        # Gate plane in local X:
        #   [world_y 3.0777, 3.3777] -> [home + 2.0777, home + 2.3777]
        first = pre_x
        step_count = 6
        self.crossing_steps = [
            first + (post_x - first) * (i / step_count)
            for i in range(1, step_count + 1)
        ]

        self.target = (self.home[0], self.home[1], gate_z)

        self.get_logger().info(
            "Extracted left opening (Gazebo ENU): "
            f"center=({self.GATE_CENTER_WORLD_X:.4f}, "
            f"{self.GATE_CENTER_WORLD_Y:.4f}, "
            f"{self.GATE_CENTER_WORLD_Z:.4f}), "
            f"size={self.OPENING_WIDTH:.4f} x {self.OPENING_HEIGHT:.4f} m"
        )
        self.get_logger().info(
            "Derived PX4 local NED gate center: "
            f"x={gate_x:.3f}, y={gate_y:.3f}, z={gate_z:.3f}"
        )
        self.get_logger().info(
            f"Pre-gate target: x={pre_x:.3f}, y={gate_y:.3f}, z={gate_z:.3f}"
        )
        self.get_logger().info(
            f"Post-gate target: x={post_x:.3f}, y={gate_y:.3f}, z={gate_z:.3f}"
        )

    # ------------------------------ state logic -----------------------------

    def _set_phase(self, phase: Phase, message: str) -> None:
        self.phase = phase
        self.phase_started = time.monotonic()
        self.stable_since = None
        self.corridor_error_since = None
        self.get_logger().info(message)

    def _position_xyz(self) -> Tuple[float, float, float]:
        assert self.local_position is not None
        return (
            float(self.local_position.x),
            float(self.local_position.y),
            float(self.local_position.z),
        )

    def _distance_to_target(self, target: Tuple[float, float, float]) -> float:
        x, y, z = self._position_xyz()
        return math.sqrt(
            (x - target[0]) ** 2
            + (y - target[1]) ** 2
            + (z - target[2]) ** 2
        )

    def _within_target(self, target: Tuple[float, float, float], tol: float) -> bool:
        return self._distance_to_target(target) <= tol

    def _stable_at_target(
        self,
        target: Tuple[float, float, float],
        tolerance: float,
        seconds: float,
    ) -> bool:
        now = time.monotonic()

        if self._within_target(target, tolerance):
            if self.stable_since is None:
                self.stable_since = now
            return (now - self.stable_since) >= seconds

        self.stable_since = None
        return False

    def _log_progress(self, label: str, target: Tuple[float, float, float]) -> None:
        now = time.monotonic()
        if now - self.last_progress_log < 1.0:
            return

        x, y, z = self._position_xyz()
        dist = self._distance_to_target(target)
        self.get_logger().info(
            f"{label} | Current: ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"| Goal: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) "
            f"| Distance: {dist:.3f} m"
        )
        self.last_progress_log = now

    def _crossing_corridor_safe(self, gate_y: float, gate_z: float) -> bool:
        """
        During a pass, the along-route X error is expected while moving.
        We only judge lateral (local Y) and vertical (local Z) alignment.
        """
        _, current_y, current_z = self._position_xyz()
        lateral_error = abs(current_y - gate_y)
        vertical_error = abs(current_z - gate_z)

        return (
            lateral_error <= self.CROSSING_LATERAL_LIMIT
            and vertical_error <= self.CROSSING_VERTICAL_LIMIT
        )

    def _abort_crossing(self, reason: str) -> None:
        self.abort_hold_target = self._position_xyz()
        self._set_phase(
            Phase.ABORT_HOLD,
            "SAFETY ABORT: "
            + reason
            + ". Holding current position, then landing.",
        )

    def _timer_callback(self) -> None:
        if self.local_position is None:
            if self.phase == Phase.WAIT_POSITION:
                self.get_logger().info(
                    "Waiting for /fmu/out/vehicle_local_position ..."
                )
            return

        # Capture a local reference once. It makes the route robust to the
        # normal few-centimetre PX4 local-origin offset at spawn.
        if self.home is None:
            self.home = self._position_xyz()
            self._build_route()
            self.get_logger().info(
                f"Captured PX4 local home: x={self.home[0]:.3f}, "
                f"y={self.home[1]:.3f}, z={self.home[2]:.3f}"
            )

        assert self.target is not None
        self._publish_offboard_heartbeat()
        self._publish_position_setpoint(self.target)

        if self.phase == Phase.WAIT_POSITION:
            self.offboard_setpoint_counter += 1
            self._log_progress("PRESTREAM", self.target)

            if self.offboard_setpoint_counter >= self.PRESTREAM_TICKS:
                self._request_offboard()
                self._request_arm()
                self._set_phase(
                    Phase.TAKEOFF_TO_GATE_HEIGHT,
                    "TAKEOFF_TO_GATE_HEIGHT: climbing at home position "
                    "to the extracted gate-center height.",
                )
            return

        if self.phase == Phase.TAKEOFF_TO_GATE_HEIGHT:
            self._log_progress("TAKEOFF", self.target)

            if self._stable_at_target(
                self.target,
                self.POSITION_REACHED_TOL,
                self.PRE_GATE_STABLE_SECONDS,
            ):
                gate_x, gate_y, gate_z = self._gate_center_local_target()
                self.target = (self.home[0], gate_y, gate_z)
                self._set_phase(
                    Phase.ALIGN_TO_GATE_CENTERLINE,
                    "ALIGN_TO_GATE_CENTERLINE: moving laterally in open space "
                    "to the left-opening centerline.",
                )
            return

        if self.phase == Phase.ALIGN_TO_GATE_CENTERLINE:
            self._log_progress("ALIGN_CENTERLINE", self.target)

            if self._stable_at_target(
                self.target,
                self.POSITION_REACHED_TOL,
                self.PRE_GATE_STABLE_SECONDS,
            ):
                _, gate_y, gate_z = self._gate_center_local_target()
                self.target = (
                    self._local_x_for_world_y(self.PRE_GATE_WORLD_Y),
                    gate_y,
                    gate_z,
                )
                self._set_phase(
                    Phase.PRE_GATE_STABLE,
                    "PRE_GATE_APPROACH: moving to the safe pre-gate position.",
                )
            return

        if self.phase == Phase.PRE_GATE_STABLE:
            self._log_progress("PRE_GATE", self.target)

            if self._stable_at_target(
                self.target,
                self.POSITION_REACHED_TOL,
                self.PRE_GATE_STABLE_SECONDS,
            ):
                self.crossing_index = 0
                _, gate_y, gate_z = self._gate_center_local_target()
                self.target = (
                    self.crossing_steps[self.crossing_index],
                    gate_y,
                    gate_z,
                )
                self._set_phase(
                    Phase.CROSSING,
                    "CROSSING START: position-control steps are enabled. "
                    "Lateral and vertical corridor checks are active.",
                )
            return

        if self.phase == Phase.CROSSING:
            _, gate_y, gate_z = self._gate_center_local_target()

            if self._crossing_corridor_safe(gate_y, gate_z):
                self.corridor_error_since = None
            else:
                now = time.monotonic()
                if self.corridor_error_since is None:
                    self.corridor_error_since = now
                elif now - self.corridor_error_since >= self.CROSSING_ERROR_TIMEOUT:
                    self._abort_crossing(
                        f"alignment deviation beyond safe corridor "
                        f"(limit Y/Z = {self.CROSSING_LATERAL_LIMIT:.2f}/"
                        f"{self.CROSSING_VERTICAL_LIMIT:.2f} m)"
                    )
                    return

            self._log_progress(
                f"CROSS_STEP_{self.crossing_index + 1}/{len(self.crossing_steps)}",
                self.target,
            )

            if self._stable_at_target(
                self.target,
                self.POSITION_REACHED_TOL,
                self.STEP_STABLE_SECONDS,
            ):
                self.crossing_index += 1

                if self.crossing_index >= len(self.crossing_steps):
                    self._set_phase(
                        Phase.POST_GATE_HOLD,
                        "POST_GATE_HOLD: traversal steps completed. "
                        "Holding beyond the frame before landing.",
                    )
                else:
                    self.target = (
                        self.crossing_steps[self.crossing_index],
                        gate_y,
                        gate_z,
                    )
                    self.stable_since = None
                    self.get_logger().info(
                        f"Advancing to crossing step "
                        f"{self.crossing_index + 1}/{len(self.crossing_steps)}."
                    )
            return

        if self.phase == Phase.POST_GATE_HOLD:
            self._log_progress("POST_GATE_HOLD", self.target)

            if time.monotonic() - self.phase_started >= self.POST_GATE_HOLD_SECONDS:
                self._request_land()
                self._set_phase(
                    Phase.LANDING,
                    "LANDING: AUTO LAND requested after successful pass.",
                )
            return

        if self.phase == Phase.ABORT_HOLD:
            assert self.abort_hold_target is not None
            self.target = self.abort_hold_target
            self._log_progress("ABORT_HOLD", self.target)

            if time.monotonic() - self.phase_started >= 2.0:
                self._request_land()
                self._set_phase(
                    Phase.LANDING,
                    "LANDING: AUTO LAND requested after safety abort.",
                )
            return

        if self.phase == Phase.LANDING:
            # Continue setpoint/heartbeat publishing while PX4 transitions.
            # Completion is based on disarm state if available.
            self._log_progress("LANDING", self.target)

            if self.vehicle_status is not None:
                if (
                    self.vehicle_status.arming_state
                    == VehicleStatus.ARMING_STATE_DISARMED
                ):
                    self._set_phase(
                        Phase.FINISHED,
                        "Vehicle disarmed. Left-gate traversal mission finished.",
                    )
            return

        if self.phase == Phase.FINISHED:
            return

    def emergency_land_on_shutdown(self) -> None:
        if self.phase not in (Phase.FINISHED, Phase.LANDING):
            self.get_logger().warning(
                "Shutdown requested: asking PX4 for AUTO LAND."
            )
            self._request_land()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GeometryLeftGateTraversal()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Do not try to send AUTO LAND when no telemetry was ever received:
        # the vehicle was never armed by this node. This also avoids ROS context
        # shutdown races that can produce a noisy traceback after Ctrl+C.
        if node.local_position is not None and node.phase not in (Phase.FINISHED, Phase.LANDING):
            try:
                node.emergency_land_on_shutdown()
                time.sleep(0.3)
            except Exception:
                pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
