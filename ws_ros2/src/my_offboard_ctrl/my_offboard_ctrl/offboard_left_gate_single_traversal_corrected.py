#!/usr/bin/env python3
"""
Conservative, geometry-corrected one-way traversal of the LEFT gate opening.

This revision corrects two issues found from the actual Gazebo collision model:

1) The X500 collision envelope is not centered at the Gazebo model root.
   Its collision geometry spans approximately:
       root_z + 0.013 m  to root_z + 0.300 m
   so root_z is commanded to 1.196 m, not to the opening center 1.353 m.

2) The X500 collision footprint is ~0.627 m wide when aligned with the gate.
   The 0.801 m opening then leaves only about 0.087 m clearance per side.
   The vehicle must not cross with the initial yaw ~= 0.9 rad, because the
   rotated footprint becomes wider than the opening.

This node therefore:
  - rotates to yaw = 0.0 rad in open space,
  - verifies yaw and centerline alignment before approaching the gate,
  - uses 24 short full-position-control steps (no velocity-only crossing),
  - aborts BEFORE advancing if lateral / vertical / yaw safety checks fail,
  - never relies on collision to "discover" the route.

Launch assumptions (same as the successful prior setup):
  PX4_GZ_MODEL_POSE="1,1,0.1,0,0,0.9"
  PX4_GZ_WORLD=test_world
  PX4_SYS_AUTOSTART=4010
  MicroXRCEAgent udp4 -p 8888

NOTE:
This is still a simulation experiment. It is conservative, but it cannot
guarantee collision-free execution without a real-time collision-distance
sensor. If it aborts, do not repeatedly rerun it; inspect the log first.
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
    WAIT_TELEMETRY = auto()
    TAKEOFF_TO_SAFE_ROOT_HEIGHT = auto()
    YAW_ALIGN_OPEN_SPACE = auto()
    LATERAL_ALIGN_OPEN_SPACE = auto()
    MOVE_TO_PRE_GATE = auto()
    PRE_GATE_VERIFY = auto()
    CROSSING = auto()
    POST_GATE_HOLD = auto()
    ABORT_HOLD = auto()
    LANDING = auto()
    FINISHED = auto()


class LeftGateTraversalCorrected(Node):
    # ----- Extracted left opening (Gazebo world coordinates) -----
    GATE_CENTER_WORLD_X = 0.5395
    GATE_CENTER_WORLD_Y = 3.2277
    GATE_CENTER_WORLD_Z = 1.3527
    GATE_MIN_WORLD_Y = 3.0777
    GATE_MAX_WORLD_Y = 3.3777
    OPENING_WIDTH = 0.8014
    OPENING_HEIGHT = 0.8014

    # ----- Known clean spawn, read from the Gazebo pose topic after reset -----
    SPAWN_WORLD_X = 1.0000
    SPAWN_WORLD_Y = 1.0000
    SPAWN_ROOT_WORLD_Z = -0.0105

    # ----- Collision-envelope calculation from x500_base/model.sdf -----
    # Collision geometry in world Z is root_z + [0.013, 0.300] m.
    COLLISION_BOTTOM_FROM_ROOT = 0.0130
    COLLISION_TOP_FROM_ROOT = 0.3005
    COLLISION_CENTER_FROM_ROOT = (
        COLLISION_BOTTOM_FROM_ROOT + COLLISION_TOP_FROM_ROOT
    ) / 2.0

    # Root height that centers the *collision envelope* in the gate opening.
    TARGET_ROOT_WORLD_Z = (
        GATE_CENTER_WORLD_Z - COLLISION_CENTER_FROM_ROOT
    )

    # Collision envelope width, aligned to gate axes:
    # Rotor boxes yield approximately -0.3136 to +0.3136 m.
    COLLISION_WIDTH_ALIGNED = 0.6272
    GEOMETRIC_SIDE_CLEARANCE = (
        OPENING_WIDTH - COLLISION_WIDTH_ALIGNED
    ) / 2.0

    # ---- Route in Gazebo world coordinates ----
    # The gate obstacle occupies world Y 3.0777..3.3777.
    PRE_GATE_WORLD_Y = 2.5500
    POST_GATE_WORLD_Y = 3.9000

    # Empirical mapping for this existing project:
    # Gazebo world Y displacement -> PX4 local X
    # Gazebo world X displacement -> PX4 local Y
    #
    # At the clean spawn, Gazebo yaw=0.9 and PX4 heading≈0.86. Therefore
    # command yaw=0.0 to align collision axes with world X/Y / gate axes.
    TARGET_PX4_YAW = 0.0

    # ----- Conservative checks -----
    TIMER_PERIOD = 0.10
    PRESTREAM_TICKS = 20
    POSITION_TOL = 0.035          # smaller than 8.7 cm geometric margin
    YAW_TOL = 0.055               # about 3.2 degrees
    STABLE_SECONDS = 3.0
    STEP_STABLE_SECONDS = 1.2
    POST_GATE_HOLD_SECONDS = 5.0

    # During crossing, only local Y and local Z are cross-section axes.
    # This value deliberately stays below geometric side clearance.
    LATERAL_LIMIT = 0.035
    VERTICAL_LIMIT = 0.045
    YAW_LIMIT = 0.070
    SAFETY_VIOLATION_SECONDS = 0.25

    # 24 short steps: ~5.6 cm each across full pre->post route.
    CROSSING_STEP_COUNT = 24

    def __init__(self) -> None:
        super().__init__("offboard_left_gate_single_traversal_corrected")

        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10
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

        self.timer = self.create_timer(self.TIMER_PERIOD, self._tick)

        self.local_position: Optional[VehicleLocalPosition] = None
        self.vehicle_status: Optional[VehicleStatus] = None
        self.home: Optional[Tuple[float, float, float]] = None

        self.phase = Phase.WAIT_TELEMETRY
        self.phase_started = time.monotonic()
        self.stable_since: Optional[float] = None
        self.safety_violation_since: Optional[float] = None
        self.last_log = 0.0

        self.target: Optional[Tuple[float, float, float]] = None
        self.abort_target: Optional[Tuple[float, float, float]] = None
        self.crossing_targets: list[Tuple[float, float, float]] = []
        self.crossing_index = 0

        self.prestream_count = 0
        self.offboard_requested = False
        self.arm_requested = False
        self.land_requested = False

        self.get_logger().info(
            "Corrected left-gate traversal started. "
            "It will yaw-align in open space before any gate approach."
        )

    # ---------------- PX4 messages ----------------

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _on_local_position(self, msg: VehicleLocalPosition) -> None:
        self.local_position = msg

    def _on_vehicle_status(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

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
        msg.velocity = [float("nan"), float("nan"), float("nan")]
        msg.acceleration = [float("nan"), float("nan"), float("nan")]
        msg.jerk = [float("nan"), float("nan"), float("nan")]
        msg.yaw = float(self.TARGET_PX4_YAW)
        msg.yawspeed = float("nan")
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
        for idx in range(1, 8):
            setattr(msg, f"param{idx}", float(params.get(f"param{idx}", 0.0)))
        self.command_pub.publish(msg)

    def _request_offboard(self) -> None:
        if not self.offboard_requested:
            self._command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1.0,
                param2=6.0,
            )
            self.offboard_requested = True
            self.get_logger().info("Requested OFFBOARD mode.")

    def _request_arm(self) -> None:
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

    # ---------------- Route and geometry ----------------

    def _pos(self) -> Tuple[float, float, float]:
        assert self.local_position is not None
        return (
            float(self.local_position.x),
            float(self.local_position.y),
            float(self.local_position.z),
        )

    def _heading(self) -> float:
        assert self.local_position is not None
        return float(self.local_position.heading)

    @staticmethod
    def _angle_error(current: float, target: float) -> float:
        return (current - target + math.pi) % (2.0 * math.pi) - math.pi

    def _root_target_local_z(self) -> float:
        assert self.home is not None
        root_up_delta = self.TARGET_ROOT_WORLD_Z - self.SPAWN_ROOT_WORLD_Z
        # PX4 local Z is NED Down: positive world-up delta means negative local Z.
        return self.home[2] - root_up_delta

    def _gate_center_local_y(self) -> float:
        assert self.home is not None
        return self.home[1] + (
            self.GATE_CENTER_WORLD_X - self.SPAWN_WORLD_X
        )

    def _local_x_for_world_y(self, world_y: float) -> float:
        assert self.home is not None
        return self.home[0] + (world_y - self.SPAWN_WORLD_Y)

    def _build_route(self) -> None:
        assert self.home is not None

        gate_y = self._gate_center_local_y()
        gate_z = self._root_target_local_z()
        pre_x = self._local_x_for_world_y(self.PRE_GATE_WORLD_Y)
        post_x = self._local_x_for_world_y(self.POST_GATE_WORLD_Y)

        self.crossing_targets = [
            (
                pre_x + (post_x - pre_x) * i / self.CROSSING_STEP_COUNT,
                gate_y,
                gate_z,
            )
            for i in range(1, self.CROSSING_STEP_COUNT + 1)
        ]

        self.get_logger().info(
            "Static geometry: opening=0.8014m x 0.8014m; "
            f"aligned X500 collision width≈{self.COLLISION_WIDTH_ALIGNED:.4f}m; "
            f"side clearance≈{self.GEOMETRIC_SIDE_CLEARANCE:.4f}m."
        )
        self.get_logger().info(
            f"Corrected root target world Z={self.TARGET_ROOT_WORLD_Z:.4f}m "
            f"(collision center at gate Z={self.GATE_CENTER_WORLD_Z:.4f}m)."
        )
        self.get_logger().info(
            f"Route local NED: pre=({pre_x:.3f}, {gate_y:.3f}, {gate_z:.3f}), "
            f"post=({post_x:.3f}, {gate_y:.3f}, {gate_z:.3f}), "
            f"yaw={self.TARGET_PX4_YAW:.3f}."
        )

    # ---------------- State helpers ----------------

    def _set_phase(self, phase: Phase, detail: str) -> None:
        self.phase = phase
        self.phase_started = time.monotonic()
        self.stable_since = None
        self.safety_violation_since = None
        self.get_logger().info(detail)

    def _distance(self, target: Tuple[float, float, float]) -> float:
        x, y, z = self._pos()
        return math.sqrt(
            (x - target[0]) ** 2
            + (y - target[1]) ** 2
            + (z - target[2]) ** 2
        )

    def _stable_target(
        self, target: Tuple[float, float, float], seconds: float
    ) -> bool:
        now = time.monotonic()
        yaw_ok = abs(self._angle_error(self._heading(), self.TARGET_PX4_YAW)) <= self.YAW_TOL
        pos_ok = self._distance(target) <= self.POSITION_TOL

        if pos_ok and yaw_ok:
            if self.stable_since is None:
                self.stable_since = now
            return now - self.stable_since >= seconds

        self.stable_since = None
        return False

    def _log(self, label: str, target: Tuple[float, float, float]) -> None:
        now = time.monotonic()
        if now - self.last_log < 1.0:
            return

        x, y, z = self._pos()
        dist = self._distance(target)
        heading = self._heading()
        yaw_err = self._angle_error(heading, self.TARGET_PX4_YAW)
        self.get_logger().info(
            f"{label} | current=({x:.3f}, {y:.3f}, {z:.3f}) "
            f"| goal=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) "
            f"| d={dist:.3f}m | heading={heading:.3f} | yaw_err={yaw_err:.3f}"
        )
        self.last_log = now

    def _crossing_safe(self, target: Tuple[float, float, float]) -> bool:
        x, y, z = self._pos()
        _ = x  # Along-route error is handled by reached-step logic.
        lateral_error = abs(y - target[1])
        vertical_error = abs(z - target[2])
        yaw_error = abs(self._angle_error(self._heading(), self.TARGET_PX4_YAW))

        return (
            lateral_error <= self.LATERAL_LIMIT
            and vertical_error <= self.VERTICAL_LIMIT
            and yaw_error <= self.YAW_LIMIT
        )

    def _abort(self, reason: str) -> None:
        self.abort_target = self._pos()
        self._set_phase(
            Phase.ABORT_HOLD,
            f"SAFETY ABORT: {reason}. Holding current point, then landing."
        )

    # ---------------- Main loop ----------------

    def _tick(self) -> None:
        if self.local_position is None:
            if self.phase == Phase.WAIT_TELEMETRY:
                self.get_logger().info(
                    "Waiting for /fmu/out/vehicle_local_position ..."
                )
            return

        if self.home is None:
            self.home = self._pos()
            self._build_route()
            self.target = (
                self.home[0],
                self.home[1],
                self._root_target_local_z(),
            )
            self.get_logger().info(
                f"Captured local home: ({self.home[0]:.3f}, "
                f"{self.home[1]:.3f}, {self.home[2]:.3f})"
            )

        assert self.target is not None

        self._publish_heartbeat()
        self._publish_setpoint(self.target)

        if self.phase == Phase.WAIT_TELEMETRY:
            self.prestream_count += 1
            self._log("PRESTREAM", self.target)

            if self.prestream_count >= self.PRESTREAM_TICKS:
                self._request_offboard()
                self._request_arm()
                self._set_phase(
                    Phase.TAKEOFF_TO_SAFE_ROOT_HEIGHT,
                    "TAKEOFF: rise at home XY to corrected collision-centered height."
                )
            return

        if self.phase == Phase.TAKEOFF_TO_SAFE_ROOT_HEIGHT:
            self._log("TAKEOFF", self.target)
            if self._stable_target(self.target, self.STABLE_SECONDS):
                self.target = (
                    self.home[0],
                    self.home[1],
                    self._root_target_local_z(),
                )
                self._set_phase(
                    Phase.YAW_ALIGN_OPEN_SPACE,
                    "YAW_ALIGN_OPEN_SPACE: hold at home XY until yaw is aligned to 0 rad."
                )
            return

        if self.phase == Phase.YAW_ALIGN_OPEN_SPACE:
            self._log("YAW_ALIGN", self.target)
            if self._stable_target(self.target, self.STABLE_SECONDS):
                self.target = (
                    self.home[0],
                    self._gate_center_local_y(),
                    self._root_target_local_z(),
                )
                self._set_phase(
                    Phase.LATERAL_ALIGN_OPEN_SPACE,
                    "LATERAL_ALIGN_OPEN_SPACE: move sideways to gate centerline while still far from the frame."
                )
            return

        if self.phase == Phase.LATERAL_ALIGN_OPEN_SPACE:
            self._log("LATERAL_ALIGN", self.target)
            if self._stable_target(self.target, self.STABLE_SECONDS):
                self.target = (
                    self._local_x_for_world_y(self.PRE_GATE_WORLD_Y),
                    self._gate_center_local_y(),
                    self._root_target_local_z(),
                )
                self._set_phase(
                    Phase.MOVE_TO_PRE_GATE,
                    "MOVE_TO_PRE_GATE: approach the gate while yaw/height/centerline remain locked."
                )
            return

        if self.phase == Phase.MOVE_TO_PRE_GATE:
            self._log("PRE_GATE_APPROACH", self.target)
            if self._stable_target(self.target, self.STABLE_SECONDS):
                self._set_phase(
                    Phase.PRE_GATE_VERIFY,
                    "PRE_GATE_VERIFY: holding for final strict alignment before crossing."
                )
            return

        if self.phase == Phase.PRE_GATE_VERIFY:
            self._log("PRE_GATE_VERIFY", self.target)
            if self._stable_target(self.target, self.STABLE_SECONDS):
                self.crossing_index = 0
                self.target = self.crossing_targets[0]
                self._set_phase(
                    Phase.CROSSING,
                    f"CROSSING START: {self.CROSSING_STEP_COUNT} short position steps; "
                    "strict lateral/vertical/yaw corridor active."
                )
            return

        if self.phase == Phase.CROSSING:
            now = time.monotonic()

            if self._crossing_safe(self.target):
                self.safety_violation_since = None
            else:
                if self.safety_violation_since is None:
                    self.safety_violation_since = now
                elif now - self.safety_violation_since >= self.SAFETY_VIOLATION_SECONDS:
                    self._abort(
                        "cross-section alignment exceeded limits "
                        f"(Y={self.LATERAL_LIMIT:.3f}, Z={self.VERTICAL_LIMIT:.3f}, "
                        f"yaw={self.YAW_LIMIT:.3f})"
                    )
                    return

            self._log(
                f"CROSS_STEP_{self.crossing_index + 1}/{len(self.crossing_targets)}",
                self.target,
            )

            if self._stable_target(self.target, self.STEP_STABLE_SECONDS):
                self.crossing_index += 1
                if self.crossing_index >= len(self.crossing_targets):
                    self._set_phase(
                        Phase.POST_GATE_HOLD,
                        "POST_GATE_HOLD: all crossing steps completed; holding safely beyond frame."
                    )
                else:
                    self.target = self.crossing_targets[self.crossing_index]
                    self.stable_since = None
                    self.get_logger().info(
                        f"Advance to crossing step "
                        f"{self.crossing_index + 1}/{len(self.crossing_targets)}."
                    )
            return

        if self.phase == Phase.POST_GATE_HOLD:
            self._log("POST_GATE_HOLD", self.target)
            if time.monotonic() - self.phase_started >= self.POST_GATE_HOLD_SECONDS:
                self._request_land()
                self._set_phase(
                    Phase.LANDING,
                    "LANDING: AUTO LAND after successful one-way traversal."
                )
            return

        if self.phase == Phase.ABORT_HOLD:
            assert self.abort_target is not None
            self.target = self.abort_target
            self._log("ABORT_HOLD", self.target)
            if time.monotonic() - self.phase_started >= 2.0:
                self._request_land()
                self._set_phase(
                    Phase.LANDING,
                    "LANDING: AUTO LAND after safety abort."
                )
            return

        if self.phase == Phase.LANDING:
            self._log("LANDING", self.target)
            if (
                self.vehicle_status is not None
                and self.vehicle_status.arming_state
                == VehicleStatus.ARMING_STATE_DISARMED
            ):
                self._set_phase(
                    Phase.FINISHED,
                    "Vehicle disarmed. Corrected left-gate traversal completed."
                )
            return

    def shutdown_land(self) -> None:
        if self.local_position is None:
            return
        if self.phase not in (Phase.FINISHED, Phase.LANDING):
            try:
                self._request_land()
                time.sleep(0.3)
            except Exception:
                pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeftGateTraversalCorrected()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown_land()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
