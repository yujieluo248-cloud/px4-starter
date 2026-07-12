#!/usr/bin/env python3
"""
Gate step-approach calibration.

Purpose:
- DO NOT cross the gate.
- Keep Y and Z locked with position control at the previously tested centerline
  and MID height.
- Move from the pre-gate point forward in three small 0.07 m steps.
- Hold at each step so the real front edge of the obstacle can be observed.

Mission:
WAIT_POSITION -> TAKEOFF -> FLY_TO_PRE_GATE
-> STEP_1_HOLD -> STEP_2_HOLD -> STEP_3_HOLD
-> RETURN_HOME -> LANDING -> FINISHED
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)


class GateStepApproach(Node):
    def __init__(self):
        super().__init__('offboard_gate_step_approach')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos,
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos,
        )
        self.command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos,
        )

        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos,
        )
        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            qos,
        )

        self.position = VehicleLocalPosition()
        self.status = VehicleStatus()

        self.home_x = None
        self.home_y = None
        self.home_z = None

        self.goal_x = None
        self.goal_y = None
        self.goal_z = None

        # Calibrated relative map values.
        self.gate_center_y = -0.46
        self.traverse_height = 0.70

        # Previously successful safe staging point:
        # gate estimated near +2.08 m, safety distance 0.85 m -> +1.23 m
        self.pre_gate_rel_x = 1.23

        # IMPORTANT:
        # The earlier collision began around relative X ~1.46-1.52.
        # This program intentionally stops at +1.44 m at most.
        # It DOES NOT attempt to cross the obstacle.
        self.approach_steps = [
            ('STEP_1', 1.30),
            ('STEP_2', 1.37),
            ('STEP_3', 1.44),
        ]

        self.takeoff_xy_tolerance = 0.25
        self.takeoff_z_tolerance = 0.18

        self.position_xy_tolerance = 0.12
        self.position_z_tolerance = 0.08

        self.pre_gate_hold_seconds = 2.0
        self.step_hold_seconds = 4.0
        self.return_hold_seconds = 3.0

        self.takeoff_timeout_seconds = 35.0
        self.position_timeout_seconds = 25.0

        # Safety stop:
        # If position tracking suddenly drifts far from its setpoint, request land.
        self.max_xy_tracking_error = 0.45
        self.max_z_tracking_error = 0.35

        self.state = 'WAIT_POSITION'
        self.state_start_counter = None
        self.step_index = 0

        self.counter = 0
        self.last_log_counter = -10
        self.land_requested = False

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            'Gate step-approach calibration started. '
            'This mission does NOT cross the gate.'
        )

    def now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def position_callback(self, msg):
        self.position = msg

    def status_callback(self, msg):
        self.status = msg

    def start_state(self, state):
        self.state = state
        self.state_start_counter = self.counter

    def elapsed(self):
        if self.state_start_counter is None:
            return 0.0
        return (self.counter - self.state_start_counter) / 10.0

    def set_goal(self, x, y, z):
        self.goal_x = float(x)
        self.goal_y = float(y)
        self.goal_z = float(z)

    def rel_x(self):
        return self.position.x - self.home_x

    def rel_y(self):
        return self.position.y - self.home_y

    def errors(self):
        xy_error = math.sqrt(
            (self.position.x - self.goal_x) ** 2
            + (self.position.y - self.goal_y) ** 2
        )
        z_error = abs(self.position.z - self.goal_z)
        return xy_error, z_error

    def reached_takeoff(self):
        xy_error, z_error = self.errors()
        return (
            xy_error <= self.takeoff_xy_tolerance
            and z_error <= self.takeoff_z_tolerance
        )

    def reached_position(self):
        xy_error, z_error = self.errors()
        return (
            xy_error <= self.position_xy_tolerance
            and z_error <= self.position_z_tolerance
        )

    def has_large_tracking_error(self):
        xy_error, z_error = self.errors()
        return (
            xy_error > self.max_xy_tracking_error
            or z_error > self.max_z_tracking_error
        )

    def publish_position_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.now_us()
        self.mode_pub.publish(msg)

    def publish_position_setpoint(self):
        if self.goal_x is None:
            return

        nan = float('nan')

        msg = TrajectorySetpoint()
        msg.position = [self.goal_x, self.goal_y, self.goal_z]
        msg.velocity = [nan, nan, nan]
        msg.acceleration = [nan, nan, nan]
        msg.yaw = nan
        msg.yawspeed = nan
        msg.timestamp = self.now_us()
        self.setpoint_pub.publish(msg)

    def command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.now_us()
        self.command_pub.publish(msg)

    def request_offboard(self):
        self.command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0,
        )
        self.get_logger().info('Requested OFFBOARD mode')

    def request_arm(self):
        self.command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0,
        )
        self.get_logger().info('Requested ARM')

    def request_land(self):
        self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Requested AUTO LAND')

    def abort_to_land(self, reason):
        self.get_logger().error(
            f'SAFETY ABORT: {reason}. Requesting AUTO LAND.'
        )
        if not self.land_requested:
            self.request_land()
            self.land_requested = True
        self.start_state('LANDING')

    def build_route(self):
        self.flight_z = self.home_z - self.traverse_height

        self.pre_gate_x = self.home_x + self.pre_gate_rel_x
        self.pre_gate_y = self.home_y + self.gate_center_y

    def set_pre_gate_goal(self):
        self.set_goal(self.pre_gate_x, self.pre_gate_y, self.flight_z)

    def set_current_step_goal(self):
        name, relative_x = self.approach_steps[self.step_index]

        self.set_goal(
            self.home_x + relative_x,
            self.home_y + self.gate_center_y,
            self.flight_z,
        )

        self.get_logger().info(
            f'Approaching {name}: '
            f'relative X={relative_x:.2f} m, '
            f'relative Y={self.gate_center_y:.2f} m. '
            'Y and Z remain position-locked.'
        )

    def log_progress(self):
        if self.state in ('LANDING', 'MISSION_FINISHED'):
            self.get_logger().info(
                f'[{self.state}] Current: '
                f'x={self.position.x:.2f}, '
                f'y={self.position.y:.2f}, '
                f'z={self.position.z:.2f}'
            )
            return

        xy_error, z_error = self.errors()

        text = (
            f'[{self.state}] Current: '
            f'x={self.position.x:.2f}, '
            f'y={self.position.y:.2f}, '
            f'z={self.position.z:.2f} | '
            f'Relative: x={self.rel_x():.2f}, '
            f'y={self.rel_y():.2f} | '
            f'Goal: x={self.goal_x:.2f}, '
            f'y={self.goal_y:.2f}, '
            f'z={self.goal_z:.2f} | '
            f'XYerr={xy_error:.2f}, Zerr={z_error:.2f} | '
            f'State time={self.elapsed():.1f}s'
        )

        if self.state in ('STEP_HOLD', 'FLY_TO_STEP'):
            name, relative_x = self.approach_steps[self.step_index]
            text += (
                f' | {name}: target relative X={relative_x:.2f} m'
            )

        self.get_logger().info(text)

    def timer_callback(self):
        position_states = {
            'TAKEOFF',
            'FLY_TO_PRE_GATE',
            'PRE_GATE_HOLD',
            'FLY_TO_STEP',
            'STEP_HOLD',
            'RETURN_HOME',
            'FINAL_HOLD',
        }

        if self.state in position_states and self.goal_x is not None:
            self.publish_position_mode()
            self.publish_position_setpoint()

        if self.state == 'WAIT_POSITION':
            if self.position.timestamp == 0:
                return

            self.home_x = self.position.x
            self.home_y = self.position.y
            self.home_z = self.position.z

            self.build_route()

            self.set_goal(
                self.home_x,
                self.home_y,
                self.flight_z,
            )

            self.start_state('TAKEOFF')

            self.get_logger().info(
                'Captured home: '
                f'x={self.home_x:.2f}, '
                f'y={self.home_y:.2f}, '
                f'z={self.home_z:.2f}'
            )

            self.get_logger().info(
                'Safety plan: takeoff to MID height, move to pre-gate, '
                'then stop at relative X=1.30, 1.37, 1.44 m. '
                'No gate crossing command is present.'
            )

        if self.counter - self.last_log_counter >= 10:
            self.log_progress()
            self.last_log_counter = self.counter

        # PX4 requires setpoint streaming before Offboard.
        if self.counter < 10:
            self.counter += 1
            return

        if self.state == 'LANDING':
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.start_state('MISSION_FINISHED')
                self.get_logger().info(
                    'Vehicle disarmed. Gate step-approach calibration finished.'
                )
            self.counter += 1
            return

        if self.state == 'MISSION_FINISHED':
            self.counter += 1
            return

        if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if self.counter % 10 == 0:
                self.request_offboard()
            self.counter += 1
            return

        if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
            if self.counter % 10 == 0:
                self.request_arm()
            self.counter += 1
            return
        # 只有在门前悬停、或者执行很短的阶梯逼近时，
        # 才检查是否出现异常的大幅偏离。
        #
        # FLY_TO_PRE_GATE 和 RETURN_HOME 属于正常的较长距离飞行，
        # 不能用“距离目标超过 0.45 m”作为中止条件。
        if self.state in (
            'PRE_GATE_HOLD',
            'FLY_TO_STEP',
            'STEP_HOLD',
            'FINAL_HOLD',
        ):
            if self.has_large_tracking_error():
                self.abort_to_land('large tracking deviation')
                self.counter += 1
                return

        if self.state == 'TAKEOFF':
            if self.reached_takeoff():
                self.set_pre_gate_goal()
                self.start_state('FLY_TO_PRE_GATE')
                self.get_logger().info(
                    'Takeoff accepted. Flying to safe pre-gate point.'
                )
            elif self.elapsed() > self.takeoff_timeout_seconds:
                self.abort_to_land('takeoff timeout')

        elif self.state == 'FLY_TO_PRE_GATE':
            if self.reached_position():
                self.start_state('PRE_GATE_HOLD')
                self.get_logger().info(
                    f'Reached pre-gate point. Holding for '
                    f'{self.pre_gate_hold_seconds:.1f} seconds.'
                )
            elif self.elapsed() > self.position_timeout_seconds:
                self.abort_to_land('pre-gate timeout')

        elif self.state == 'PRE_GATE_HOLD':
            if self.elapsed() >= self.pre_gate_hold_seconds:
                self.step_index = 0
                self.set_current_step_goal()
                self.start_state('FLY_TO_STEP')

        elif self.state == 'FLY_TO_STEP':
            if self.reached_position():
                self.start_state('STEP_HOLD')
                name, _ = self.approach_steps[self.step_index]
                self.get_logger().info(
                    f'Reached {name}. Holding for '
                    f'{self.step_hold_seconds:.1f} seconds. '
                    'Observe actual distance to the obstacle front edge.'
                )
            elif self.elapsed() > self.position_timeout_seconds:
                self.abort_to_land('step movement timeout')

        elif self.state == 'STEP_HOLD':
            if self.elapsed() >= self.step_hold_seconds:
                self.step_index += 1

                if self.step_index < len(self.approach_steps):
                    self.set_current_step_goal()
                    self.start_state('FLY_TO_STEP')
                else:
                    self.set_goal(
                        self.home_x,
                        self.home_y,
                        self.flight_z,
                    )
                    self.start_state('RETURN_HOME')
                    self.get_logger().info(
                        'All safe approach steps complete. '
                        'Returning above home point.'
                    )

        elif self.state == 'RETURN_HOME':
            if self.reached_position():
                self.start_state('FINAL_HOLD')
                self.get_logger().info(
                    f'Home reached. Holding for '
                    f'{self.return_hold_seconds:.1f} seconds.'
                )
            elif self.elapsed() > self.position_timeout_seconds:
                self.abort_to_land('return-home timeout')

        elif self.state == 'FINAL_HOLD':
            if (
                self.elapsed() >= self.return_hold_seconds
                and not self.land_requested
            ):
                self.request_land()
                self.land_requested = True
                self.start_state('LANDING')

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = GateStepApproach()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
