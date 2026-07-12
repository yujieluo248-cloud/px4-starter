#!/usr/bin/env python3
"""
Gate height micro-calibration at a fixed safe horizontal position.

Purpose:
- Does NOT cross the gate.
- Does NOT move closer to the gate after reaching the chosen safe point.
- Keeps the same horizontal point:
    relative X = 1.30 m
    relative Y = -0.46 m
- Compares three nearby heights:
    BASELINE_MID = 0.70 m above home
    HEIGHT_A     = 0.80 m above home
    HEIGHT_B     = 0.88 m above home

Mission:
WAIT_POSITION
-> TAKEOFF
-> FLY_TO_FIXED_POINT
-> BASELINE_HOLD
-> FLY_TO_HEIGHT_A
-> HEIGHT_A_HOLD
-> FLY_TO_HEIGHT_B
-> HEIGHT_B_HOLD
-> RETURN_HOME
-> FINAL_HOLD
-> LANDING
-> MISSION_FINISHED
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


class GateHeightMicroCalibration(Node):
    def __init__(self):
        super().__init__('offboard_gate_height_micro_calibration')

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

        # Fixed horizontal test location.
        # This is the STEP_1 point from the safe approach experiment.
        self.fixed_rel_x = 1.30
        self.fixed_rel_y = -0.46

        # Height tests, all relative to the current run home point.
        self.baseline_height = 0.70
        self.height_a = 0.80
        self.height_b = 0.88

        self.baseline_hold_seconds = 6.0
        self.height_a_hold_seconds = 6.0
        self.height_b_hold_seconds = 6.0
        self.final_hold_seconds = 3.0

        # More relaxed takeoff completion so the mission does not wait too long.
        self.takeoff_xy_tolerance = 0.25
        self.takeoff_z_tolerance = 0.18

        # Position completion for fixed-point and height transitions.
        self.point_xy_tolerance = 0.16
        self.point_z_tolerance = 0.10

        self.takeoff_timeout_seconds = 35.0
        self.move_timeout_seconds = 30.0

        self.state = 'WAIT_POSITION'
        self.state_start_counter = None
        self.counter = 0
        self.last_log_counter = -10
        self.land_requested = False

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            'Gate height micro-calibration started. '
            'This mission keeps the horizontal point fixed and '
            'does not cross the gate.'
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

    def elapsed_seconds(self):
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

    def current_relative_height(self):
        # PX4 local frame uses NED: more negative z means higher.
        return self.home_z - self.position.z

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

    def reached_point(self):
        xy_error, z_error = self.errors()
        return (
            xy_error <= self.point_xy_tolerance
            and z_error <= self.point_z_tolerance
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

    def send_command(self, command, param1=0.0, param2=0.0):
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
        self.send_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0,
        )
        self.get_logger().info('Requested OFFBOARD mode')

    def request_arm(self):
        self.send_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0,
        )
        self.get_logger().info('Requested ARM')

    def request_land(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Requested AUTO LAND')

    def abort_to_land(self, reason):
        self.get_logger().error(
            f'Mission abort: {reason}. Requesting AUTO LAND.'
        )
        if not self.land_requested:
            self.request_land()
            self.land_requested = True
        self.start_state('LANDING')

    def fixed_point_x(self):
        return self.home_x + self.fixed_rel_x

    def fixed_point_y(self):
        return self.home_y + self.fixed_rel_y

    def z_for_height(self, height):
        return self.home_z - height

    def set_fixed_point_height_goal(self, height):
        self.set_goal(
            self.fixed_point_x(),
            self.fixed_point_y(),
            self.z_for_height(height),
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
        self.get_logger().info(
            f'[{self.state}] Current: '
            f'x={self.position.x:.2f}, '
            f'y={self.position.y:.2f}, '
            f'z={self.position.z:.2f} | '
            f'Relative XY: x={self.rel_x():.2f}, '
            f'y={self.rel_y():.2f} | '
            f'Actual relative height={self.current_relative_height():.2f} m | '
            f'Goal: x={self.goal_x:.2f}, '
            f'y={self.goal_y:.2f}, '
            f'z={self.goal_z:.2f} | '
            f'XYerr={xy_error:.2f}, Zerr={z_error:.2f} | '
            f'State time={self.elapsed_seconds():.1f}s'
        )

    def go_to_height_a(self):
        self.set_fixed_point_height_goal(self.height_a)
        self.start_state('FLY_TO_HEIGHT_A')
        self.get_logger().info(
            f'Raising only at fixed XY point: '
            f'{self.baseline_height:.2f} -> {self.height_a:.2f} m '
            'above home.'
        )

    def go_to_height_b(self):
        self.set_fixed_point_height_goal(self.height_b)
        self.start_state('FLY_TO_HEIGHT_B')
        self.get_logger().info(
            f'Raising only at fixed XY point: '
            f'{self.height_a:.2f} -> {self.height_b:.2f} m '
            'above home.'
        )

    def go_home(self):
        self.set_goal(
            self.home_x,
            self.home_y,
            self.z_for_height(self.baseline_height),
        )
        self.start_state('RETURN_HOME')
        self.get_logger().info(
            'All micro-height tests complete. Returning above home.'
        )

    def timer_callback(self):
        controlled_states = {
            'TAKEOFF',
            'FLY_TO_FIXED_POINT',
            'BASELINE_HOLD',
            'FLY_TO_HEIGHT_A',
            'HEIGHT_A_HOLD',
            'FLY_TO_HEIGHT_B',
            'HEIGHT_B_HOLD',
            'RETURN_HOME',
            'FINAL_HOLD',
        }

        if self.state in controlled_states and self.goal_x is not None:
            self.publish_position_mode()
            self.publish_position_setpoint()

        if self.state == 'WAIT_POSITION':
            if self.position.timestamp == 0:
                return

            self.home_x = self.position.x
            self.home_y = self.position.y
            self.home_z = self.position.z

            self.set_goal(
                self.home_x,
                self.home_y,
                self.z_for_height(self.baseline_height),
            )
            self.start_state('TAKEOFF')

            self.get_logger().info(
                f'Captured home: x={self.home_x:.2f}, '
                f'y={self.home_y:.2f}, z={self.home_z:.2f}'
            )
            self.get_logger().info(
                'Fixed horizontal test point: '
                f'relative X={self.fixed_rel_x:.2f}, '
                f'relative Y={self.fixed_rel_y:.2f}. '
                f'Heights: baseline={self.baseline_height:.2f} m, '
                f'A={self.height_a:.2f} m, '
                f'B={self.height_b:.2f} m.'
            )

        if self.counter - self.last_log_counter >= 10:
            self.log_progress()
            self.last_log_counter = self.counter

        # PX4 needs continuous setpoints before OFFBOARD is accepted.
        if self.counter < 10:
            self.counter += 1
            return

        if self.state == 'LANDING':
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.start_state('MISSION_FINISHED')
                self.get_logger().info(
                    'Vehicle disarmed. Height micro-calibration finished.'
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

        if self.state == 'TAKEOFF':
            if self.reached_takeoff():
                self.set_fixed_point_height_goal(self.baseline_height)
                self.start_state('FLY_TO_FIXED_POINT')
                self.get_logger().info(
                    'Takeoff accepted. Flying to the fixed STEP_1 '
                    'horizontal test point.'
                )
            elif self.elapsed_seconds() > self.takeoff_timeout_seconds:
                self.abort_to_land('takeoff timeout')

        elif self.state == 'FLY_TO_FIXED_POINT':
            if self.reached_point():
                self.start_state('BASELINE_HOLD')
                self.get_logger().info(
                    f'Reached BASELINE_MID at fixed XY. Holding for '
                    f'{self.baseline_hold_seconds:.1f}s. '
                    'Observe and capture a screenshot.'
                )
            elif self.elapsed_seconds() > self.move_timeout_seconds:
                self.abort_to_land('fixed-point movement timeout')

        elif self.state == 'BASELINE_HOLD':
            if self.elapsed_seconds() >= self.baseline_hold_seconds:
                self.go_to_height_a()

        elif self.state == 'FLY_TO_HEIGHT_A':
            if self.reached_point():
                self.start_state('HEIGHT_A_HOLD')
                self.get_logger().info(
                    f'Reached HEIGHT_A={self.height_a:.2f}m. Holding for '
                    f'{self.height_a_hold_seconds:.1f}s. '
                    'Observe and capture a screenshot.'
                )
            elif self.elapsed_seconds() > self.move_timeout_seconds:
                self.abort_to_land('height-A movement timeout')

        elif self.state == 'HEIGHT_A_HOLD':
            if self.elapsed_seconds() >= self.height_a_hold_seconds:
                self.go_to_height_b()

        elif self.state == 'FLY_TO_HEIGHT_B':
            if self.reached_point():
                self.start_state('HEIGHT_B_HOLD')
                self.get_logger().info(
                    f'Reached HEIGHT_B={self.height_b:.2f}m. Holding for '
                    f'{self.height_b_hold_seconds:.1f}s. '
                    'Observe and capture a screenshot.'
                )
            elif self.elapsed_seconds() > self.move_timeout_seconds:
                self.abort_to_land('height-B movement timeout')

        elif self.state == 'HEIGHT_B_HOLD':
            if self.elapsed_seconds() >= self.height_b_hold_seconds:
                self.go_home()

        elif self.state == 'RETURN_HOME':
            if self.reached_point():
                self.start_state('FINAL_HOLD')
                self.get_logger().info(
                    f'Home reached. Holding for '
                    f'{self.final_hold_seconds:.1f}s.'
                )
            elif self.elapsed_seconds() > self.move_timeout_seconds:
                self.abort_to_land('return-home timeout')

        elif self.state == 'FINAL_HOLD':
            if (
                self.elapsed_seconds() >= self.final_hold_seconds
                and not self.land_requested
            ):
                self.request_land()
                self.land_requested = True
                self.start_state('LANDING')

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = GateHeightMicroCalibration()

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
