#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)


class GateLowSpeedTraversal(Node):
    """Fixed-map, low-speed, single gate traversal demo.

    Route:
    takeoff -> pre-gate hold -> cross gate -> far-side hold
    -> cross back -> return home -> land.

    This uses the already validated horizontal centerline and MID height.
    It is not vision-based obstacle avoidance.
    """

    def __init__(self):
        super().__init__('offboard_gate_low_speed_traversal')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.position_callback, qos)
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.status_callback, qos)

        self.position = VehicleLocalPosition()
        self.status = VehicleStatus()

        self.home_x = self.home_y = self.home_z = None
        self.goal_x = self.goal_y = self.goal_z = None

        # Relative to the current HOME point.
        self.gate_plane_x_min = 2.08
        self.gate_plane_x_max = 2.38
        self.gate_center_y = -0.46

        # Chosen from the successful vertical calibration.
        self.traverse_height = 0.70

        self.pre_gate_safety_distance = 0.85
        self.post_gate_clearance = 0.75
        self.cross_speed = 0.15

        self.pre_gate_hold_seconds = 3.0
        self.post_gate_hold_seconds = 4.0
        self.pre_gate_return_hold_seconds = 2.0
        self.final_hold_seconds = 3.0

        # The relaxed takeoff tolerance prevents the earlier long wait.
        self.takeoff_xy_tolerance = 0.25
        self.takeoff_z_tolerance = 0.18

        self.position_xy_tolerance = 0.18
        self.position_z_tolerance = 0.10

        self.takeoff_timeout_seconds = 35.0
        self.position_timeout_seconds = 30.0
        self.cross_timeout_seconds = 22.0

        self.state = 'WAIT_POSITION'
        self.state_start_counter = None
        self.counter = 0
        self.last_log_counter = -10
        self.land_requested = False

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(
            'Low-speed gate traversal started. Waiting for PX4 position...')

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
        xy = math.sqrt(
            (self.position.x - self.goal_x) ** 2 +
            (self.position.y - self.goal_y) ** 2
        )
        z = abs(self.position.z - self.goal_z)
        return xy, z

    def reached_takeoff(self):
        xy, z = self.errors()
        return xy <= self.takeoff_xy_tolerance and z <= self.takeoff_z_tolerance

    def reached_position_goal(self):
        xy, z = self.errors()
        return xy <= self.position_xy_tolerance and z <= self.position_z_tolerance

    def publish_mode(self, position=False, velocity=False):
        msg = OffboardControlMode()
        msg.position = position
        msg.velocity = velocity
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.now_us()
        self.mode_pub.publish(msg)

    def publish_position_setpoint(self):
        nan = float('nan')
        msg = TrajectorySetpoint()
        msg.position = [self.goal_x, self.goal_y, self.goal_z]
        msg.velocity = [nan, nan, nan]
        msg.acceleration = [nan, nan, nan]
        msg.yaw = nan
        msg.yawspeed = nan
        msg.timestamp = self.now_us()
        self.setpoint_pub.publish(msg)

    def publish_velocity_setpoint(self, vx):
        nan = float('nan')
        msg = TrajectorySetpoint()
        msg.position = [nan, nan, nan]
        msg.velocity = [float(vx), 0.0, 0.0]
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
        self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info('Requested OFFBOARD mode')

    def request_arm(self):
        self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info('Requested ARM')

    def request_land(self):
        self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Requested AUTO LAND')

    def build_route(self):
        self.pre_rel_x = self.gate_plane_x_min - self.pre_gate_safety_distance
        self.post_rel_x = self.gate_plane_x_max + self.post_gate_clearance
        self.flight_z = self.home_z - self.traverse_height

        self.pre_x = self.home_x + self.pre_rel_x
        self.pre_y = self.home_y + self.gate_center_y
        self.post_x = self.home_x + self.post_rel_x
        self.post_y = self.home_y + self.gate_center_y

    def abort_land(self, reason):
        self.get_logger().error(f'Mission abort: {reason}. Landing now.')
        if not self.land_requested:
            self.request_land()
            self.land_requested = True
        self.start_state('LANDING')

    def log(self):
        if self.state in ('LANDING', 'MISSION_FINISHED'):
            self.get_logger().info(
                f'[{self.state}] Current: x={self.position.x:.2f}, '
                f'y={self.position.y:.2f}, z={self.position.z:.2f}')
            return

        text = (
            f'[{self.state}] Current: x={self.position.x:.2f}, '
            f'y={self.position.y:.2f}, z={self.position.z:.2f} | '
            f'Relative: x={self.rel_x():.2f}, y={self.rel_y():.2f}'
        )

        if self.goal_x is not None and self.state not in ('CROSS_OUT', 'CROSS_BACK'):
            xy, z = self.errors()
            text += (
                f' | Goal: x={self.goal_x:.2f}, y={self.goal_y:.2f}, '
                f'z={self.goal_z:.2f} | XYerr={xy:.2f}, Zerr={z:.2f}'
            )

        if self.state == 'CROSS_OUT':
            text += (
                f' | Outbound crossing: {self.rel_x():.2f}/'
                f'{self.post_rel_x:.2f} m | vx=+{self.cross_speed:.2f}')

        if self.state == 'CROSS_BACK':
            text += (
                f' | Return crossing: {self.rel_x():.2f}/'
                f'{self.pre_rel_x:.2f} m | vx=-{self.cross_speed:.2f}')

        text += f' | State time={self.elapsed():.1f}s'
        self.get_logger().info(text)

    def timer_callback(self):
        position_states = {
            'TAKEOFF', 'FLY_TO_PRE_GATE', 'PRE_GATE_HOLD',
            'POST_GATE_HOLD', 'PRE_GATE_RETURN_HOLD',
            'RETURN_HOME', 'FINAL_HOLD'
        }

        if self.state in position_states and self.goal_x is not None:
            self.publish_mode(position=True)
            self.publish_position_setpoint()
        elif self.state == 'CROSS_OUT':
            self.publish_mode(velocity=True)
            self.publish_velocity_setpoint(+self.cross_speed)
        elif self.state == 'CROSS_BACK':
            self.publish_mode(velocity=True)
            self.publish_velocity_setpoint(-self.cross_speed)

        if self.state == 'WAIT_POSITION':
            if self.position.timestamp == 0:
                return

            self.home_x = self.position.x
            self.home_y = self.position.y
            self.home_z = self.position.z
            self.build_route()
            self.set_goal(self.home_x, self.home_y, self.flight_z)
            self.start_state('TAKEOFF')

            self.get_logger().info(
                f'Captured home: x={self.home_x:.2f}, y={self.home_y:.2f}, '
                f'z={self.home_z:.2f}')
            self.get_logger().info(
                f'MID traverse height={self.traverse_height:.2f}m; '
                f'pre-gate=({self.pre_rel_x:.2f}, {self.gate_center_y:.2f}); '
                f'post-gate=({self.post_rel_x:.2f}, {self.gate_center_y:.2f}); '
                f'speed={self.cross_speed:.2f}m/s')

        if self.counter - self.last_log_counter >= 10:
            self.log()
            self.last_log_counter = self.counter

        # PX4 requires streaming setpoints before switching to Offboard.
        if self.counter < 10:
            self.counter += 1
            return

        if self.state == 'LANDING':
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.start_state('MISSION_FINISHED')
                self.get_logger().info(
                    'Vehicle disarmed. Low-speed gate traversal finished.')
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
                self.set_goal(self.pre_x, self.pre_y, self.flight_z)
                self.start_state('FLY_TO_PRE_GATE')
                self.get_logger().info(
                    'Takeoff accepted. Flying to pre-gate staging point.')
            elif self.elapsed() > self.takeoff_timeout_seconds:
                self.abort_land('takeoff timeout')

        elif self.state == 'FLY_TO_PRE_GATE':
            if self.reached_position_goal():
                self.start_state('PRE_GATE_HOLD')
                self.get_logger().info(
                    f'Reached pre-gate point. Holding for '
                    f'{self.pre_gate_hold_seconds:.1f}s.')
            elif self.elapsed() > self.position_timeout_seconds:
                self.abort_land('pre-gate timeout')

        elif self.state == 'PRE_GATE_HOLD':
            if self.elapsed() >= self.pre_gate_hold_seconds:
                self.start_state('CROSS_OUT')
                self.get_logger().info(
                    'Starting low-speed outbound gate crossing.')

        elif self.state == 'CROSS_OUT':
            if self.rel_x() >= self.post_rel_x:
                self.set_goal(self.post_x, self.post_y, self.flight_z)
                self.start_state('POST_GATE_HOLD')
                self.get_logger().info(
                    'Crossed gate. Holding on far side.')
            elif self.elapsed() > self.cross_timeout_seconds:
                self.abort_land('outbound crossing timeout')

        elif self.state == 'POST_GATE_HOLD':
            if self.elapsed() >= self.post_gate_hold_seconds:
                self.start_state('CROSS_BACK')
                self.get_logger().info(
                    'Starting low-speed return through gate.')

        elif self.state == 'CROSS_BACK':
            if self.rel_x() <= self.pre_rel_x:
                self.set_goal(self.pre_x, self.pre_y, self.flight_z)
                self.start_state('PRE_GATE_RETURN_HOLD')
                self.get_logger().info(
                    'Returned to pre-gate side. Holding briefly.')
            elif self.elapsed() > self.cross_timeout_seconds:
                self.abort_land('return crossing timeout')

        elif self.state == 'PRE_GATE_RETURN_HOLD':
            if self.elapsed() >= self.pre_gate_return_hold_seconds:
                self.set_goal(self.home_x, self.home_y, self.flight_z)
                self.start_state('RETURN_HOME')
                self.get_logger().info('Returning above home point.')

        elif self.state == 'RETURN_HOME':
            if self.reached_position_goal():
                self.start_state('FINAL_HOLD')
                self.get_logger().info('Home reached. Final hold started.')
            elif self.elapsed() > self.position_timeout_seconds:
                self.abort_land('return-home timeout')

        elif self.state == 'FINAL_HOLD':
            if self.elapsed() >= self.final_hold_seconds and not self.land_requested:
                self.request_land()
                self.land_requested = True
                self.start_state('LANDING')

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = GateLowSpeedTraversal()
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
