#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus


class GateVerticalCalibration(Node):
    """
    门框前纵向三点标定：LOW / MID / HIGH。
    保持已经验证过的横向中心线，不穿门。
    """

    def __init__(self):
        super().__init__('offboard_gate_vertical_calibration')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.position_sub = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.position_callback, qos)
        self.status_sub = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.status_callback, qos)

        self.position = VehicleLocalPosition()
        self.status = VehicleStatus()

        self.home_x = self.home_y = self.home_z = None
        self.goal_x = self.goal_y = self.goal_z = None

        # 已验证的门前横向中心位置（相对于 home）
        self.gate_plane_x_min = 2.08
        self.gate_center_y = -0.46
        self.pre_gate_safety_distance = 0.85

        # 三个高度。NED 中程序会自动变成 home_z - height。
        self.height_tests = [
            ('LOW_TEST', 0.45),
            ('MID_TEST', 0.70),
            ('HIGH_TEST', 0.95),
        ]

        # 先到最低测试高度，再开始飞到门前。
        self.initial_takeoff_height = 0.45

        # 普通航点允许的三维距离误差。
        self.horizontal_reached_threshold = 0.15

        # 高度标定必须更严格。
        self.vertical_reached_threshold = 0.05
                
        self.point_hold_seconds = 5.0
        self.return_hold_seconds = 3.0

        self.state = 'WAIT_POSITION'
        self.points = []
        self.index = 0
        self.counter = 0
        self.last_log_counter = -10
        self.hold_start_counter = None
        self.land_requested = False

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(
            'Gate vertical calibration started. '
            'Waiting for PX4 local-position data...')

    def now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def position_callback(self, msg):
        self.position = msg

    def status_callback(self, msg):
        self.status = msg

    def set_goal(self, x, y, z):
        self.goal_x, self.goal_y, self.goal_z = float(x), float(y), float(z)

    def distance_to_goal(self):
        dx = self.position.x - self.goal_x
        dy = self.position.y - self.goal_y
        dz = self.position.z - self.goal_z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def reached_goal_precisely(self):
        """
        用更严格的条件判断是否真正稳定到达标定点。

        水平方向允许 0.15 m，
        高度方向只允许 0.05 m。
        """
        horizontal_error = math.sqrt(
            (self.position.x - self.goal_x) ** 2
            + (self.position.y - self.goal_y) ** 2
        )

        vertical_error = abs(self.position.z - self.goal_z)

        return (
            horizontal_error <= self.horizontal_reached_threshold
            and vertical_error <= self.vertical_reached_threshold
        )
    
    def publish_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.now_us()
        self.offboard_mode_pub.publish(msg)

    def publish_setpoint(self):
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

    def publish_command(self, command, param1=0.0, param2=0.0):
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
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.get_logger().info('Requested OFFBOARD mode')

    def request_arm(self):
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info('Requested ARM')

    def request_land(self):
        self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Requested AUTO LAND')

    def build_points(self):
        rel_x = self.gate_plane_x_min - self.pre_gate_safety_distance
        rel_y = self.gate_center_y
        self.points = []
        for name, height in self.height_tests:
            self.points.append({
                'name': name,
                'height': height,
                'rel_x': rel_x,
                'rel_y': rel_y,
                'x': self.home_x + rel_x,
                'y': self.home_y + rel_y,
                'z': self.home_z - height,
            })

    def print_plan(self):
        self.get_logger().info(
            'Stable pre-gate point relative to home: '
            f'X={self.points[0]["rel_x"]:.2f}, '
            f'Y={self.points[0]["rel_y"]:.2f}')
        for i, p in enumerate(self.points, start=1):
            self.get_logger().info(
                f'{i}. {p["name"]}: desired height={p["height"]:.2f} m | '
                f'target=({p["x"]:.2f}, {p["y"]:.2f}, {p["z"]:.2f})')

    def set_current_point(self):
        p = self.points[self.index]
        self.set_goal(p['x'], p['y'], p['z'])
        self.get_logger().info(
            f'Flying to {p["name"]}: desired height={p["height"]:.2f} m')

    def log_progress(self):
        if self.state in ('LANDING', 'MISSION_FINISHED'):
            self.get_logger().info(
                f'[{self.state}] Current: '
                f'x={self.position.x:.2f}, y={self.position.y:.2f}, '
                f'z={self.position.z:.2f}')
            return

        if self.goal_x is None:
            return

        rel_x = self.position.x - self.home_x
        rel_y = self.position.y - self.home_y
        text = (
            f'[{self.state}] Current: '
            f'x={self.position.x:.2f}, y={self.position.y:.2f}, z={self.position.z:.2f} | '
            f'Relative: x={rel_x:.2f}, y={rel_y:.2f} | '
            f'Goal: x={self.goal_x:.2f}, y={self.goal_y:.2f}, z={self.goal_z:.2f} | '
            f'Distance: {self.distance_to_goal():.2f} m'
        )
        if self.state in ('FLY_TO_TEST', 'HOLD_TEST'):
            p = self.points[self.index]
            text += f' | {p["name"]}, desired height={p["height"]:.2f} m'
        self.get_logger().info(text)

    def timer_callback(self):
        if self.state not in ('LANDING', 'MISSION_FINISHED'):
            self.publish_heartbeat()

        if self.state == 'WAIT_POSITION':
            if self.position.timestamp == 0:
                return

            self.home_x = self.position.x
            self.home_y = self.position.y
            self.home_z = self.position.z
            self.build_points()
            self.set_goal(
                self.home_x, self.home_y,
                self.home_z - self.initial_takeoff_height)
            self.state = 'TAKEOFF'

            self.get_logger().info(
                f'Captured home point: x={self.home_x:.2f}, '
                f'y={self.home_y:.2f}, z={self.home_z:.2f}')
            self.get_logger().info(
                f'Initial takeoff target: x={self.goal_x:.2f}, '
                f'y={self.goal_y:.2f}, z={self.goal_z:.2f}')
            self.print_plan()

        if self.state in (
            'TAKEOFF', 'FLY_TO_TEST', 'HOLD_TEST',
            'RETURN_HOME', 'FINAL_HOLD'):
            self.publish_setpoint()

        if self.counter - self.last_log_counter >= 10:
            self.log_progress()
            self.last_log_counter = self.counter

        if self.counter < 10:
            self.counter += 1
            return

        if self.state == 'LANDING':
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.state = 'MISSION_FINISHED'
                self.get_logger().info(
                    'Vehicle disarmed. Gate vertical calibration finished.')
            self.counter += 1
            return

        if self.state == 'MISSION_FINISHED':
            self.counter += 1
            return

        if self.counter % 10 == 0:
            if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.request_offboard()

            elif self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.request_arm()

            elif self.state == 'TAKEOFF':
                if self.reached_goal_precisely():
                    self.state = 'FLY_TO_TEST'
                    self.index = 0
                    self.get_logger().info(
                        'Reached initial height. Starting LOW-MID-HIGH tests.')
                    self.set_current_point()

            elif self.state == 'FLY_TO_TEST':
                if self.reached_goal_precisely():
                    p = self.points[self.index]
                    self.state = 'HOLD_TEST'
                    self.hold_start_counter = self.counter
                    self.get_logger().info(
                        f'Reached {p["name"]}. Holding for '
                        f'{self.point_hold_seconds:.1f} seconds. '
                        'Observe top-bar and lower-frame clearance.')

            elif self.state == 'HOLD_TEST':
                if self.counter - self.hold_start_counter >= int(self.point_hold_seconds * 10):
                    self.index += 1
                    if self.index < len(self.points):
                        self.state = 'FLY_TO_TEST'
                        self.set_current_point()
                    else:
                        self.set_goal(
                            self.home_x, self.home_y,
                            self.home_z - self.initial_takeoff_height)
                        self.state = 'RETURN_HOME'
                        self.get_logger().info(
                            'All vertical tests complete. Returning above home.')

            elif self.state == 'RETURN_HOME':
                if self.reached_goal_precisely():
                    self.state = 'FINAL_HOLD'
                    self.hold_start_counter = self.counter
                    self.get_logger().info(
                        f'Returned above home. Holding for '
                        f'{self.return_hold_seconds:.1f} seconds.')

            elif self.state == 'FINAL_HOLD':
                if (
                    self.counter - self.hold_start_counter >= int(self.return_hold_seconds * 10)
                    and not self.land_requested
                ):
                    self.request_land()
                    self.land_requested = True
                    self.state = 'LANDING'

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = GateVerticalCalibration()
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
