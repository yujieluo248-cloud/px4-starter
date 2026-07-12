#!/usr/bin/env python3

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


class OffboardGateThreePointCalibration(Node):
    """
    大门框前左、中、右三点标定任务。

    飞行流程：
    1. 原地起飞到约 0.9 m；
    2. 飞到门框前的左侧标定点，悬停；
    3. 飞到门框前的中心标定点，悬停；
    4. 飞到门框前的右侧标定点，悬停；
    5. 返回起飞点上方；
    6. 自动降落。

    注意：
    本程序不会穿门。
    它的目的是帮助观察门洞横向余量和真正安全中心线。
    """

    def __init__(self):
        super().__init__('offboard_gate_three_point_calibration')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---------- ROS 2 -> PX4 ----------
        self.offboard_mode_pub = self.create_publisher(
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

        # ---------- PX4 -> ROS 2 ----------
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos,
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            qos,
        )

        self.position = VehicleLocalPosition()
        self.status = VehicleStatus()

        # ---------- 起飞点 ----------
        self.home_x = None
        self.home_y = None
        self.home_z = None

        # ---------- 当前目标点 ----------
        self.goal_x = None
        self.goal_y = None
        self.goal_z = None

        # ---------- 飞行参数 ----------
        self.takeoff_height = 0.9
        self.reached_threshold = 0.25

        # 每个标定点停留时间。
        self.point_hold_seconds = 5.0

        # 返回起点后的悬停时间。
        self.return_hold_seconds = 3.0

        # ---------- 大门框已知近似位置 ----------
        # 这些参数来自前面的场景解析和坐标标定。
        #
        # 门框平面大致位于相对起飞点：
        # X = +2.08 ~ +2.38
        #
        # 门框横向中心大致位于：
        # Y = -0.46
        self.gate_plane_x_min = 2.08
        self.gate_plane_x_max = 2.38
        self.gate_center_y = -0.46

        # 停在门前多远，不进入门框平面。
        self.pre_gate_safety_distance = 0.85

        # 左、中、右点相对“门洞中心线”的横向偏移。
        #
        # 注意：
        # 这里的 left / center / right 指的是地图坐标意义，
        # 最终以 Gazebo 画面里看到的门洞左右为准。
        self.lateral_offsets = [
            ('LEFT_TEST', -0.20),
            ('CENTER_TEST', 0.00),
            ('RIGHT_TEST', 0.20),
        ]

        # ---------- 状态 ----------
        self.mission_state = 'WAIT_POSITION'

        self.calibration_points = []
        self.point_index = 0

        self.counter = 0
        self.last_progress_log_counter = -10

        self.hold_start_counter = None
        self.land_requested = False

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            'Gate three-point calibration node started. '
            'Waiting for PX4 local-position data...'
        )

    def now_us(self):
        """生成 PX4 所需的微秒时间戳。"""
        return int(self.get_clock().now().nanoseconds / 1000)

    def position_callback(self, msg):
        self.position = msg

    def status_callback(self, msg):
        self.status = msg

    def publish_heartbeat(self):
        """持续声明当前节点使用 Offboard 位置控制。"""
        msg = OffboardControlMode()

        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.now_us()

        self.offboard_mode_pub.publish(msg)

    def publish_setpoint(self):
        """持续发布当前位置阶段的目标点。"""
        if self.goal_x is None:
            return

        nan = float('nan')

        msg = TrajectorySetpoint()
        msg.position = [
            float(self.goal_x),
            float(self.goal_y),
            float(self.goal_z),
        ]

        # 当前只做位置控制。
        msg.velocity = [nan, nan, nan]
        msg.acceleration = [nan, nan, nan]

        # 不强制指定机头方向。
        msg.yaw = nan
        msg.yawspeed = nan
        msg.timestamp = self.now_us()

        self.setpoint_pub.publish(msg)

    def publish_command(self, command, param1=0.0, param2=0.0):
        """向 PX4 发布模式、解锁、降落等命令。"""
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
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )
        self.get_logger().info('Requested OFFBOARD mode')

    def request_arm(self):
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )
        self.get_logger().info('Requested ARM')

    def request_land(self):
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND
        )
        self.get_logger().info('Requested AUTO LAND')

    def set_goal(self, x, y, z):
        """设置当前目标点。"""
        self.goal_x = float(x)
        self.goal_y = float(y)
        self.goal_z = float(z)

    def distance_to_goal(self):
        """计算当前位置到目标点的三维距离。"""
        dx = self.position.x - self.goal_x
        dy = self.position.y - self.goal_y
        dz = self.position.z - self.goal_z

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def relative_position(self):
        """返回无人机相对于起飞点的 x、y 位移。"""
        return (
            self.position.x - self.home_x,
            self.position.y - self.home_y,
        )

    def print_gate_information(self):
        """打印本次标定使用的门框与测试点信息。"""
        staging_x = (
            self.gate_plane_x_min
            - self.pre_gate_safety_distance
        )

        self.get_logger().info(
            'Large gate relative map: '
            f'plane X=[{self.gate_plane_x_min:.2f}, '
            f'{self.gate_plane_x_max:.2f}], '
            f'center Y={self.gate_center_y:.2f}'
        )

        self.get_logger().info(
            'Safe pre-gate X coordinate: '
            f'{staging_x:.2f} m relative to home'
        )

        self.get_logger().info(
            'Three-point lateral offsets: '
            'LEFT=-0.20 m, CENTER=0.00 m, RIGHT=+0.20 m'
        )

    def build_calibration_points(self):
        """创建门框前左、中、右三个悬停标定点。"""
        staging_rel_x = (
            self.gate_plane_x_min
            - self.pre_gate_safety_distance
        )

        target_z = self.home_z - self.takeoff_height

        self.calibration_points = []

        for name, offset_y in self.lateral_offsets:
            rel_x = staging_rel_x
            rel_y = self.gate_center_y + offset_y

            self.calibration_points.append(
                {
                    'name': name,
                    'offset_y': offset_y,
                    'rel_x': rel_x,
                    'rel_y': rel_y,
                    'x': self.home_x + rel_x,
                    'y': self.home_y + rel_y,
                    'z': target_z,
                }
            )

    def print_calibration_points(self):
        """把三个标定点打印到终端。"""
        self.get_logger().info('Three gate-calibration points created:')

        for index, point in enumerate(
            self.calibration_points,
            start=1,
        ):
            self.get_logger().info(
                f"  {index}. {point['name']} | "
                f"lateral_offset={point['offset_y']:+.2f} m | "
                f"relative=({point['rel_x']:.2f}, "
                f"{point['rel_y']:.2f}) | "
                f"target=({point['x']:.2f}, "
                f"{point['y']:.2f}, "
                f"{point['z']:.2f})"
            )

    def set_current_calibration_point(self):
        """设置当前左/中/右标定点。"""
        point = self.calibration_points[self.point_index]

        self.set_goal(
            point['x'],
            point['y'],
            point['z'],
        )

        self.get_logger().info(
            f"Flying to calibration point "
            f"{self.point_index + 1}/"
            f"{len(self.calibration_points)}: "
            f"{point['name']} | "
            f"lateral_offset={point['offset_y']:+.2f} m"
        )

    def start_three_point_calibration(self):
        """起飞完成后开始左、中、右三点标定。"""
        self.point_index = 0
        self.mission_state = 'FLY_TO_CALIBRATION_POINT'

        self.get_logger().info(
            'Takeoff completed. '
            'Starting left-center-right gate calibration.'
        )

        self.set_current_calibration_point()

    def begin_point_hold(self):
        """抵达一个标定点后开始悬停。"""
        point = self.calibration_points[self.point_index]

        self.mission_state = 'HOLD_CALIBRATION_POINT'
        self.hold_start_counter = self.counter

        self.get_logger().info(
            f"Reached {point['name']}. "
            f"Holding for {self.point_hold_seconds:.1f} seconds. "
            'Observe the distance to both gate pillars.'
        )

    def advance_calibration_point(self):
        """当前点悬停完成后，切换到下一个点或返航。"""
        self.point_index += 1

        if self.point_index >= len(self.calibration_points):
            self.return_home()
            return

        self.mission_state = 'FLY_TO_CALIBRATION_POINT'
        self.set_current_calibration_point()

    def return_home(self):
        """回到起飞点正上方。"""
        self.set_goal(
            self.home_x,
            self.home_y,
            self.home_z - self.takeoff_height,
        )

        self.mission_state = 'RETURN_HOME'

        self.get_logger().info(
            'All three gate points completed. '
            'Returning above home point.'
        )

    def log_progress(self):
        """每秒打印当前位置、目标点、距离及当前标定阶段。"""
        if self.mission_state in ('LANDING', 'MISSION_FINISHED'):
            self.get_logger().info(
                f'[{self.mission_state}] '
                f'Current: x={self.position.x:.2f}, '
                f'y={self.position.y:.2f}, '
                f'z={self.position.z:.2f}'
            )
            return

        if self.goal_x is None:
            return

        rel_x, rel_y = self.relative_position()
        distance = self.distance_to_goal()

        message = (
            f'[{self.mission_state}] '
            f'Current: '
            f'x={self.position.x:.2f}, '
            f'y={self.position.y:.2f}, '
            f'z={self.position.z:.2f} | '
            f'Relative: '
            f'x={rel_x:.2f}, '
            f'y={rel_y:.2f} | '
            f'Goal: '
            f'x={self.goal_x:.2f}, '
            f'y={self.goal_y:.2f}, '
            f'z={self.goal_z:.2f} | '
            f'Distance: {distance:.2f} m'
        )

        if self.mission_state in (
            'FLY_TO_CALIBRATION_POINT',
            'HOLD_CALIBRATION_POINT',
        ):
            point = self.calibration_points[self.point_index]

            message += (
                f" | Point: {point['name']} "
                f"| Offset: {point['offset_y']:+.2f} m"
            )

        elif self.mission_state == 'HOLD_CALIBRATION_POINT':
            elapsed = (
                self.counter - self.hold_start_counter
            ) / 10.0

            message += (
                f' | Hold: {elapsed:.1f}/'
                f'{self.point_hold_seconds:.1f} s'
            )

        elif self.mission_state == 'FINAL_HOLD':
            elapsed = (
                self.counter - self.hold_start_counter
            ) / 10.0

            message += (
                f' | Home hold: {elapsed:.1f}/'
                f'{self.return_hold_seconds:.1f} s'
            )

        self.get_logger().info(message)

    def timer_callback(self):
        """
        状态机：

        WAIT_POSITION
        → TAKEOFF
        → FLY_TO_CALIBRATION_POINT
        → HOLD_CALIBRATION_POINT
        → RETURN_HOME
        → FINAL_HOLD
        → LANDING
        → MISSION_FINISHED
        """

        if self.mission_state not in (
            'LANDING',
            'MISSION_FINISHED',
        ):
            self.publish_heartbeat()

        if self.mission_state == 'WAIT_POSITION':
            if self.position.timestamp == 0:
                return

            self.home_x = self.position.x
            self.home_y = self.position.y
            self.home_z = self.position.z

            self.set_goal(
                self.home_x,
                self.home_y,
                self.home_z - self.takeoff_height,
            )

            self.build_calibration_points()

            self.mission_state = 'TAKEOFF'

            self.get_logger().info(
                'Captured home point: '
                f'x={self.home_x:.2f}, '
                f'y={self.home_y:.2f}, '
                f'z={self.home_z:.2f}'
            )

            self.get_logger().info(
                'Takeoff target: '
                f'x={self.goal_x:.2f}, '
                f'y={self.goal_y:.2f}, '
                f'z={self.goal_z:.2f}'
            )

            self.print_gate_information()
            self.print_calibration_points()

        if self.mission_state in (
            'TAKEOFF',
            'FLY_TO_CALIBRATION_POINT',
            'HOLD_CALIBRATION_POINT',
            'RETURN_HOME',
            'FINAL_HOLD',
        ):
            self.publish_setpoint()

        if self.counter - self.last_progress_log_counter >= 10:
            self.log_progress()
            self.last_progress_log_counter = self.counter

        # 前 1 秒只持续预发送 heartbeat 和 setpoint。
        if self.counter < 10:
            self.counter += 1
            return

        if self.mission_state == 'LANDING':
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.mission_state = 'MISSION_FINISHED'

                self.get_logger().info(
                    'Vehicle disarmed. '
                    'Three-point gate calibration finished.'
                )

            self.counter += 1
            return

        if self.mission_state == 'MISSION_FINISHED':
            self.counter += 1
            return

        if self.counter % 10 == 0:
            if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.request_offboard()

            elif self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.request_arm()

            elif self.mission_state == 'TAKEOFF':
                if self.distance_to_goal() < self.reached_threshold:
                    self.start_three_point_calibration()

            elif self.mission_state == 'FLY_TO_CALIBRATION_POINT':
                if self.distance_to_goal() < self.reached_threshold:
                    self.begin_point_hold()

            elif self.mission_state == 'HOLD_CALIBRATION_POINT':
                hold_counter = int(self.point_hold_seconds * 10)

                if (
                    self.counter - self.hold_start_counter
                    >= hold_counter
                ):
                    self.advance_calibration_point()

            elif self.mission_state == 'RETURN_HOME':
                if self.distance_to_goal() < self.reached_threshold:
                    self.mission_state = 'FINAL_HOLD'
                    self.hold_start_counter = self.counter

                    self.get_logger().info(
                        'Returned above home point. '
                        f'Holding for '
                        f'{self.return_hold_seconds:.1f} seconds.'
                    )

            elif self.mission_state == 'FINAL_HOLD':
                hold_counter = int(self.return_hold_seconds * 10)

                if (
                    self.counter - self.hold_start_counter
                    >= hold_counter
                    and not self.land_requested
                ):
                    self.request_land()
                    self.land_requested = True
                    self.mission_state = 'LANDING'

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = OffboardGateThreePointCalibration()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        # Ctrl+C 时不额外发送命令，
        # 避免 ROS context 已关闭后继续发布导致报错。
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()