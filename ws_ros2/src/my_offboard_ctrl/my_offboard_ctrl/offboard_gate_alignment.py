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


class OffboardGateAlignment(Node):
    """
    大门框前对准任务。

    任务流程：
    1. 原地起飞到约 0.9 m；
    2. 飞到大门框前方的安全等待点；
    3. 在门框前对准并悬停 8 秒；
    4. 返回起飞点上方；
    5. 自动降落。

    注意：
    本程序不会穿过门框，只用于验证：
    - 门框相对坐标是否正确；
    - 无人机是否在门洞中线前方；
    - 当前飞行高度是否适合后续穿门。
    """

    def __init__(self):
        super().__init__('offboard_gate_alignment')

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

        # ---------- 任务参数 ----------
        # 起飞高度。当前仅做“门前对准”，不需要飞得太高。
        self.takeoff_height = 0.9

        # 与目标点距离小于该值，则认为到达。
        self.reached_threshold = 0.25

        # 在门框前悬停观察的时间。
        self.gate_hold_seconds = 8.0

        # 返回起飞点上方后的悬停时间。
        self.return_hold_seconds = 3.0

        # ---------- 已知大门框地图 ----------
        # 根据 Gazebo 模型和 PX4 / Gazebo 坐标标定得出的近似值。
        #
        # 大门框所在平面大致位于：
        # relative_x = +2.08 ~ +2.38
        #
        # 大门框横向中心大致位于：
        # relative_y = -0.46
        self.gate_plane_x_min = 2.08
        self.gate_plane_x_max = 2.38
        self.gate_center_y = -0.46

        # 在门框之前保持的安全距离。
        self.pre_gate_safety_distance = 0.85

        # ---------- 状态变量 ----------
        self.mission_state = 'WAIT_POSITION'

        self.counter = 0
        self.last_progress_log_counter = -10

        self.hold_start_counter = None
        self.land_requested = False

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            'Gate alignment node started. '
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
        """持续声明当前节点采用 Offboard 位置控制。"""
        msg = OffboardControlMode()

        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.now_us()

        self.offboard_mode_pub.publish(msg)

    def publish_setpoint(self):
        """持续发布当前目标位置。"""
        if self.goal_x is None:
            return

        nan = float('nan')

        msg = TrajectorySetpoint()
        msg.position = [
            float(self.goal_x),
            float(self.goal_y),
            float(self.goal_z),
        ]

        # 只使用位置控制。
        msg.velocity = [nan, nan, nan]
        msg.acceleration = [nan, nan, nan]

        # 不强制改变机头朝向。
        msg.yaw = nan
        msg.yawspeed = nan
        msg.timestamp = self.now_us()

        self.setpoint_pub.publish(msg)

    def publish_command(self, command, param1=0.0, param2=0.0):
        """发送 PX4 命令。"""
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
        """请求进入 Offboard 模式。"""
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )

        self.get_logger().info('Requested OFFBOARD mode')

    def request_arm(self):
        """请求解锁。"""
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )

        self.get_logger().info('Requested ARM')

    def request_land(self):
        """请求 PX4 自动降落。"""
        self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        self.get_logger().info('Requested AUTO LAND')

    def set_goal(self, x, y, z):
        """设置当前目标点。"""
        self.goal_x = float(x)
        self.goal_y = float(y)
        self.goal_z = float(z)

    def distance_to_goal(self):
        """计算当前位置与目标点的三维距离。"""
        dx = self.position.x - self.goal_x
        dy = self.position.y - self.goal_y
        dz = self.position.z - self.goal_z

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def relative_position(self):
        """计算无人机相对于起飞点的 X、Y 位移。"""
        return (
            self.position.x - self.home_x,
            self.position.y - self.home_y,
        )

    def print_gate_map(self):
        """打印本次门框对准所用的地图信息。"""
        staging_x = (
            self.gate_plane_x_min
            - self.pre_gate_safety_distance
        )

        self.get_logger().info(
            'Large gate map relative to home: '
            f'gate plane X=[{self.gate_plane_x_min:.2f}, '
            f'{self.gate_plane_x_max:.2f}], '
            f'gate center Y={self.gate_center_y:.2f}'
        )

        self.get_logger().info(
            'Safe pre-gate staging point relative to home: '
            f'X={staging_x:.2f}, '
            f'Y={self.gate_center_y:.2f}'
        )

    def start_gate_alignment(self):
        """起飞完成后，飞向门框前方的对准点。"""
        staging_rel_x = (
            self.gate_plane_x_min
            - self.pre_gate_safety_distance
        )

        staging_rel_y = self.gate_center_y

        self.set_goal(
            self.home_x + staging_rel_x,
            self.home_y + staging_rel_y,
            self.home_z - self.takeoff_height,
        )

        self.mission_state = 'FLY_TO_GATE_STAGING'

        self.get_logger().info(
            'Takeoff completed. '
            'Flying to safe pre-gate staging point.'
        )

        self.get_logger().info(
            'Gate staging target: '
            f'x={self.goal_x:.2f}, '
            f'y={self.goal_y:.2f}, '
            f'z={self.goal_z:.2f}'
        )

    def return_home(self):
        """从门框前返回起飞点上方。"""
        self.set_goal(
            self.home_x,
            self.home_y,
            self.home_z - self.takeoff_height,
        )

        self.mission_state = 'RETURN_HOME'

        self.get_logger().info(
            'Gate alignment observation completed. '
            'Returning above home point.'
        )

    def log_progress(self):
        """每秒打印一次当前飞行位置与任务进度。"""
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

        if self.mission_state == 'HOLD_AT_GATE':
            elapsed = (
                self.counter - self.hold_start_counter
            ) / 10.0

            message += (
                f' | Gate observation: '
                f'{elapsed:.1f}/{self.gate_hold_seconds:.1f} s'
            )

        elif self.mission_state == 'FINAL_HOLD':
            elapsed = (
                self.counter - self.hold_start_counter
            ) / 10.0

            message += (
                f' | Home hold: '
                f'{elapsed:.1f}/{self.return_hold_seconds:.1f} s'
            )

        self.get_logger().info(message)

    def timer_callback(self):
        """
        状态机：

        WAIT_POSITION
        → TAKEOFF
        → FLY_TO_GATE_STAGING
        → HOLD_AT_GATE
        → RETURN_HOME
        → FINAL_HOLD
        → LANDING
        → MISSION_FINISHED
        """

        # 降落和结束后不再维持 Offboard。
        if self.mission_state not in (
            'LANDING',
            'MISSION_FINISHED',
        ):
            self.publish_heartbeat()

        # 第一次获得位置时，记录起飞点。
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

            self.print_gate_map()

        # 起飞、对准、返回阶段均持续发位置目标。
        if self.mission_state in (
            'TAKEOFF',
            'FLY_TO_GATE_STAGING',
            'HOLD_AT_GATE',
            'RETURN_HOME',
            'FINAL_HOLD',
        ):
            self.publish_setpoint()

        # 每秒打印一次飞行信息。
        if self.counter - self.last_progress_log_counter >= 10:
            self.log_progress()
            self.last_progress_log_counter = self.counter

        # 前 1 秒只预发送 setpoint / heartbeat。
        if self.counter < 10:
            self.counter += 1
            return

        if self.mission_state == 'LANDING':
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.mission_state = 'MISSION_FINISHED'

                self.get_logger().info(
                    'Vehicle disarmed. '
                    'Gate-alignment mission finished.'
                )

            self.counter += 1
            return

        if self.mission_state == 'MISSION_FINISHED':
            self.counter += 1
            return

        # 每秒检查一次状态和任务条件。
        if self.counter % 10 == 0:
            if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.request_offboard()

            elif self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.request_arm()

            elif self.mission_state == 'TAKEOFF':
                if self.distance_to_goal() < self.reached_threshold:
                    self.start_gate_alignment()

            elif self.mission_state == 'FLY_TO_GATE_STAGING':
                if self.distance_to_goal() < self.reached_threshold:
                    self.mission_state = 'HOLD_AT_GATE'
                    self.hold_start_counter = self.counter

                    self.get_logger().info(
                        'Reached pre-gate staging point. '
                        f'Holding for {self.gate_hold_seconds:.1f} seconds.'
                    )

            elif self.mission_state == 'HOLD_AT_GATE':
                hold_counter = int(self.gate_hold_seconds * 10)

                if (
                    self.counter - self.hold_start_counter
                    >= hold_counter
                ):
                    self.return_home()

            elif self.mission_state == 'RETURN_HOME':
                if self.distance_to_goal() < self.reached_threshold:
                    self.mission_state = 'FINAL_HOLD'
                    self.hold_start_counter = self.counter

                    self.get_logger().info(
                        'Returned above home point. '
                        f'Holding for {self.return_hold_seconds:.1f} seconds.'
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
    node = OffboardGateAlignment()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        # Ctrl+C 时不额外发送指令，
        # 避免 ROS 2 关闭上下文后报错。
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()