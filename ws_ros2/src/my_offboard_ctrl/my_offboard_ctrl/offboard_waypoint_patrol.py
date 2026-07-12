#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)


class OffboardWaypointPatrol(Node):
    """
    安全往返巡逻任务：

    1. 原地起飞到 1.5 米；
    2. 沿世界坐标 +Y 方向移动 1 米；
    3. 回到起飞点正上方；
    4. 再次移动到 +Y 方向 1 米；
    5. 再次回到起飞点正上方；
    6. 悬停 4 秒；
    7. 自动降落。
    """

    def __init__(self):
        super().__init__('offboard_waypoint_patrol')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---------------- ROS 2 -> PX4 ----------------
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

        # ---------------- PX4 -> ROS 2 ----------------
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

        # 起飞前记录的当前位置。
        self.home_x = None
        self.home_y = None
        self.home_z = None

        # 当前持续发送给 PX4 的目标点。
        self.goal_x = None
        self.goal_y = None
        self.goal_z = None

        # ---------------- 可调任务参数 ----------------
        self.takeoff_height = 1.5
        self.patrol_distance = 1.0
        self.reached_threshold = 0.30

        # 巡逻结束后，在最后一个点悬停多少秒再降落。
        self.final_hold_seconds = 4.0

        # ---------------- 任务状态 ----------------
        self.mission_state = 'WAIT_POSITION'

        # 用于依次访问 P1、P2、P3、P4。
        self.waypoints = []
        self.waypoint_index = 0

        # 每 0.1 秒一次，即 10 Hz。
        self.counter = 0

        # 用于每秒打印一次进度。
        self.last_progress_log_counter = -10

        # 最后悬停开始的循环编号。
        self.final_hold_start_counter = None

        # 降落命令只发送一次。
        self.land_requested = False

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            'Waypoint patrol node started. '
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
        """
        告诉 PX4：
        外部程序正在持续使用 Offboard 位置控制。
        """
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.now_us()

        self.offboard_mode_pub.publish(msg)

    def publish_setpoint(self):
        """
        发布当前位置阶段对应的目标点。

        PX4 使用本地 NED 坐标：
        x：北向
        y：东向
        z：向下

        因此 z 变小表示无人机向上飞。
        """
        if self.goal_x is None:
            return

        msg = TrajectorySetpoint()

        msg.position = [
            float(self.goal_x),
            float(self.goal_y),
            float(self.goal_z),
        ]

        nan = float('nan')

        # 这里只做位置控制，不提供速度和加速度前馈。
        msg.velocity = [nan, nan, nan]
        msg.acceleration = [nan, nan, nan]

        # 保持当前朝向，不强制转向。
        msg.yaw = nan
        msg.yawspeed = nan
        msg.timestamp = self.now_us()

        self.setpoint_pub.publish(msg)

    def publish_command(self, command, param1=0.0, param2=0.0):
        """发送 PX4 指令，例如 Offboard、解锁和降落。"""
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

    def distance_to_goal(self):
        """计算当前位置与当前目标点之间的三维距离。"""
        dx = self.position.x - self.goal_x
        dy = self.position.y - self.goal_y
        dz = self.position.z - self.goal_z

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def set_goal(self, x, y, z):
        """切换当前飞行目标点。"""
        self.goal_x = float(x)
        self.goal_y = float(y)
        self.goal_z = float(z)

    def print_waypoints(self):
        """把本次巡逻路线打印到终端。"""
        self.get_logger().info('Patrol route created:')

        for index, waypoint in enumerate(self.waypoints, start=1):
            x, y, z = waypoint

            self.get_logger().info(
                f'  P{index}: '
                f'x={x:.2f}, '
                f'y={y:.2f}, '
                f'z={z:.2f}'
            )

    def start_patrol(self):
        """起飞完成后，开始执行第一个巡逻点。"""
        self.waypoint_index = 0

        x, y, z = self.waypoints[self.waypoint_index]
        self.set_goal(x, y, z)

        self.mission_state = 'PATROL'

        self.get_logger().info(
            f'Takeoff completed. '
            f'Starting patrol at P{self.waypoint_index + 1}.'
        )

        self.get_logger().info(
            f'Current target P{self.waypoint_index + 1}: '
            f'x={self.goal_x:.2f}, '
            f'y={self.goal_y:.2f}, '
            f'z={self.goal_z:.2f}'
        )

    def go_to_next_waypoint(self):
        """巡逻点到达后，切换到下一个点。"""
        reached_number = self.waypoint_index + 1

        self.get_logger().info(
            f'Reached patrol waypoint P{reached_number}.'
        )

        self.waypoint_index += 1

        # 所有航点完成。
        if self.waypoint_index >= len(self.waypoints):
            self.mission_state = 'FINAL_HOLD'
            self.final_hold_start_counter = self.counter

            self.get_logger().info(
                'Patrol route completed. '
                f'Holding for {self.final_hold_seconds:.1f} seconds '
                'before landing.'
            )
            return

        # 继续飞向下一个巡逻点。
        x, y, z = self.waypoints[self.waypoint_index]
        self.set_goal(x, y, z)

        self.get_logger().info(
            f'Flying to patrol waypoint P{self.waypoint_index + 1}: '
            f'x={self.goal_x:.2f}, '
            f'y={self.goal_y:.2f}, '
            f'z={self.goal_z:.2f}'
        )

    def log_progress(self):
        """每秒打印一次位置、目标点、距离与状态。"""
        if self.goal_x is None:
            return

        distance = self.distance_to_goal()

        extra_info = ''

        if self.mission_state == 'PATROL':
            extra_info = (
                f' | Patrol point: '
                f'P{self.waypoint_index + 1}/{len(self.waypoints)}'
            )

        elif self.mission_state == 'FINAL_HOLD':
            extra_info = ' | Patrol complete: final hold'

        elif self.mission_state == 'LANDING':
            extra_info = ' | Auto landing in progress'

        elif self.mission_state == 'MISSION_FINISHED':
            extra_info = ' | Mission finished'

        self.get_logger().info(
            f'[{self.mission_state}] '
            f'Current: '
            f'x={self.position.x:.2f}, '
            f'y={self.position.y:.2f}, '
            f'z={self.position.z:.2f} | '
            f'Goal: '
            f'x={self.goal_x:.2f}, '
            f'y={self.goal_y:.2f}, '
            f'z={self.goal_z:.2f} | '
            f'Distance: {distance:.2f} m'
            f'{extra_info}'
        )

    def timer_callback(self):
        """
        整个任务状态机：

        WAIT_POSITION
        → TAKEOFF
        → PATROL
        → FINAL_HOLD
        → LANDING
        → MISSION_FINISHED
        """

        # 降落阶段以后，不再持续要求 PX4 保持 Offboard。
        if self.mission_state not in ('LANDING', 'MISSION_FINISHED'):
            self.publish_heartbeat()

        # 第一次拿到位置时，记录起点并建立任务路线。
        if self.mission_state == 'WAIT_POSITION':
            if self.position.timestamp == 0:
                return

            self.home_x = self.position.x
            self.home_y = self.position.y
            self.home_z = self.position.z

            takeoff_z = self.home_z - self.takeoff_height

            # 先执行垂直起飞。
            self.set_goal(
                self.home_x,
                self.home_y,
                takeoff_z,
            )

            # 已验证 +Y 方向相对安全，因此做往返巡逻。
            self.waypoints = [
                (self.home_x, self.home_y + self.patrol_distance, takeoff_z),
                (self.home_x, self.home_y, takeoff_z),
                (self.home_x, self.home_y + self.patrol_distance, takeoff_z),
                (self.home_x, self.home_y, takeoff_z),
            ]

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

            self.print_waypoints()

        # 起飞、巡逻、最后悬停阶段都持续发送目标点。
        if self.mission_state in ('TAKEOFF', 'PATROL', 'FINAL_HOLD'):
            self.publish_setpoint()

        # 每秒打印一次进度。
        if (
            self.goal_x is not None
            and self.counter - self.last_progress_log_counter >= 10
        ):
            self.log_progress()
            self.last_progress_log_counter = self.counter

        # 前约 1 秒只持续发送心跳和目标点。
        if self.counter < 10:
            self.counter += 1
            return

        # 降落中的检查。
        if self.mission_state == 'LANDING':
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.mission_state = 'MISSION_FINISHED'

                self.get_logger().info(
                    'Vehicle disarmed. Patrol mission finished.'
                )

            self.counter += 1
            return

        # 已完成则保持节点运行，方便你查看最后日志。
        if self.mission_state == 'MISSION_FINISHED':
            self.counter += 1
            return

        # 每秒检查一次飞行状态，避免频繁反复发命令。
        if self.counter % 10 == 0:
            # 还未进入 Offboard。
            if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.request_offboard()

            # 已进入 Offboard 但尚未解锁。
            elif self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.request_arm()

            # 解锁后，才开始执行任务状态机。
            elif self.mission_state == 'TAKEOFF':
                if self.distance_to_goal() < self.reached_threshold:
                    self.start_patrol()

            elif self.mission_state == 'PATROL':
                if self.distance_to_goal() < self.reached_threshold:
                    self.go_to_next_waypoint()

            elif self.mission_state == 'FINAL_HOLD':
                hold_counter = int(self.final_hold_seconds * 10)

                if (
                    self.counter - self.final_hold_start_counter
                    >= hold_counter
                    and not self.land_requested
                ):
                    self.request_land()
                    self.land_requested = True
                    self.mission_state = 'LANDING'

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = OffboardWaypointPatrol()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        # 不在 Ctrl+C 时额外发布 PX4 命令，避免 ROS 2 已关闭时的报错。
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()