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


class OffboardSingleWaypoint(Node):
    """
    飞行任务：
    1. 起飞到起点上方 1.5 米；
    2. 向本地 x 正方向飞 3 米；
    3. 在航点处持续悬停。
    """

    def __init__(self):
        super().__init__('offboard_single_waypoint')

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

        # 起飞时记录的原始位置
        self.home_x = None
        self.home_y = None
        self.home_z = None

        # 当前持续发送给 PX4 的目标点
        self.goal_x = None
        self.goal_y = None
        self.goal_z = None

        # 任务参数
        self.takeoff_height = 1.5       # 向上飞 1.5 米
        self.forward_distance = 1.0     # 向 x 正方向飞 1 米
        self.reached_threshold = 0.30   # 距离目标小于 0.3 米认为到达

        # 当前任务阶段
        self.mission_state = 'WAIT_POSITION'

        self.counter = 0

        # 每秒打印一次当前位置、目标点和距离。
        self.last_progress_log_counter = -10

        # 每 0.1 秒运行一次，即 10 Hz。
        self.timer = self.create_timer(0.1, self.timer_callback)

    def now_us(self):
        """生成 PX4 需要的微秒时间戳。"""
        return int(self.get_clock().now().nanoseconds / 1000)

    def position_callback(self, msg):
        self.position = msg

    def status_callback(self, msg):
        self.status = msg

    def publish_heartbeat(self):
        """
        告诉 PX4：
        当前外部程序正在持续发送“位置控制”指令。
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
        发布当前目标点。
        PX4 使用 NED 坐标：
        x 北向，y 东向，z 向下。
        因此向上飞时 z 会变小。
        """
        msg = TrajectorySetpoint()

        msg.position = [
            float(self.goal_x),
            float(self.goal_y),
            float(self.goal_z),
        ]

        # 这里只做位置控制，不额外提供速度、加速度前馈。
        nan = float('nan')
        msg.velocity = [nan, nan, nan]
        msg.acceleration = [nan, nan, nan]

        msg.yaw = 0.0
        msg.yawspeed = nan
        msg.timestamp = self.now_us()

        self.setpoint_pub.publish(msg)

    def publish_command(self, command, param1=0.0, param2=0.0):
        """发布 PX4 命令，例如切 Offboard、解锁、降落。"""
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
        """请求自动降落。"""
        self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Requested AUTO LAND')

    def distance_to_goal(self):
        """计算当前位置与当前目标点的三维距离。"""
        dx = self.position.x - self.goal_x
        dy = self.position.y - self.goal_y
        dz = self.position.z - self.goal_z

        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    def log_progress(self):
        """每秒打印一次当前坐标、目标坐标与距离。"""
        if self.goal_x is None:
            return

        distance = self.distance_to_goal()

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
    )

    def timer_callback(self):
        # Offboard 模式下，必须一直发送心跳。
        self.publish_heartbeat()

        # 第一次收到有效位置时，记录起点，并设置“起飞目标点”。
        if self.mission_state == 'WAIT_POSITION':
            if self.position.timestamp == 0:
                return

            self.home_x = self.position.x
            self.home_y = self.position.y
            self.home_z = self.position.z

            # 起飞：保持 x、y 不动，只向上飞。
            self.goal_x = self.home_x
            self.goal_y = self.home_y
            self.goal_z = self.home_z - self.takeoff_height

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

        # 有目标点后，持续发给 PX4。
        self.publish_setpoint()

        # 每 10 次循环打印一次。定时器频率为 10 Hz，因此约每秒打印一次。
        if self.counter - self.last_progress_log_counter >= 10:
            self.log_progress()
            self.last_progress_log_counter = self.counter

        # 先连续发送约 1 秒心跳与目标点。
        if self.counter < 10:
            self.counter += 1
            return

        # 每秒请求一次 Offboard 或 Arm，避免终端刷屏。
        if self.counter % 10 == 0:
            if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.request_offboard()

            elif self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.request_arm()

            # 已解锁后才进入真正的任务阶段。
            elif self.mission_state == 'TAKEOFF':
                distance = self.distance_to_goal()

                if distance < self.reached_threshold:
                    # 起飞完成：把目标点改成“前方 3 米”。
                    self.goal_x = self.home_x 
                    self.goal_y = self.home_y + self.forward_distance
                    self.goal_z = self.home_z - self.takeoff_height

                    self.mission_state = 'FLY_TO_WAYPOINT'

                    self.get_logger().info(
                                        'Takeoff completed. '
                    f'Moving along +Y by {self.forward_distance:.1f} m.'
                    )

                    self.get_logger().info(
                        'Waypoint target: '
                        f'x={self.goal_x:.2f}, '
                        f'y={self.goal_y:.2f}, '
                        f'z={self.goal_z:.2f}'
                    )

            elif self.mission_state == 'FLY_TO_WAYPOINT':
                distance = self.distance_to_goal()

                if distance < self.reached_threshold:
                    self.mission_state = 'HOLD_WAYPOINT'

                    self.get_logger().info(
                        'Waypoint reached. Holding position.'
                    )

            elif self.mission_state == 'HOLD_WAYPOINT':
                # 什么都不切换，持续发送同一个目标点，即悬停。
                pass

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = OffboardSingleWaypoint()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info(
            'Ctrl+C received. Requesting AUTO LAND...'
        )
        node.request_land()

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()