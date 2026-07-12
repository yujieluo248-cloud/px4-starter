#!/usr/bin/env python3

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


class OffboardTakeoffHover(Node):
    """PX4 Offboard: 起飞到指定高度并保持悬停。"""

    def __init__(self):
        super().__init__('offboard_takeoff_hover')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ROS 2 -> PX4
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

        # PX4 -> ROS 2
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

        self.home_x = None
        self.home_y = None
        self.target_z = None

        # 向上飞 1.5 m。PX4 NED 坐标中，向上是 z 变小。
        self.takeoff_height = 1.5

        self.counter = 0
        self.hover_reported = False

        # 10 Hz 持续发布 Offboard 心跳与位置目标。
        self.timer = self.create_timer(0.1, self.timer_callback)

    def now_us(self):
        """生成 PX4 所需的微秒时间戳。"""
        return int(self.get_clock().now().nanoseconds / 1000)

    def position_callback(self, msg):
        self.position = msg

    def status_callback(self, msg):
        self.status = msg

    def publish_heartbeat(self):
        """声明外部控制器正在发送位置控制指令。"""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.now_us()

        self.offboard_mode_pub.publish(msg)

    def publish_setpoint(self):
        """持续发送同一个局部 NED 位置目标。"""
        msg = TrajectorySetpoint()

        msg.position = [
            float(self.home_x),
            float(self.home_y),
            float(self.target_z),
        ]

        msg.yaw = 0.0
        msg.timestamp = self.now_us()

        self.setpoint_pub.publish(msg)

    def publish_command(self, command, param1=0.0, param2=0.0):
        """向 PX4 发送模式切换、解锁等 VehicleCommand。"""
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
        """请求 PX4 进入 Offboard 模式。"""
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )

        self.get_logger().info('Requested OFFBOARD mode')

    def request_arm(self):
        """请求 PX4 解锁。"""
        self.publish_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )

        self.get_logger().info('Requested ARM')

    def request_land(self):
        """请求 PX4 自动降落。"""
        self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Requested AUTO LAND')

    def timer_callback(self):
        self.publish_heartbeat()

        # 等待 PX4 发布有效局部位置。
        if self.home_x is None:
            if self.position.timestamp == 0:
                return

            self.home_x = self.position.x
            self.home_y = self.position.y

            # NED: z 向下为正，因此向上飞 1.5 米要减 1.5。
            self.target_z = self.position.z - self.takeoff_height

            self.get_logger().info(
                f'Captured home point: '
                f'x={self.home_x:.2f}, '
                f'y={self.home_y:.2f}, '
                f'z={self.position.z:.2f}; '
                f'target_z={self.target_z:.2f}'
            )

        # Offboard 期间必须持续发送目标点。
        self.publish_setpoint()

        # 先连续发 10 个心跳和 setpoint，约 1 秒。
        if self.counter < 10:
            self.counter += 1
            return

        # 每秒最多请求一次，避免刷屏发送命令。
        if self.counter % 10 == 0:
            if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.request_offboard()

            elif self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.request_arm()

            elif not self.hover_reported:
                self.hover_reported = True
                self.get_logger().info(
                    f'Vehicle armed. Holding '
                    f'{self.takeoff_height:.1f} m above home point.'
                )

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = OffboardTakeoffHover()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C received. Requesting AUTO LAND...')
        node.request_land()

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()