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


class ObstacleAwarePatrol(Node):
    """
    已知地图约束下的障碍物感知巡逻。

    已解析出的主要障碍墙，相对起飞点的大致范围：
        X: +2.08 m ~ +2.38 m
        Y: -1.06 m ~ +1.39 m
        高度最高约 1.85 m

    本程序不会穿过障碍墙，而是在其前方的安全区域巡逻。
    每个航点和每一段航线都会先经过安全检查。
    """

    def __init__(self):
        super().__init__('offboard_obstacle_aware_patrol')

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

        # ---------------- 起飞点 ----------------
        self.home_x = None
        self.home_y = None
        self.home_z = None

        # ---------------- 当前目标点 ----------------
        self.goal_x = None
        self.goal_y = None
        self.goal_z = None

        # ---------------- 飞行参数 ----------------
        self.takeoff_height = 1.6
        self.reached_threshold = 0.30

        # 每个巡逻点到达后停留多久。
        self.waypoint_hold_seconds = 2.0

        # 完整矩形路线重复几轮。
        self.patrol_rounds = 2

        # ---------------- 已知障碍物地图 ----------------
        # 这些值是“相对于起飞点”的 PX4 本地 NED 坐标。
        #
        # 障碍墙本体：
        # x: +2.08 ~ +2.38
        # y: -1.06 ~ +1.39
        self.obstacle_x_min = 2.08
        self.obstacle_x_max = 2.38
        self.obstacle_y_min = -1.06
        self.obstacle_y_max = 1.39

        # 给障碍物额外留出安全边界。
        self.safety_margin = 0.65

        # ---------------- 状态变量 ----------------
        self.mission_state = 'WAIT_POSITION'

        # 每个元素格式：
        # (名称, 相对 home 的 x, 相对 home 的 y)
        self.route_template = []
        self.route = []
        self.route_index = 0
        self.current_round = 1

        # 到点停留的开始时刻。
        self.hold_start_counter = None

        # 计时器与日志。
        self.counter = 0
        self.last_progress_log_counter = -10

        # 防止重复发送降落指令。
        self.land_requested = False

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            'Obstacle-aware patrol node started. '
            'Waiting for PX4 local-position data...'
        )

    def now_us(self):
        """生成 PX4 使用的微秒时间戳。"""
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
        """向 PX4 持续发送当前位置阶段的目标点。"""
        if self.goal_x is None:
            return

        nan = float('nan')

        msg = TrajectorySetpoint()
        msg.position = [
            float(self.goal_x),
            float(self.goal_y),
            float(self.goal_z),
        ]

        # 纯位置控制，不叠加速度或加速度前馈。
        msg.velocity = [nan, nan, nan]
        msg.acceleration = [nan, nan, nan]

        # 不强制改变机头朝向。
        msg.yaw = nan
        msg.yawspeed = nan
        msg.timestamp = self.now_us()

        self.setpoint_pub.publish(msg)

    def publish_command(self, command, param1=0.0, param2=0.0):
        """发送 PX4 模式切换、解锁、降落等命令。"""
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
        self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info('Requested AUTO LAND')

    def set_goal(self, x, y, z):
        """切换当前目标点。"""
        self.goal_x = float(x)
        self.goal_y = float(y)
        self.goal_z = float(z)

    def distance_to_goal(self):
        """返回当前位置到当前目标点的三维距离。"""
        dx = self.position.x - self.goal_x
        dy = self.position.y - self.goal_y
        dz = self.position.z - self.goal_z

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def relative_xy(self, x, y):
        """把绝对 PX4 坐标转成相对起飞点坐标。"""
        return x - self.home_x, y - self.home_y

    def is_inside_inflated_obstacle(self, rel_x, rel_y):
        """
        判断一个相对起飞点坐标是否落在
        “障碍物 + 安全边界”的危险区域中。
        """
        safe_x_min = self.obstacle_x_min - self.safety_margin
        safe_x_max = self.obstacle_x_max + self.safety_margin
        safe_y_min = self.obstacle_y_min - self.safety_margin
        safe_y_max = self.obstacle_y_max + self.safety_margin

        return (
            safe_x_min <= rel_x <= safe_x_max
            and safe_y_min <= rel_y <= safe_y_max
        )

    def segment_is_safe(self, start_rel_x, start_rel_y, end_rel_x, end_rel_y):
        """
        对一段航线做简单采样检查。
        只要采样点落入“膨胀后的障碍区”，就判定不安全。
        """
        sample_count = 50

        for index in range(sample_count + 1):
            ratio = index / sample_count

            x = start_rel_x + (end_rel_x - start_rel_x) * ratio
            y = start_rel_y + (end_rel_y - start_rel_y) * ratio

            if self.is_inside_inflated_obstacle(x, y):
                return False

        return True

    def log_map(self):
        """打印障碍物本体与膨胀安全边界。"""
        safe_x_min = self.obstacle_x_min - self.safety_margin
        safe_x_max = self.obstacle_x_max + self.safety_margin
        safe_y_min = self.obstacle_y_min - self.safety_margin
        safe_y_max = self.obstacle_y_max + self.safety_margin

        self.get_logger().info(
            'Known obstacle barrier relative to home: '
            f'X=[{self.obstacle_x_min:.2f}, {self.obstacle_x_max:.2f}], '
            f'Y=[{self.obstacle_y_min:.2f}, {self.obstacle_y_max:.2f}]'
        )

        self.get_logger().info(
            'Inflated no-fly boundary with safety margin '
            f'{self.safety_margin:.2f} m: '
            f'X=[{safe_x_min:.2f}, {safe_x_max:.2f}], '
            f'Y=[{safe_y_min:.2f}, {safe_y_max:.2f}]'
        )

    def create_route_template(self):
        """
        设计一个位于障碍墙前方的矩形巡逻路线。

        最大前向 x 为 +1.10 m。
        膨胀后的障碍区从约 +1.43 m 才开始，
        因此路线整体保持在安全侧。
        """
        return [
            ('P1 lower-left', 0.35, -1.20),
            ('P2 lower-right', 1.10, -1.20),
            ('P3 upper-right', 1.10, 1.55),
            ('P4 upper-left', 0.35, 1.55),
        ]

    def build_route(self):
        """将相对航点扩展为实际可飞行的绝对 PX4 坐标路线。"""
        self.route = []

        for round_index in range(1, self.patrol_rounds + 1):
            for name, rel_x, rel_y in self.route_template:
                absolute_x = self.home_x + rel_x
                absolute_y = self.home_y + rel_y
                absolute_z = self.home_z - self.takeoff_height

                self.route.append(
                    {
                        'name': name,
                        'round': round_index,
                        'rel_x': rel_x,
                        'rel_y': rel_y,
                        'x': absolute_x,
                        'y': absolute_y,
                        'z': absolute_z,
                    }
                )

        # 最后返回起飞点正上方。
        self.route.append(
            {
                'name': 'RETURN_HOME',
                'round': self.patrol_rounds,
                'rel_x': 0.0,
                'rel_y': 0.0,
                'x': self.home_x,
                'y': self.home_y,
                'z': self.home_z - self.takeoff_height,
            }
        )

    def validate_route(self):
        """
        检查：
        1. 所有航点是否在危险区外；
        2. 相邻航点之间的航段是否穿过危险区。
        """
        previous_rel_x = 0.0
        previous_rel_y = 0.0

        for waypoint in self.route:
            rel_x = waypoint['rel_x']
            rel_y = waypoint['rel_y']

            if self.is_inside_inflated_obstacle(rel_x, rel_y):
                self.get_logger().error(
                    f"Unsafe waypoint: {waypoint['name']} "
                    f"relative=({rel_x:.2f}, {rel_y:.2f})"
                )
                return False

            if not self.segment_is_safe(
                previous_rel_x,
                previous_rel_y,
                rel_x,
                rel_y,
            ):
                self.get_logger().error(
                    f"Unsafe segment detected before "
                    f"{waypoint['name']}."
                )
                return False

            previous_rel_x = rel_x
            previous_rel_y = rel_y

        return True

    def print_route(self):
        """在真正起飞前打印路线。"""
        self.get_logger().info(
            f'Patrol route created: {self.patrol_rounds} rounds.'
        )

        for index, waypoint in enumerate(self.route, start=1):
            self.get_logger().info(
                f"  {index:02d}. "
                f"Round {waypoint['round']} | "
                f"{waypoint['name']} | "
                f"relative=({waypoint['rel_x']:.2f}, "
                f"{waypoint['rel_y']:.2f}) | "
                f"target=({waypoint['x']:.2f}, "
                f"{waypoint['y']:.2f}, "
                f"{waypoint['z']:.2f})"
            )

    def set_current_waypoint(self):
        """把当前 route_index 对应的路线点设为目标。"""
        waypoint = self.route[self.route_index]

        self.set_goal(
            waypoint['x'],
            waypoint['y'],
            waypoint['z'],
        )

        self.get_logger().info(
            f"Flying to route point "
            f"{self.route_index + 1}/{len(self.route)} | "
            f"Round {waypoint['round']} | "
            f"{waypoint['name']} | "
            f"relative=({waypoint['rel_x']:.2f}, "
            f"{waypoint['rel_y']:.2f})"
        )

    def start_patrol(self):
        """起飞完成后开始巡逻。"""
        self.route_index = 0
        self.current_round = 1
        self.mission_state = 'FLY_TO_WAYPOINT'

        self.get_logger().info(
            'Takeoff completed. Starting obstacle-aware patrol.'
        )

        self.set_current_waypoint()

    def begin_waypoint_hold(self):
        """当前航点到达后开始停留。"""
        waypoint = self.route[self.route_index]

        self.mission_state = 'HOLD_WAYPOINT'
        self.hold_start_counter = self.counter

        self.get_logger().info(
            f"Reached {waypoint['name']}. "
            f"Holding for {self.waypoint_hold_seconds:.1f} seconds."
        )

    def advance_waypoint(self):
        """航点停留结束，进入下一个航点或返回后的最后悬停。"""
        self.route_index += 1

        if self.route_index >= len(self.route):
            self.mission_state = 'FINAL_HOLD'
            self.hold_start_counter = self.counter

            self.get_logger().info(
                'Route completed. Holding above home for 3.0 seconds '
                'before landing.'
            )
            return

        self.mission_state = 'FLY_TO_WAYPOINT'
        self.set_current_waypoint()

    def log_progress(self):
        """每秒打印一次状态、当前位置、目标点、距离和相对位置。"""
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

        distance = self.distance_to_goal()

        rel_x, rel_y = self.relative_xy(
            self.position.x,
            self.position.y,
        )

        extra_info = ''

        if self.mission_state in ('FLY_TO_WAYPOINT', 'HOLD_WAYPOINT'):
            waypoint = self.route[self.route_index]

            extra_info = (
                f" | Target: {waypoint['name']} "
                f"| Round {waypoint['round']}/{self.patrol_rounds}"
            )

        elif self.mission_state == 'FINAL_HOLD':
            extra_info = ' | Final hold above home'

        self.get_logger().info(
            f'[{self.mission_state}] '
            f'Current: x={self.position.x:.2f}, '
            f'y={self.position.y:.2f}, '
            f'z={self.position.z:.2f} | '
            f'Relative: x={rel_x:.2f}, y={rel_y:.2f} | '
            f'Goal: x={self.goal_x:.2f}, '
            f'y={self.goal_y:.2f}, '
            f'z={self.goal_z:.2f} | '
            f'Distance: {distance:.2f} m'
            f'{extra_info}'
        )

    def timer_callback(self):
        """
        任务状态：

        WAIT_POSITION
        → TAKEOFF
        → FLY_TO_WAYPOINT
        → HOLD_WAYPOINT
        → FINAL_HOLD
        → LANDING
        → MISSION_FINISHED
        """

        # 在降落与任务完成后，不再维持 Offboard 位置控制。
        if self.mission_state not in ('LANDING', 'MISSION_FINISHED'):
            self.publish_heartbeat()

        # 初次取得位置：记录起飞点、建立路线。
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

            self.route_template = self.create_route_template()
            self.build_route()

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

            self.log_map()
            self.print_route()

            if not self.validate_route():
                self.mission_state = 'ABORTED'

                self.get_logger().error(
                    'Route validation failed. '
                    'Mission aborted before arming.'
                )
                return

            self.mission_state = 'TAKEOFF'

        # 起飞和巡逻阶段持续给 PX4 发布位置目标。
        if self.mission_state in (
            'TAKEOFF',
            'FLY_TO_WAYPOINT',
            'HOLD_WAYPOINT',
            'FINAL_HOLD',
        ):
            self.publish_setpoint()

        # 每秒打印一次任务进度。
        if self.counter - self.last_progress_log_counter >= 10:
            self.log_progress()
            self.last_progress_log_counter = self.counter

        # 先预发送一秒 heartbeat / setpoint，再请求 Offboard。
        if self.counter < 10:
            self.counter += 1
            return

        # 若路线校验失败，不解锁也不飞。
        if self.mission_state == 'ABORTED':
            self.counter += 1
            return

        # 等待自动降落完成。
        if self.mission_state == 'LANDING':
            if self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.mission_state = 'MISSION_FINISHED'

                self.get_logger().info(
                    'Vehicle disarmed. Obstacle-aware patrol finished.'
                )

            self.counter += 1
            return

        if self.mission_state == 'MISSION_FINISHED':
            self.counter += 1
            return

        # 每秒检查一次飞控状态和任务条件。
        if self.counter % 10 == 0:
            if self.status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.request_offboard()

            elif self.status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
                self.request_arm()

            elif self.mission_state == 'TAKEOFF':
                if self.distance_to_goal() < self.reached_threshold:
                    self.start_patrol()

            elif self.mission_state == 'FLY_TO_WAYPOINT':
                if self.distance_to_goal() < self.reached_threshold:
                    self.begin_waypoint_hold()

            elif self.mission_state == 'HOLD_WAYPOINT':
                hold_counter = int(self.waypoint_hold_seconds * 10)

                if self.counter - self.hold_start_counter >= hold_counter:
                    self.advance_waypoint()

            elif self.mission_state == 'FINAL_HOLD':
                final_hold_counter = 30

                if (
                    self.counter - self.hold_start_counter
                    >= final_hold_counter
                    and not self.land_requested
                ):
                    self.request_land()
                    self.land_requested = True
                    self.mission_state = 'LANDING'

        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAwarePatrol()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        # Ctrl+C 时不再额外发送命令，避免 ROS context 已关闭后的报错。
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()