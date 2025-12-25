from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from typing import Optional, Tuple
import time
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray

try:
    from gazebo_msgs.srv import SetModelState
    from gazebo_msgs.msg import ModelState
    _HAS_MODEL_STATE = True
except Exception:
    _HAS_MODEL_STATE = False

_ROS_INITIALIZED = False

def ensure_ros_init(node_name: str = "gym_mobile_robot_env") -> None:
    global _ROS_INITIALIZED
    if _ROS_INITIALIZED or rospy.core.is_initialized():
        return
    rospy.init_node(node_name, anonymous=True, disable_signals=True)
    _ROS_INITIALIZED = True

def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi

def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)

class ROSGazeboMobileRobotEnv(gym.Env):
    metadata = {"render_modes": []}
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_FORWARD = 2

    def __init__(
        self,
        *,
        scan_topic: str = "/scan",
        odom_topic: str = "/odom",
        cmd_vel_topic: str = "/cmd_vel",
        reset_world_service: str = "/gazebo/reset_world",
        reset_sim_service: str = "/gazebo/reset_simulation",
        set_model_state_service: str = "/gazebo/set_model_state",
        robot_model_name: str | None = None,
        init_x: float = 0.0,
        init_y: float = 0.0,
        init_yaw: float = 0.0,
        max_steps: int = 250,
        max_lidar_range: float = 3.5,
        forward_v: float = 0.15,
        turn_v: float = 0.15,
        turn_omega: float = 1.5,
        publish_hz: float = 30.0,# 控制频率
        #动作持续时间 = 1/publish_hz * n 个周期
        action_duration: float = 0.1 ,# 动作持续的控制周期数
        RTH: float = 0.20,
        CTH: float = 0.15,
        r_reach: float = 200.0,
        r_load: float = 50.0,
        r_out: float = -50.0,
        r_collision: float = -150.0,
        p_r: float = 10,
        r_o: float = -0.1,
        goal_x: float | None = None,
        goal_y: float | None = None,
        waypoints: list[tuple[float, float]] | None = None,
        waypoint_rth: float = 0.20,
        random_goal: bool = False,
        max_goal_distance: float = 8.0,
        wait_timeout: float = 1.0,
        obstacle_mode: bool = False,
        enable_viz: bool = True,
        viz_frame: str = "odom",
        max_path_len: int = 3000,
        render_mode: str | None = None,   # <<< 新增这一行（放最后也行）
    ):
        super().__init__()
        ensure_ros_init()

        self.scan_topic = scan_topic
        self.odom_topic = odom_topic
        self.cmd_vel_topic = cmd_vel_topic
        self.reset_world_service = reset_world_service
        self.reset_sim_service = reset_sim_service
        self.set_model_state_service = set_model_state_service
        self.robot_model_name = robot_model_name
        
        # 参数保存
        self.init_x = init_x
        self.init_y = init_y
        self.init_yaw = init_yaw
        self.max_steps = max_steps
        self.max_lidar_range = max_lidar_range
        self.forward_v = forward_v
        self.turn_v = turn_v
        self.turn_omega = turn_omega
        self.publish_hz = publish_hz
        self.action_duration = action_duration
        self.RTH = RTH
        self.CTH = CTH
        self.r_reach = r_reach
        self.r_load = r_load
        self.r_out = r_out
        self.r_collision = r_collision
        self.p_r = p_r
        self.r_o = r_o
        self.random_goal = random_goal
        self.waypoints = waypoints
        self.waypoint_rth = waypoint_rth
        self.wp_idx = 0
        self.max_goal_distance = max_goal_distance
        self.wait_timeout = wait_timeout
        self.obstacle_mode = obstacle_mode
        self.enable_viz = enable_viz
        self.viz_frame = viz_frame
        self.max_path_len = max_path_len

        if goal_x is None:
            goal_x = rospy.get_param("~goal_x", 1.0)
        if goal_y is None:
            goal_y = rospy.get_param("~goal_y", 0.0)
        self.goal_x = float(goal_x)
        self.goal_y = float(goal_y)

        # ---------------------------------------------------------------------
        # 定义空间
        # ---------------------------------------------------------------------
        self.action_space = spaces.Discrete(3)
        obs_low = np.array([0.0] * 90 + [-1.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([1.0] * 90 + [1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ---------------------------------------------------------------------
        # [修改] ROS 通讯：后台订阅模式
        # ---------------------------------------------------------------------
        self._cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        
        # 1. 定义缓存变量
        self._current_scan: Optional[LaserScan] = None
        self._current_odom: Optional[Odometry] = None

        # 2. 建立长连接订阅 (Subscriber)
        # 这会让数据在后台自动更新，不需要我们在 step 里去 "wait"
        self._scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self._scan_cb, queue_size=1)
        self._odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=1)

        if self.enable_viz:
            self._waypoints_pub = rospy.Publisher("/kfdqn_viz/waypoints", MarkerArray, queue_size=1, latch=True)
            self._current_wp_pub = rospy.Publisher("/kfdqn_viz/current_wp", Marker, queue_size=1)
            self._trajectory_pub = rospy.Publisher("/kfdqn_viz/trajectory", Path, queue_size=1)

        self._np_random = np.random.default_rng()
        self.prev_action = 0.0 
        self.prev_dis = 0.0
        self.step_count = 0
        self.goal = np.array([self.goal_x, self.goal_y], dtype=np.float32)
        self.path_msg: Optional[Path] = None

    # -------------------------------------------------------------------------
    # [新增] 回调函数：只做最简单的数据缓存
    # -------------------------------------------------------------------------
    def _scan_cb(self, msg: LaserScan):
        self._current_scan = msg

    def _odom_cb(self, msg: Odometry):
        self._current_odom = msg

    # -------------------------------------------------------------------------
    # 辅助函数
    # -------------------------------------------------------------------------
    def _get_pose(self, odom: Odometry) -> Tuple[float, float, float]:
        pos = odom.pose.pose.position
        ori = odom.pose.pose.orientation
        yaw = _yaw_from_quat(ori.x, ori.y, ori.z, ori.w)
        return float(pos.x), float(pos.y), float(yaw)

    def current_goal(self) -> Tuple[float, float]:
        if self.waypoints is None:
            return float(self.goal_x), float(self.goal_y)
        return self.waypoints[self.wp_idx]

    def _lidar90(self, scan: LaserScan) -> np.ndarray:
        if scan.range_max > 0.0:
            self.max_lidar_range = float(scan.range_max)

        ranges = np.array(list(scan.ranges), dtype=np.float32)
        if ranges.size == 0:
            return np.full(90, self.max_lidar_range, dtype=np.float32)
        
        n = ranges.size
        bins = np.full(90, self.max_lidar_range, dtype=np.float32)
        for i in range(90):
            start = int(i * n / 90)
            end = int((i + 1) * n / 90)
            if end <= start:
                segment = ranges[min(start, n - 1) : min(start, n - 1) + 1]
            else:
                segment = ranges[start:end]
            finite = segment[np.isfinite(segment)]
            if finite.size == 0:
                bins[i] = self.max_lidar_range
            else:
                bins[i] = float(np.min(finite))
        
        bins = np.clip(bins, 0.0, self.max_lidar_range)
        return bins.astype(np.float32)

    def _normalize_obs(self, lidar: np.ndarray, theta_d: float, dis: float, prev_action: float) -> np.ndarray:
        lidar_norm = lidar / self.max_lidar_range
        theta_norm = theta_d / math.pi
        dis_norm = np.clip(dis / self.max_goal_distance, 0.0, 1.0)
        return np.concatenate([lidar_norm, np.array([theta_norm, dis_norm, prev_action], dtype=np.float32)], axis=0)

    # ... [可视化相关的函数保持不变] ...
    def _publish_waypoints_markers(self) -> None:
        if not self.enable_viz: return
        marker_array = MarkerArray()
        points = self.waypoints if self.waypoints else [(self.goal_x, self.goal_y)]
        for idx, (gx, gy) in enumerate(points):
            marker = Marker()
            marker.header.frame_id = self.viz_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "waypoints"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = gx
            marker.pose.position.y = gy
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.12
            marker.scale.y = 0.12
            marker.scale.z = 0.12
            marker.color.b = 1.0
            marker.color.a = 0.6
            marker_array.markers.append(marker)
        self._waypoints_pub.publish(marker_array)

    def _publish_current_wp_marker(self) -> None:
        if not self.enable_viz: return
        gx, gy = self.current_goal()
        marker = Marker()
        marker.header.frame_id = self.viz_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "current_wp"
        marker.id = 999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = gx
        marker.pose.position.y = gy
        marker.pose.position.z = 0.06
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18
        marker.color.r = 1.0
        marker.color.a = 1.0
        self._current_wp_pub.publish(marker)

    def _call_reset(self) -> None:
        # 1. 优先尝试 reset_world (不回退时间，更稳)
        try:
            rospy.wait_for_service(self.reset_world_service, timeout=self.wait_timeout)
            rospy.ServiceProxy(self.reset_world_service, Empty)()
            return 
        except Exception: 
            pass

        # 2. 只有当 reset_world 失败时，才尝试 reset_sim (回退时间)
        try:
            rospy.wait_for_service(self.reset_sim_service, timeout=self.wait_timeout)
            rospy.ServiceProxy(self.reset_sim_service, Empty)()
        except Exception: 
            pass

    def _call_set_model_state(self) -> None:
        if not self.robot_model_name or not _HAS_MODEL_STATE: return
        try:
            rospy.wait_for_service(self.set_model_state_service, timeout=self.wait_timeout)
            srv = rospy.ServiceProxy(self.set_model_state_service, SetModelState)
            state = ModelState()
            state.model_name = self.robot_model_name
            state.pose.position.x = self.init_x
            state.pose.position.y = self.init_y
            state.pose.position.z = 0.03 
            state.pose.orientation.z = math.sin(self.init_yaw / 2.0)
            state.pose.orientation.w = math.cos(self.init_yaw / 2.0)
            
            # [新增] 强制清零速度
            state.twist.linear.x = 0.0
            state.twist.linear.y = 0.0
            state.twist.linear.z = 0.0
            state.twist.angular.x = 0.0
            state.twist.angular.y = 0.0
            state.twist.angular.z = 0.0
            
            srv(state)
        except Exception: pass

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self._cmd_pub.publish(cmd)

    # -------------------------------------------------------------------------
    # Gym 接口: Reset
    # -------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.enable_viz:
            self.path_msg = Path()
            self.path_msg.header.frame_id = self.viz_frame
            self._trajectory_pub.publish(self.path_msg)  # 立刻清空显示
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        odom_seq0 = self._current_odom.header.seq if self._current_odom else -1
        scan_seq0 = self._current_scan.header.seq if self._current_scan else -1

        # 在重置物理前，先清空脏数据，防止读到上一回合的残留
        self._current_scan = None
        self._current_odom = None

        self._call_reset()
        self._call_set_model_state()
        self._publish_cmd(0.0, 0.0)
        rospy.sleep(0.15) 

        # Reset 时，必须确保有第一帧数据
        # 只有这里需要阻塞等待，因为刚 reset 完不知道订阅回调是否触发了
        start_wait = time.time()
        while True:
            scan = self._current_scan
            odom = self._current_odom
            if scan and odom:
                if scan.header.seq > scan_seq0 and odom.header.seq > odom_seq0:
                    x, y, yaw = self._get_pose(odom)
                    if (x - self.init_x) ** 2 + (y - self.init_y) ** 2 < (0.10 ** 2):  # 10cm 容差
                        break
            if time.time() - start_wait > 5.0:
                raise RuntimeError("Reset timeout: No fresh Scan/Odom near init pose.")
            time.sleep(0.005)

        self.step_count = 0
        self.prev_action = 0.0
        self.wp_idx = 0

        # 随机目标逻辑
        if self.waypoints is None and self.random_goal:
            angle = float(self._np_random.uniform(-math.pi, math.pi))
            radius = float(self._np_random.uniform(0.5, self.max_goal_distance))
            self.goal_x = float(radius * math.cos(angle))
            self.goal_y = float(radius * math.sin(angle))
            self.goal = np.array([self.goal_x, self.goal_y], dtype=np.float32)
        else:
            gx, gy = self.current_goal()
            self.goal = np.array([gx, gy], dtype=np.float32)
        
        if self.enable_viz:
            self._publish_waypoints_markers()
            self._publish_current_wp_marker()
            self.path_msg = Path()
            self.path_msg.header.frame_id = self.viz_frame

        # 直接从缓存变量读取
        lidar = self._lidar90(self._current_scan)
        x, y, yaw = self._get_pose(self._current_odom)
        gx, gy = self.current_goal()
        dx = gx - x
        dy = gy - y
        dis = float(math.hypot(dx, dy))
        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)
        self.prev_dis = dis

        obs = self._normalize_obs(lidar, theta_d, dis, self.prev_action)
        info = {"min_lidar": float(np.min(lidar)), "theta_d": float(theta_d), "dis": float(dis)}
        return obs.astype(np.float32), info

    # -------------------------------------------------------------------------
    # Gym 接口: Step (移除 wait_for_message)
    # -------------------------------------------------------------------------
    def step(self, action: int):
        assert self.action_space.contains(action)
        self.step_count += 1

        # 1. 动作解码
        if action == self.ACTION_LEFT:
            lin, ang = self.turn_v, self.turn_omega
        elif action == self.ACTION_RIGHT:
            lin, ang = self.turn_v, -self.turn_omega
        else:
            lin, ang = self.forward_v, 0.0

        # 记录当前的 seq，用于确保读到新数据
        scan_seq0 = self._current_scan.header.seq if self._current_scan else -1

        # 2. 持续发布动作 (保持 action_duration 的时间)
        # 在这段时间内，机器人会一直执行这个速度
        start_time = rospy.get_time()
        end_time = start_time + self.action_duration
        rate = rospy.Rate(self.publish_hz)

        while rospy.get_time() < end_time:
            if rospy.is_shutdown(): break
            try:
                self._publish_cmd(lin, ang)
                rate.sleep()
            except rospy.ROSTimeMovedBackwardsException:
                break
        
        # 动作结束了不刹车。
        # 机器人会带着当前的速度惯性，直接进入下一次 step 的决策。
        # 如果下一次决策够快，速度会无缝衔接。
        # 3. 确保读到的是“动作执行期间/之后”产生的最新数据
        # 即使不停车，我们也需要保证数据不是动作开始前的老黄历
        # 3. 确保读到的是新数据（使用 sim time）
        t0 = rospy.get_time()
        timeout = 0.7  # 这里的 0.1 变成“仿真秒”，更合理
        rate_wait = rospy.Rate(120)  # 等待时的轮询频率（仿真Hz）

        while True:
            scan = self._current_scan
            odom = self._current_odom
            if scan and odom and scan.header.seq > scan_seq0:
                break
            if rospy.get_time() - t0 > timeout:
                break
            rate_wait.sleep()
            
        # 简单防护：万一中途 reset 导致数据丢失
        if scan is None or odom is None:
             raise RuntimeError("Data loss during step (Subscribers are empty).")

        if self.enable_viz:
            if self.path_msg is None:
                self.path_msg = Path()
                self.path_msg.header.frame_id = self.viz_frame
            pose = PoseStamped()
            pose.header.frame_id = self.viz_frame
            pose.header.stamp = rospy.Time.now()
            pose.pose = odom.pose.pose
            self.path_msg.poses.append(pose)
            if len(self.path_msg.poses) > self.max_path_len:
                self.path_msg.poses = self.path_msg.poses[-self.max_path_len :]
            self._trajectory_pub.publish(self.path_msg)

        lidar = self._lidar90(scan)
        min_lidar = float(np.min(lidar))
        x, y, yaw = self._get_pose(odom)
        gx, gy = self.current_goal()
        dx = gx - x
        dy = gy - y
        dis = float(math.hypot(dx, dy))
        theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)

        terminated = False
        truncated = False
        info = {"min_lidar": float(min_lidar), "theta_d": float(theta_d), "dis": float(dis)}

        if min_lidar < self.CTH:
            reward = self.r_collision
            terminated = True
            info["is_collision"] = True
        else:
            reward = (self.prev_dis - dis) * self.p_r + self.r_o
            
            if dis < self.waypoint_rth:
                if self.waypoints is not None and self.wp_idx < len(self.waypoints) - 1:
                    self.wp_idx += 1
                    new_gx, new_gy = self.current_goal()
                    dx = new_gx - x
                    dy = new_gy - y
                    dis = float(math.hypot(dx, dy))
                    theta_d = _wrap_angle(math.atan2(dy, dx) - yaw)
                    self.prev_dis = dis 
                    
                    reward += self.r_load
                    info["waypoint_reached"] = True
                    info["wp_idx"] = self.wp_idx
                    if self.enable_viz:
                        self._publish_current_wp_marker()
                else:
                    reward = self.r_reach
                    terminated = True
                    info["is_success"] = True

        if self.step_count >= self.max_steps:
            truncated = True
            reward += self.r_out 

        self.prev_action = float(action) / 2.0
        if not terminated:
            self.prev_dis = dis
        
        if terminated or truncated:
            self._publish_cmd(0.0, 0.0)

        obs = self._normalize_obs(lidar, theta_d, dis, self.prev_action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self) -> None:
        self._publish_cmd(0.0, 0.0)

# 注册环境 (保持不变)
if "GoalReachROS-v0" not in registry:
    register(
        id="GoalReachROS-v0",
        entry_point="envs_ros.ros_gazebo_mobile_robot_env:ROSGazeboMobileRobotEnv",
        kwargs={
            "obstacle_mode": False,
            "robot_model_name": "turtlebot3_burger",
            "waypoints": [(0.7, 0.5), (1.6, 0.5), (1.3, -0.4), (0.8, 0.0), (0.8, -0.5)],
            "waypoint_rth": 0.20,
        },
    )

if "ObstacleAvoidROS-v0" not in registry:
    register(
        id="ObstacleAvoidROS-v0",
        entry_point="envs_ros.ros_gazebo_mobile_robot_env:ROSGazeboMobileRobotEnv",
        kwargs={
            "obstacle_mode": True,
            "robot_model_name": "turtlebot3_burger",
            "waypoints": [(1.4, 0.9), (-0.15, 0.1)],
            "waypoint_rth": 0.20,
        },
    )