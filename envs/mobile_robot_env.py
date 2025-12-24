from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class MobileRobotEnv(gym.Env):
    """2D mobile robot with LiDAR, goal reach, and optional obstacles."""

    metadata = {"render_modes": []}

    # Action encoding: 0=left, 1=right, 2=forward
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_FORWARD = 2

    def __init__(
        self,
        *,
        obstacle_mode: bool = False,
        arena_x: Tuple[float, float] = (-2.0, 2.0),
        arena_y: Tuple[float, float] = (-2.0, 2.0),
        max_episode_steps: int = 200,
        max_lidar_range: float = 3.0,
        lidar_num: int = 90,
        lidar_fov_rad: float = 2 * math.pi,
        forward_step: float = 0.08,
        turn_delta_rad: float = 0.261799,  # 15 deg
        robot_radius: float = 0.08,
        RTH: float = 0.20,
        CTH: float = 0.15,
        r_reach: float = 100.0,
        r_collision: float = -100.0,
        p_r: float = 5.0,
        r_o: float = -0.01,
        obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
    ):
        super().__init__()
        self.obstacle_mode = obstacle_mode
        self.arena_x = arena_x
        self.arena_y = arena_y
        self.max_episode_steps = max_episode_steps
        self.max_lidar_range = max_lidar_range
        if lidar_num != 90:
            raise ValueError("lidar_num must be 90 to keep observation shape (93,).")
        self.lidar_num = lidar_num
        self.lidar_fov_rad = lidar_fov_rad
        self.forward_step = forward_step
        self.turn_delta_rad = turn_delta_rad
        self.robot_radius = robot_radius
        self.RTH = RTH
        self.CTH = CTH
        self.r_reach = r_reach
        self.r_collision = r_collision
        self.p_r = p_r
        self.r_o = r_o

        if obstacles is not None:
            self.obstacles = list(obstacles)
        elif obstacle_mode:
            self.obstacles = [
                (0.0, 0.0, 0.30),
                (0.8, 0.6, 0.20),
                (-0.9, -0.6, 0.20),
            ]
        else:
            self.obstacles = []

        self.action_space = spaces.Discrete(3)
        max_diag = math.hypot(self.arena_x[1] - self.arena_x[0], self.arena_y[1] - self.arena_y[0])
        obs_low = np.array([0.0] * self.lidar_num + [-math.pi, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([self.max_lidar_range] * self.lidar_num + [math.pi, max_diag, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._lidar_angles = np.linspace(
            -self.lidar_fov_rad / 2.0,
            self.lidar_fov_rad / 2.0,
            self.lidar_num,
            endpoint=False,
        )

        self.np_random: Optional[np.random.Generator] = None
        self.position = np.zeros(2, dtype=np.float32)
        self.yaw = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.prev_action = 0.0  # reset => 0.0; step => action_id / 2.0
        self.prev_dis = 0.0
        self.step_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None or self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.step_count = 0
        self.prev_action = 0.0
        self.yaw = float(self.np_random.uniform(-math.pi, math.pi))

        self.position = self._sample_position(min_dist=self.robot_radius + 0.05)
        self.goal = self._sample_goal(self.position)

        dis = self._distance_to_goal(self.position)
        self.prev_dis = dis
        lidar = self._compute_lidar(self.position, self.yaw)
        theta_d = _wrap_angle(math.atan2(self.goal[1] - self.position[1], self.goal[0] - self.position[0]) - self.yaw)
        obs = self._build_obs(lidar, theta_d, dis, self.prev_action)
        info = {
            "min_lidar": float(np.min(lidar)),
            "theta_d": float(theta_d),
            "dis": float(dis),
        }
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.step_count += 1

        prev_dis = self.prev_dis
        clipped = False

        if action == self.ACTION_LEFT:
            self.yaw = _wrap_angle(self.yaw + self.turn_delta_rad)
        elif action == self.ACTION_RIGHT:
            self.yaw = _wrap_angle(self.yaw - self.turn_delta_rad)
        elif action == self.ACTION_FORWARD:
            move = np.array([math.cos(self.yaw), math.sin(self.yaw)], dtype=np.float32) * self.forward_step
            new_pos = self.position + move
            min_x, max_x = self.arena_x
            min_y, max_y = self.arena_y
            clipped_x = float(np.clip(new_pos[0], min_x + self.robot_radius, max_x - self.robot_radius))
            clipped_y = float(np.clip(new_pos[1], min_y + self.robot_radius, max_y - self.robot_radius))
            clipped = (clipped_x != new_pos[0]) or (clipped_y != new_pos[1])
            self.position = np.array([clipped_x, clipped_y], dtype=np.float32)

        lidar = self._compute_lidar(self.position, self.yaw)
        min_lidar = float(np.min(lidar))
        if clipped:
            min_lidar = min(min_lidar, 0.0)

        dis = self._distance_to_goal(self.position)
        theta_d = _wrap_angle(math.atan2(self.goal[1] - self.position[1], self.goal[0] - self.position[0]) - self.yaw)

        terminated = False
        truncated = False
        info = {
            "min_lidar": float(min_lidar),
            "theta_d": float(theta_d),
            "dis": float(dis),
        }

        if dis < self.RTH:
            reward = self.r_reach
            terminated = True
            info["is_success"] = True
        elif min_lidar < self.CTH:
            reward = self.r_collision
            terminated = True
            info["is_collision"] = True
        else:
            reward = (prev_dis - dis) * self.p_r + self.r_o

        if self.step_count >= self.max_episode_steps:
            truncated = True

        self.prev_action = float(action) / 2.0
        self.prev_dis = dis

        obs = self._build_obs(lidar, theta_d, dis, self.prev_action)
        return obs, float(reward), terminated, truncated, info

    def _build_obs(self, lidar: np.ndarray, theta_d: float, dis: float, prev_action: float) -> np.ndarray:
        obs = np.concatenate(
            [
                lidar.astype(np.float32),
                np.array([theta_d, dis, prev_action], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    def _distance_to_goal(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(self.goal - pos))

    def _sample_position(self, min_dist: float) -> np.ndarray:
        min_x, max_x = self.arena_x
        min_y, max_y = self.arena_y
        for _ in range(500):
            x = float(self.np_random.uniform(min_x + min_dist, max_x - min_dist))
            y = float(self.np_random.uniform(min_y + min_dist, max_y - min_dist))
            if self._is_position_safe((x, y), min_dist):
                return np.array([x, y], dtype=np.float32)
        raise RuntimeError("Failed to sample a valid position.")

    def _sample_goal(self, robot_pos: np.ndarray) -> np.ndarray:
        min_dist = self.robot_radius + 0.05
        for _ in range(500):
            pos = self._sample_position(min_dist)
            if np.linalg.norm(pos - robot_pos) > max(self.RTH * 2.0, 0.5):
                return pos
        raise RuntimeError("Failed to sample a valid goal.")

    def _is_position_safe(self, pos: Tuple[float, float], margin: float) -> bool:
        x, y = pos
        min_x, max_x = self.arena_x
        min_y, max_y = self.arena_y
        if x <= min_x + margin or x >= max_x - margin:
            return False
        if y <= min_y + margin or y >= max_y - margin:
            return False
        for ox, oy, r in self.obstacles:
            if math.hypot(x - ox, y - oy) <= r + margin:
                return False
        return True

    def _compute_lidar(self, pos: np.ndarray, yaw: float) -> np.ndarray:
        lidar = np.full(self.lidar_num, self.max_lidar_range, dtype=np.float32)
        for i, rel in enumerate(self._lidar_angles):
            angle = yaw + rel
            dx = math.cos(angle)
            dy = math.sin(angle)
            dist_wall = self._ray_intersect_walls(pos, (dx, dy))
            dist = dist_wall if dist_wall is not None else self.max_lidar_range
            for ox, oy, r in self.obstacles:
                dist_obs = self._ray_circle_intersect(pos, (dx, dy), (ox, oy), r)
                if dist_obs is not None:
                    dist = min(dist, dist_obs)
            lidar[i] = min(dist, self.max_lidar_range)
        return lidar

    def _ray_intersect_walls(self, origin: np.ndarray, direction: Tuple[float, float]) -> Optional[float]:
        x0, y0 = float(origin[0]), float(origin[1])
        dx, dy = direction
        min_x, max_x = self.arena_x
        min_y, max_y = self.arena_y
        t_candidates: List[float] = []

        if abs(dx) > 1e-9:
            for x_wall in (min_x, max_x):
                t = (x_wall - x0) / dx
                if t >= 0:
                    y = y0 + t * dy
                    if min_y <= y <= max_y:
                        t_candidates.append(t)
        if abs(dy) > 1e-9:
            for y_wall in (min_y, max_y):
                t = (y_wall - y0) / dy
                if t >= 0:
                    x = x0 + t * dx
                    if min_x <= x <= max_x:
                        t_candidates.append(t)

        if not t_candidates:
            return None
        return min(t_candidates)

    def _ray_circle_intersect(
        self,
        origin: np.ndarray,
        direction: Tuple[float, float],
        center: Tuple[float, float],
        radius: float,
    ) -> Optional[float]:
        x0, y0 = float(origin[0]), float(origin[1])
        dx, dy = direction
        cx, cy = center
        fx = x0 - cx
        fy = y0 - cy

        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - radius * radius
        disc = b * b - c
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc
        if t1 >= 0:
            return t1
        if t2 >= 0:
            return t2
        return None


if "GoalReach-v0" not in registry:
    register(
        id="GoalReach-v0",
        entry_point="envs.mobile_robot_env:MobileRobotEnv",
        kwargs={"obstacle_mode": False},
    )

if "ObstacleAvoid-v0" not in registry:
    register(
        id="ObstacleAvoid-v0",
        entry_point="envs.mobile_robot_env:MobileRobotEnv",
        kwargs={"obstacle_mode": True},
    )
