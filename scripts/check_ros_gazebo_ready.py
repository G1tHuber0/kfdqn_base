#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import gymnasium as gym
import rosgraph
import rospy

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import envs_ros  # noqa: F401

REQUIRED_TOPICS = ["/scan", "/odom", "/cmd_vel"]
REQUIRED_SERVICES = ["/gazebo/reset_world", "/gazebo/reset_simulation"]


def ensure_roscore() -> None:
    try:
        if not rosgraph.is_master_online():
            raise RuntimeError("ROS master is not online.")
    except Exception as exc:
        raise RuntimeError("Failed to reach ROS master. Is roscore running?") from exc


def _system_state():
    master = rosgraph.Master("/check_ros_gazebo_ready")
    try:
        return master.getSystemState()
    except Exception as exc:
        raise RuntimeError("Failed to contact ROS master. Is roscore running?") from exc


def wait_for_topics(timeout_s: float = 2.0, poll_s: float = 0.2) -> None:
    deadline = time.time() + timeout_s
    missing = set(REQUIRED_TOPICS)
    while time.time() < deadline:
        pubs, subs, _ = _system_state()
        topics = {name for name, _ in pubs} | {name for name, _ in subs}
        missing = {topic for topic in REQUIRED_TOPICS if topic not in topics}
        if not missing:
            return
        time.sleep(poll_s)
    raise RuntimeError(f"Missing ROS topics: {', '.join(sorted(missing))}")


def check_services(timeout_s: float = 2.0) -> None:
    _, _, srvs = _system_state()
    available = {name for name, _ in srvs}
    if not any(service in available for service in REQUIRED_SERVICES):
        raise RuntimeError(
            "Missing ROS services: /gazebo/reset_world or /gazebo/reset_simulation"
        )
    for service in REQUIRED_SERVICES:
        if service not in available:
            continue
        try:
            rospy.wait_for_service(service, timeout=timeout_s)
            return
        except Exception:
            continue
    raise RuntimeError(
        "ROS service exists but is not responding: /gazebo/reset_world or /gazebo/reset_simulation"
    )


def check_env() -> None:
    env = gym.make("GoalReachROS-v0")
    try:
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, _reward, _terminated, _truncated, info = env.step(action)
        min_lidar = info.get("min_lidar")
        theta_d = info.get("theta_d")
        dis = info.get("dis")
        print(
            f"obs_shape={getattr(obs, 'shape', None)} "
            f"min_lidar={min_lidar} theta_d={theta_d} dis={dis}"
        )
    finally:
        env.close()


def main() -> None:
    try:
        ensure_roscore()
        wait_for_topics()
        check_services()
        check_env()
    except Exception as exc:
        msg = str(exc) or repr(exc)
        print(f"ERROR: {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
