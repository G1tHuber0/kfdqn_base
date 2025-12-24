import gymnasium as gym

import envs_ros  # noqa: F401


def main() -> None:
    try:
        env = gym.make("GoalReachROS-v0")
        obs, info = env.reset()
    except Exception:
        print("Please start roscore + gazebo + robot before running this test.")
        raise
    assert obs.shape == (93,)
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"reward={reward:.3f} term={terminated} trunc={truncated} "
            f"min_lidar={info.get('min_lidar')} "
            f"theta_d={info.get('theta_d')} dis={info.get('dis')}"
        )
        if terminated or truncated:
            env.reset()
    env.close()


if __name__ == "__main__":
    main()
