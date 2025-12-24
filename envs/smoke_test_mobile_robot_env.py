import gymnasium as gym

import envs.mobile_robot_env  # noqa: F401


def _run_env(env_id: str) -> None:
    env = gym.make(env_id)
    obs, info = env.reset()
    assert obs.shape == (93,)
    printed = False
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if not printed:
            print(
                f"{env_id} | min_lidar={info.get('min_lidar'):.3f} "
                f"theta_d={info.get('theta_d'):.3f} dis={info.get('dis'):.3f} "
                f"reward={reward:.3f} term={terminated} trunc={truncated}"
            )
            printed = True
        if terminated or truncated:
            env.reset()
    env.close()


def main() -> None:
    _run_env("GoalReach-v0")
    _run_env("ObstacleAvoid-v0")


if __name__ == "__main__":
    main()
