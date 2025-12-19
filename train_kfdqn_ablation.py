import gymnasium as gym
import numpy as np
import random
import time
import datetime
import os
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from utils.replay_buffer import ReplayBuffer
from agents.kfdqn_agent import KFDQNAgent
from utils.run_artifacts import save_run_config, save_metrics
from utils.metrics import compute_training_metrics
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*CartPole-v0.*")

X_THRESHOLD = 2.4
THETA_THRESHOLD_DEG = 6.4

SILENT = os.environ.get("TRAIN_SILENT", "0") == "1"

def log_line(msg: str):
    if not SILENT:
        print(msg)

def log_tqdm(msg: str):
    if not SILENT:
        tqdm.write(msg)


def make_env(env_name: str = "CartPole-v0", *, x_threshold: float = X_THRESHOLD, theta_threshold_deg: float = THETA_THRESHOLD_DEG):
    env = gym.make(env_name)
    env.unwrapped.x_threshold = x_threshold
    env.unwrapped.theta_threshold_radians = theta_threshold_deg * (np.pi / 180)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="KFDQN Ablation Training")
    parser.add_argument(
        "--variant",
        choices=["full", "no_hya", "no_hyl"],
        default="no_hya",
        help="full: 原始KFDQN; no_hya: 去除混合动作策略; no_hyl: 去除混合学习策略",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(algo="KFDQN")
    if args.variant == "no_hya":
        cfg.use_hybrid_action = False
        cfg.use_hybrid_learning = True
        variant_tag = "no_hya"
    elif args.variant == "no_hyl":
        cfg.use_hybrid_action = True
        cfg.use_hybrid_learning = False
        variant_tag = "no_hyl"
    else:
        cfg.use_hybrid_action = True
        cfg.use_hybrid_learning = True
        variant_tag = "full"

    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results/KFDQN_ablation", f"{variant_tag}_{curr_time}")
    os.makedirs(log_dir, exist_ok=True)
    save_run_config(
        log_dir,
        cfg,
        extra={
            "variant": variant_tag,
            "env_overrides": {
                "x_threshold": X_THRESHOLD,
                "theta_threshold_deg": THETA_THRESHOLD_DEG,
                "theta_threshold_radians": THETA_THRESHOLD_DEG * (np.pi / 180),
            },
        },
    )
    writer = SummaryWriter(log_dir=log_dir)

    log_line(f"\n{'='*60}")
    log_line(f"开始训练 KFDQN ({variant_tag}) | 环境: {cfg.env_name} | 设备: {cfg.device}")
    log_line(f"TensorBoard: {log_dir}")
    log_line(f"{'='*60}\n")

    base_seed = cfg.seed
    env = make_env(cfg.env_name, x_threshold=X_THRESHOLD, theta_threshold_deg=THETA_THRESHOLD_DEG)

    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)
    try:
        env.action_space.seed(base_seed)
        env.observation_space.seed(base_seed)
    except Exception:
        pass

    agent = KFDQNAgent(cfg)
    buffer = ReplayBuffer(cfg.buffer_size)

    return_list = []
    total_steps = 0
    last_log_time = time.time()
    last_log_total_steps = 0
    tqdm_position = int(os.environ.get("TQDM_POSITION", "0"))

    with tqdm(total=cfg.episodes, desc=f"KFDQN-{variant_tag}", unit="ep", dynamic_ncols=True, colour='red', position=tqdm_position, leave=True) as pbar:
        for ep in range(cfg.episodes):
            agent.update_parameters(ep)
            state, _ = env.reset(seed=base_seed + ep)

            done = False
            ep_return = 0.0
            ep_q_loss = 0.0
            ep_fuzzy_loss = 0.0
            updates = 0

            while not done:
                action, _, _ = agent.take_action(state, episode_idx=ep)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                buffer.add(state, action, reward, next_state, done)
                state = next_state
                ep_return += reward
                total_steps += 1

                if buffer.size() >= cfg.minimal_size and total_steps % getattr(cfg, "train_freq", 1) == 0:
                    b_s, b_a, b_r, b_ns, b_d = buffer.sample(cfg.batch_size)
                    transition_dict = {
                        "states": b_s,
                        "actions": b_a,
                        "rewards": b_r,
                        "next_states": b_ns,
                        "dones": b_d,
                    }
                    losses = agent.update(transition_dict, episode_idx=ep)
                    ep_q_loss += losses["q_loss"]
                    ep_fuzzy_loss += losses["fuzzy_loss"]
                    updates += 1

            return_list.append(ep_return)
            avg_q_loss = ep_q_loss / max(1, updates)
            avg_fuzzy_loss = ep_fuzzy_loss / max(1, updates)

            writer.add_scalar("Train/01_Episode_Reward", ep_return, ep)
            writer.add_scalar("Train/02_Epsilon", agent.epsilon, ep)
            writer.add_scalar("Train/03_Q_Loss", avg_q_loss, ep)
            writer.add_scalar("Train/04_Fuzzy_Loss", avg_fuzzy_loss, ep)

            if (ep + 1) % 10 == 0:
                now = time.time()
                sps = int((total_steps - last_log_total_steps) / max(1e-6, (now - last_log_time)))
                last_log_time = now
                last_log_total_steps = total_steps
                log_tqdm(f"[{variant_tag}] Ep:{ep+1} | Reward:{ep_return:.1f} | Eps:{agent.epsilon:.2f} | SPS:{sps}")

            pbar.update(1)

    metrics = compute_training_metrics(return_list, None, success_threshold=200.0, cumulative_target=50000.0)
    save_metrics(log_dir, metrics)

    writer.close()
    env.close()

    plt.figure(figsize=(10, 6))
    plt.plot(return_list, label='Raw Returns', alpha=0.3, color='#66c2ff') 
    plt.title(f'KFDQN Ablation - {variant_tag}')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.xlim(0, 500)
    plt.ylim(0, 200)
    plt.yticks([0, 50, 100, 150, 200])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, f'kfdqn_ablation_{variant_tag}.png'))
    log_line(f"[{variant_tag}] 训练结束，结果图已保存至: {log_dir}\n")


if __name__ == "__main__":
    raise SystemExit(main())
