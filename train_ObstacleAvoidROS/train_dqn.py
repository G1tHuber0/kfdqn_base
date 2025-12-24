import datetime
import math
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import envs_ros  # noqa: F401

from config import Config
from utils.replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent
from utils.run_artifacts import save_run_config, save_metrics
from utils.metrics import compute_training_metrics
from utils.seeding import episode_seed, seed_everything

ENV_NAME = "ObstacleAvoidROS-v0"
SUCCESS_THRESHOLD = -110.0
SUCCESS_REWARD_CUTOFF = -200.0
SUCCESS_TARGET_COUNT = 250
REWARD_BIN_RULES = [
    {"label": "Fail (-200)", "left": -200.0, "right": -200.0, "left_open": False, "right_open": False},
    {"label": "Success (Normal)", "left": -200.0, "right": -100.0, "left_open": True, "right_open": False},
    {"label": "Success (High Performance)", "left": -100.0, "right": math.inf, "left_open": True, "right_open": False},
]
REWARD_NORM_RANGE = (-200.0, 0.0)
CUMULATIVE_TARGET = math.inf

SILENT = os.environ.get("TRAIN_SILENT", "0") == "1"

def log_line(msg: str):
    if not SILENT:
        print(msg)

def log_tqdm(msg: str):
    if not SILENT:
        tqdm.write(msg)


def make_env(env_name: str = ENV_NAME):
    env = gym.make(env_name)
    return env


def train_dqn():
    cfg = Config(algo='DQN', env_name=ENV_NAME)
    try:
        cfg.episodes = int(os.environ.get("TRAIN_EP", cfg.episodes))
    except Exception:
        pass
    # --- TensorBoard 配置 ---
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results_ObstacleAvoidROS/DQN", f"DQN_{curr_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_run_config(log_dir, cfg)
    writer = SummaryWriter(log_dir=log_dir)
    
    log_line(f"\n{'='*60}")
    log_line(f"开始训练 DQN | 环境: {cfg.env_name} | 设备: {cfg.device}")
    log_line(f"TensorBoard: {log_dir}")
    log_line(f"{'='*60}\n")
    
    base_seed = cfg.seed
    env = make_env(cfg.env_name)
    # 统一随机种子：全局库 + 环境/动作空间
    seed_everything(base_seed, env=env)
    
    agent = DQNAgent(cfg)
    buffer = ReplayBuffer(cfg.buffer_size)
    return_list = []

    total_steps = 0  # 全局总步数
    success_count = 0

    # 使用 tqdm 接管循环
    tqdm_position = int(os.environ.get("TQDM_POSITION", "0"))
    with tqdm(total=cfg.episodes, desc="DQN", unit="ep", dynamic_ncols=True, colour='green', position=tqdm_position, leave=True) as pbar:      
        for i in range(cfg.episodes):
            agent.update_epsilon(i)
            state, _ = env.reset(seed=episode_seed(base_seed, i))
            
            done = False
            episode_return = 0
            episode_loss = 0  
            update_count = 0  
            episode_steps = 0 # 记录本回合步数
            last_info = {}
            
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                total_steps += 1
                episode_steps += 1
                last_info = info
                if buffer.size() > cfg.minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = buffer.sample(cfg.batch_size)
                    transition_dict = {
                        'states': b_s, 'actions': b_a, 'next_states': b_ns, 
                        'rewards': b_r, 'dones': b_d
                    }
                    loss = agent.update(transition_dict) 
                    if loss is not None:
                        episode_loss += loss
                        update_count += 1
                
            # --- 数据记录 ---
            return_list.append(episode_return)
            avg_loss = episode_loss / update_count if update_count > 0 else 0
            if last_info.get("is_success"):
                success_count += 1
            success_rate = success_count / (i + 1)
            # --- TensorBoard  ---
            writer.add_scalar("Train/01_Episode_Reward", episode_return, i)
            writer.add_scalar("Train/02_Epsilon", agent.epsilon, i)
            writer.add_scalar("Train/03_Avg_Loss", avg_loss, i)
            writer.add_scalar("Train/SuccessRate", success_rate, i)

            # --- 屏幕打印日志 ---
            if (i + 1) % 10 == 0:
                log_msg = (
                    f"Ep: {i+1}/{cfg.episodes} | "
                    f"Steps: {episode_steps} | " 
                    f"Epsilon: {agent.epsilon:.3f} | "
                    f"Loss: {avg_loss:.3f}"
                )
                log_tqdm(log_msg)
            # 更新进度条后缀 ---
            pbar.set_postfix({
                'total_steps': f"{total_steps}" 
            })
            pbar.update(1)

    raw_data_path = os.path.join(log_dir, "raw_episode_returns.csv")
    with open(raw_data_path, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["episode", "return"])
        for idx, val in enumerate(return_list, start=1):
            writer_csv.writerow([idx, val])
    log_line(f"原始回报数据已保存至 (CSV): {raw_data_path}")

    metrics = compute_training_metrics(
        return_list,
        None,
        success_threshold=SUCCESS_THRESHOLD,
        cumulative_target=CUMULATIVE_TARGET,
        custom_bins=REWARD_BIN_RULES,
        success_reward_cutoff=SUCCESS_REWARD_CUTOFF,
        success_target_count=SUCCESS_TARGET_COUNT,
        reward_norm_range=REWARD_NORM_RANGE,
    )
    save_metrics(log_dir, metrics)

    env.close()
    writer.close()
    
    plt.figure()
    plt.figure(figsize=(10, 6))
    plt.plot(return_list, color='#003f5c') 
    plt.title('DQN Baseline (MountainCar-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.xlim(0, cfg.episodes)
    plt.ylim(-200, 0)
    plt.yticks([-200, -150, -100, -50, 0])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, 'dqn_baseline_result.png'))
    log_line(f"训练结束，结果已保存至 {log_dir}\n")

if __name__ == '__main__':
    train_dqn()
