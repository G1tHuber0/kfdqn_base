import gymnasium as gym
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import datetime
import time 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Config
from utils.replay_buffer import ReplayBuffer
from agents.dueling_dqn_agent import DuelingDQNAgent
from utils.run_artifacts import save_run_config, save_metrics
from utils.metrics import compute_training_metrics
import warnings

# 屏蔽 CartPole-v0 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*CartPole-v0.*")

SILENT = os.environ.get("TRAIN_SILENT", "0") == "1"

def log_line(msg: str):
    if not SILENT:
        print(msg)

def log_tqdm(msg: str):
    if not SILENT:
        tqdm.write(msg)


def make_env(env_name: str = "CartPole-v0"):
    env = gym.make(env_name)
    env.unwrapped.x_threshold = 2.4  # default is 2.4
    return env


def train_dueling_dqn():
    cfg = Config(algo='Dueling')
    
    # --- TensorBoard 配置 ---
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results/DuelingDQN", f"DuelingDQN_{curr_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_run_config(log_dir, cfg)
    writer = SummaryWriter(log_dir=log_dir)
    
    log_line(f"\n{'='*60}")
    log_line(f"开始训练 Dueling DQN | 环境: {cfg.env_name} | 设备: {cfg.device}")
    log_line(f"TensorBoard: {log_dir}")
    log_line(f"{'='*60}\n")
    
    base_seed = cfg.seed
    env = make_env(cfg.env_name)

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    try:
        env.action_space.seed(base_seed)
        env.observation_space.seed(base_seed)
    except Exception:
        pass
    
    agent = DuelingDQNAgent(cfg)
    buffer = ReplayBuffer(cfg.buffer_size)
    return_list = []
    
    total_steps = 0 

    tqdm_position = int(os.environ.get("TQDM_POSITION", "0"))
    with tqdm(total=cfg.episodes, desc="DuelingDQN", unit="ep", dynamic_ncols=True, colour='yellow', position=tqdm_position, leave=True) as pbar:      
        for i in range(cfg.episodes):
            agent.update_epsilon(i)
            state, _ = env.reset(seed=base_seed + i)
            
            done = False
            episode_return = 0
            episode_loss = 0  
            update_count = 0  
            episode_steps = 0 
            
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                
                total_steps += 1
                episode_steps += 1
                
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

            # --- TensorBoard ---
            writer.add_scalar("Train/01_Episode_Reward", episode_return, i)
            writer.add_scalar("Train/02_Epsilon", agent.epsilon, i)
            writer.add_scalar("Train/03_Avg_Loss", avg_loss, i)

            # --- 屏幕打印 ---
            if (i + 1) % 10 == 0:
                log_msg = (
                    f"Ep: {i+1}/{cfg.episodes} | "
                    f"Steps: {episode_steps} | " 
                    f"Epsilon: {agent.epsilon:.3f} | "
                    f"Loss: {avg_loss:.3f}"
                )
                log_tqdm(log_msg)

            pbar.set_postfix({
                'total_steps': f"{total_steps}" 
            })
            pbar.update(1)
    metrics = compute_training_metrics(return_list, None, success_threshold=200.0, cumulative_target=50000.0)
    save_metrics(log_dir, metrics)

    env.close()
    writer.close()
    
    plt.figure()
    plt.figure(figsize=(10, 6))
    plt.plot(return_list, label='Returns', color='#fdae61')
    plt.title('Dueling DQN (CartPole-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.xlim(0, 500)
    plt.ylim(0, 200)
    plt.yticks([0, 50, 100, 150, 200])
    plt.savefig(os.path.join(log_dir, 'dueling_dqn_result.png'))
    log_line(f"训练结束，结果已保存至 {log_dir}\n")

if __name__ == '__main__':
    train_dueling_dqn()
