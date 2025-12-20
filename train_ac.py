import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Config
# AC 不使用标准的 ReplayBuffer，而是用列表或临时Buffer
from agents.ac_agent import ACAgent
from utils.run_artifacts import save_run_config, save_metrics
from utils.metrics import compute_training_metrics
from utils.seeding import episode_seed, seed_everything
import warnings

# 屏蔽 CartPole-v0 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*CartPole-v0.*")

X_THRESHOLD = 2.4

SILENT = os.environ.get("TRAIN_SILENT", "0") == "1"

def log_line(msg: str):
    if not SILENT:
        print(msg)

def log_tqdm(msg: str):
    if not SILENT:
        tqdm.write(msg)


def make_env(env_name: str = "CartPole-v0", *, x_threshold: float = X_THRESHOLD):
    env = gym.make(env_name)
    env.unwrapped.x_threshold = x_threshold
    return env


def train_ac():
    cfg = Config(algo='AC') 
    
    # --- TensorBoard 配置 ---
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results/AC", f"AC_{curr_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_run_config(
        log_dir,
        cfg,
        extra={
            "env_overrides": {
                "x_threshold": X_THRESHOLD,
            }
        },
    )
    writer = SummaryWriter(log_dir=log_dir)
    
    log_line(f"\n{'='*60}")
    log_line(f"开始训练 Actor-Critic | 环境: {cfg.env_name} | 设备: {cfg.device}")
    log_line(f"TensorBoard: {log_dir}")
    log_line(f"{'='*60}\n")
    
    base_seed = cfg.seed
    env = make_env(cfg.env_name, x_threshold=X_THRESHOLD)
    seed_everything(base_seed, env=env)
    
    agent = ACAgent(cfg)
    return_list = []
    total_steps = 0         
    # AC On-policy 临时存储
    temp_buffer = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}

    tqdm_position = int(os.environ.get("TQDM_POSITION", "0"))
    with tqdm(total=cfg.episodes, desc="AC", unit="ep", dynamic_ncols=True, colour='cyan', position=tqdm_position, leave=True) as pbar:      
        for i in range(cfg.episodes):
            agent.update_epsilon(i) # AC 通常不需要，但保持接口一致
            state, _ = env.reset(seed=episode_seed(base_seed, i))
            
            done = False
            episode_return = 0
            episode_loss = 0  
            update_count = 0  
            episode_steps = 0 
            
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 存入临时 Buffer
                temp_buffer['states'].append(state)
                temp_buffer['actions'].append(action)
                temp_buffer['rewards'].append(reward)
                temp_buffer['next_states'].append(next_state)
                temp_buffer['dones'].append(done)
                
                state = next_state
                episode_return += reward
                total_steps += 1
                episode_steps += 1
                
                # [AC 特有逻辑] 凑够 Batch 或回合结束就更新
                if len(temp_buffer['states']) >= cfg.batch_size or done:
                    batch_dict = {
                        'states': np.array(temp_buffer['states']),
                        'actions': np.array(temp_buffer['actions']),
                        'rewards': np.array(temp_buffer['rewards']),
                        'next_states': np.array(temp_buffer['next_states']),
                        'dones': np.array(temp_buffer['dones'])
                    }
                    loss = agent.update(batch_dict)
                    if loss is not None:
                        episode_loss += loss
                        update_count += 1
                    
                    # 清空 Buffer (On-Policy)
                    for k in temp_buffer: temp_buffer[k] = []

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
    plt.plot(return_list, color='#1b9e77')
    plt.title('Actor-Critic (CartPole-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.xlim(0, 500)
    plt.ylim(0, 205)
    plt.yticks([0, 50, 100, 150, 200])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, 'ac_result.png'))
    log_line(f"训练结束，结果已保存至 {log_dir}\n")

if __name__ == '__main__':
    train_ac()
