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
# AC 不使用标准的 ReplayBuffer，而是用列表或临时Buffer
from agents.ac_agent import ACAgent

def train_ac():
    cfg = Config(algo='AC') 
    
    # --- TensorBoard 配置 ---
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results/AC", f"AC_{curr_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"\n{'='*60}")
    print(f"开始训练 Actor-Critic | 环境: {cfg.env_name} | 设备: {cfg.device}")
    print(f"TensorBoard: {log_dir}")
    print(f"{'='*60}\n")
    
    env = gym.make(cfg.env_name)
    env.unwrapped.x_threshold = 2.4 
    env.unwrapped.theta_threshold_radians = 41.8 * (np.pi / 180)  # 转为弧度
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    agent = ACAgent(cfg)
    return_list = []
    
    total_steps = 0 
    last_log_time = time.time() 
    last_log_total_steps = 0    
    current_sps = 0             

    # AC On-policy 临时存储
    temp_buffer = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}

    with tqdm(total=cfg.episodes, desc="Training", unit="ep", dynamic_ncols=True, colour='cyan') as pbar:      
        for i in range(cfg.episodes):
            agent.update_epsilon(i) # AC 通常不需要，但保持接口一致
            state, _ = env.reset(seed=cfg.seed if i == 0 else None)
            
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
            if len(return_list) >= 50:
                paper_avg_score = np.mean(return_list[-50:])
            else:
                paper_avg_score = np.mean(return_list)
            avg_loss = episode_loss / update_count if update_count > 0 else 0

            # --- TensorBoard ---
            writer.add_scalar("Train/01_Episode_Reward", episode_return, i)
            writer.add_scalar("Train/02_Avg_Reward_ep50", paper_avg_score, i)
            # AC epsilon 通常为 0
            writer.add_scalar("Train/03_Epsilon", agent.epsilon, i) 
            writer.add_scalar("Train/04_Avg_Loss", avg_loss, i)

            writer.add_scalar("Train_Steps/Episode_Reward", episode_return, total_steps)

            # --- 屏幕打印 ---
            if (i + 1) % 10 == 0:
                now = time.time()
                elapsed_time = now - last_log_time
                steps_diff = total_steps - last_log_total_steps
                if elapsed_time > 0:
                    current_sps = int(steps_diff / elapsed_time)
                last_log_time = now
                last_log_total_steps = total_steps

                log_msg = (
                    f"Ep: {i+1}/{cfg.episodes} | "
                    f"Avg_Reward_ep50: {paper_avg_score:.1f} | "
                    f"Steps: {episode_steps} | " 
                    f"Loss: {avg_loss:.3f}"
                )
                tqdm.write(log_msg)

            pbar.set_postfix({
                'step': f"{current_sps}/s",
                'total_steps': f"{total_steps}" 
            })
            pbar.update(1)
            
    env.close()
    writer.close()
    
    plt.figure()
    plt.plot(return_list)
    plt.title('Actor-Critic (CartPole-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.savefig(os.path.join(log_dir, 'ac_result.png'))
    print(f"\n训练结束，结果已保存至 {log_dir}")

if __name__ == '__main__':
    train_ac()