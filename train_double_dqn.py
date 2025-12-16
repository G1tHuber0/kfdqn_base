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
from agents.double_dqn_agent import DoubleDQNAgent # [修改] 导入 DoubleDQNAgent

def train_double_dqn():
    # [修改] 指定算法名称用于配置加载(如果Config有逻辑)
    cfg = Config(algo='Double') 
    
    # --- TensorBoard 配置 ---
    # [修改] 路径改为 results/DoubleDQN
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results/DoubleDQN", f"DoubleDQN_{curr_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"\n{'='*60}")
    print(f"开始训练 Double DQN | 环境: {cfg.env_name} | 设备: {cfg.device}")
    print(f"TensorBoard: {log_dir}")
    print(f"{'='*60}\n")
    
    env = gym.make(cfg.env_name)
    env.unwrapped.x_threshold = 2.4 
    env.unwrapped.theta_threshold_radians = 41.8 * (np.pi / 180)  # 转为弧度
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # [修改] 实例化 DoubleDQNAgent
    agent = DoubleDQNAgent(cfg)
    buffer = ReplayBuffer(cfg.buffer_size)
    return_list = []
    
    total_steps = 0 

    # 速度计算变量
    last_log_time = time.time() 
    last_log_total_steps = 0    
    current_sps = 0             

    with tqdm(total=cfg.episodes, desc="Training", unit="ep", dynamic_ncols=True, colour='magenta') as pbar:      
        for i in range(cfg.episodes):
            agent.update_epsilon(i)
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
            if len(return_list) >= 50:
                paper_avg_score = np.mean(return_list[-50:])
            else:
                paper_avg_score = np.mean(return_list)
            avg_loss = episode_loss / update_count if update_count > 0 else 0

            # --- TensorBoard ---
            writer.add_scalar("Train/01_Episode_Reward", episode_return, i)
            writer.add_scalar("Train/02_Avg_Reward_ep50", paper_avg_score, i)
            writer.add_scalar("Train/03_Epsilon", agent.epsilon, i)
            writer.add_scalar("Train/04_Avg_Loss", avg_loss, i)

            writer.add_scalar("Train_Steps/Episode_Reward", episode_return, total_steps)
            writer.add_scalar("Train_Steps/Epsilon", agent.epsilon, total_steps)

            # --- 屏幕打印日志 ---
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
                    f"Epsilon: {agent.epsilon:.3f} | "
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
    plt.title('Double DQN (CartPole-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.savefig(os.path.join(log_dir, 'double_dqn_result.png'))
    print(f"\n训练结束，结果已保存至 {log_dir}")

if __name__ == '__main__':
    train_double_dqn()