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
from agents.reinforce_agent import ReinforceAgent
from utils.run_artifacts import save_run_config
import warnings

# 屏蔽 CartPole-v0 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*CartPole-v0.*")

def train_reinforce():
    cfg = Config(algo='Reinforce')
    
    # --- TensorBoard 配置 ---
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results/Reinforce", f"Reinforce_{curr_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    x_threshold = 2.4
    theta_threshold_deg = 6.4
    save_run_config(
        log_dir,
        cfg,
        extra={
            "env_overrides": {
                "x_threshold": x_threshold,
                "theta_threshold_deg": theta_threshold_deg,
                "theta_threshold_radians": theta_threshold_deg * (np.pi / 180),
            }
        },
    )
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"\n{'='*60}")
    print(f"开始训练 REINFORCE | 环境: {cfg.env_name} | 设备: {cfg.device}")
    print(f"TensorBoard: {log_dir}")
    print(f"{'='*60}\n")
    
    base_seed = cfg.seed
    env = gym.make(cfg.env_name)
    env.unwrapped.x_threshold = x_threshold
    env.unwrapped.theta_threshold_radians = theta_threshold_deg * (np.pi / 180)  # 转为弧度

    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    try:
        env.action_space.seed(base_seed)
        env.observation_space.seed(base_seed)
    except Exception:
        pass
    
    agent = ReinforceAgent(cfg)
    return_list = []
    
    total_steps = 0 
    last_log_time = time.time() 
    last_log_total_steps = 0    
    current_sps = 0             

    with tqdm(total=cfg.episodes, desc="Training", unit="ep", dynamic_ncols=True, colour='blue') as pbar:      
        for i in range(cfg.episodes):
            agent.update_epsilon(i)
            state, _ = env.reset(seed=base_seed + i)
            
            done = False
            episode_return = 0
            episode_steps = 0 
            
            # [Reinforce 特有] 收集一整条轨迹
            transition_dict = {
                'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []
            }
            
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                
                state = next_state
                episode_return += reward
                total_steps += 1
                episode_steps += 1
                
            # [Reinforce 特有] 回合结束后更新一次
            loss = agent.update(transition_dict) 
            # Reinforce 一回合只有一个 Loss，无需像DQN那样求平均
            
            # --- 数据记录 ---
            return_list.append(episode_return)
            if len(return_list) >= 50:
                paper_avg_score = np.mean(return_list[-50:])
            else:
                paper_avg_score = np.mean(return_list)
            
            # --- TensorBoard ---
            writer.add_scalar("Train/01_Episode_Reward", episode_return, i)
            writer.add_scalar("Train/02_Avg_Reward_ep50", paper_avg_score, i)
            writer.add_scalar("Train/03_Epsilon", agent.epsilon, i) # 通常为0
            writer.add_scalar("Train/04_Avg_Loss", loss, i)

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
                    f"Loss: {loss:.3f}"
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
    plt.title('REINFORCE (CartPole-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.xlim(0, 500)  # 设置 x 轴范围 (最小值, 最大值)
    plt.ylim(0, 200)   # 设置 y 轴范围 (最小值, 最大值)
    plt.savefig(os.path.join(log_dir, 'reinforce_result.png'))
    print(f"训练结束，结果已保存至 {log_dir}\n")

if __name__ == '__main__':
    train_reinforce()
