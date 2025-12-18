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
from agents.dqn_agent import DQNAgent
from utils.run_artifacts import save_run_config
import warnings

# 屏蔽 CartPole-v0 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*CartPole-v0.*")

def train_dqn():
    cfg = Config(algo='DQN')
    # --- TensorBoard 配置 ---
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results/DQN", f"DQN_{curr_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    x_threshold = 2.4
    theta_threshold_deg = 3.2
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
    print(f"开始训练 DQN | 环境: {cfg.env_name} | 设备: {cfg.device}")
    print(f"TensorBoard: {log_dir}")
    print(f"{'='*60}\n")
    
    base_seed = cfg.seed
    env = gym.make(cfg.env_name)
    env.unwrapped.x_threshold = x_threshold
    env.unwrapped.theta_threshold_radians = theta_threshold_deg * (np.pi / 180)  # 转为弧度
    # 统一随机种子：全局库 + 环境/动作空间
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    try:
        env.action_space.seed(base_seed)
        env.observation_space.seed(base_seed)
    except Exception:
        pass
    
    agent = DQNAgent(cfg)
    buffer = ReplayBuffer(cfg.buffer_size)
    return_list = []
    return_list_avg50 = []
    
    total_steps = 0  # 全局总步数

    # [新增] 为了计算平滑速度，初始化计时器
    last_log_time = time.time() 
    last_log_total_steps = 0    
    current_sps = 0             

    # 使用 tqdm 接管循环
    with tqdm(total=cfg.episodes, desc="Training", unit="ep", dynamic_ncols=True, colour='green') as pbar:      
        for i in range(cfg.episodes):
            # [说明] 这里移除了 ep_start_time，因为我们要改用区间时间来计算速度，防止分母过小
            agent.update_epsilon(i)
            state, _ = env.reset(seed=base_seed + i)
            
            done = False
            episode_return = 0
            episode_loss = 0  
            update_count = 0  
            episode_steps = 0 # 记录本回合步数
            
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                
                # 步数计数
                total_steps += 1
                episode_steps += 1
                if buffer.size() > cfg.minimal_size and total_steps % cfg.train_freq == 0:
                    b_s, b_a, b_r, b_ns, b_d = buffer.sample(cfg.batch_size)
                    transition_dict = {
                        'states': b_s, 'actions': b_a, 'next_states': b_ns, 
                        'rewards': b_r, 'dones': b_d
                    }
                    loss = agent.update(transition_dict) 
                    if loss is not None:
                        episode_loss += loss
                        update_count += 1
                
                # if buffer.size() > cfg.minimal_size:
                #     b_s, b_a, b_r, b_ns, b_d = buffer.sample(cfg.batch_size)
                #     transition_dict = {
                #         'states': b_s, 'actions': b_a, 'next_states': b_ns, 
                #         'rewards': b_r, 'dones': b_d
                #     }
                #     loss = agent.update(transition_dict) 
                #     if loss is not None:
                #         episode_loss += loss
                #         update_count += 1
            
            # --- 数据记录 ---
            return_list.append(episode_return)
            if len(return_list) >= 50:
                paper_avg_score = np.mean(return_list[-50:])
            else:
                paper_avg_score = np.mean(return_list)

            avg_loss = episode_loss / update_count if update_count > 0 else 0
            return_list_avg50.append(paper_avg_score)
            # --- TensorBoard  ---
            writer.add_scalar("Train/01_Episode_Reward", episode_return, i)
            writer.add_scalar("Train/02_Avg_Reward_ep50", paper_avg_score, i)
            writer.add_scalar("Train/03_Epsilon", agent.epsilon, i)
            writer.add_scalar("Train/04_Avg_Loss", avg_loss, i)

            #  Step 维度的记录  ---
            # 这样你可以在 TensorBoard 左侧搜索 "Steps" 看到以步数为 X 轴的图
            writer.add_scalar("Train_Steps/Episode_Reward", episode_return, total_steps)
            writer.add_scalar("Train_Steps/Epsilon", agent.epsilon, total_steps)

            # --- 屏幕打印日志 ---
            # 利用每10回合打印的机会，计算一次“过去10回合的平均速度”
            if (i + 1) % 10 == 0:
                # 计算时间差和步数差
                now = time.time()
                elapsed_time = now - last_log_time
                steps_diff = total_steps - last_log_total_steps
                
                if elapsed_time > 0:
                    current_sps = int(steps_diff / elapsed_time)
                
                # 更新计时锚点
                last_log_time = now
                last_log_total_steps = total_steps

                # (严格保留你原来的打印格式)
                log_msg = (
                    f"Ep: {i+1}/{cfg.episodes} | "
                    f"Avg_Reward_ep50: {paper_avg_score:.1f} | "
                    f"Steps: {episode_steps} | " 
                    f"Epsilon: {agent.epsilon:.3f} | "
                    f"Loss: {avg_loss:.3f}"
                )
                tqdm.write(log_msg)

            # 更新进度条后缀 ---
            pbar.set_postfix({
                'step': f"{current_sps}/s",  # 这里现在显示的是平滑后的数值
                'total_steps': f"{total_steps}" 
            })
            pbar.update(1)
            
    env.close()
    writer.close()
    
    plt.figure()
    plt.figure(figsize=(10, 6)) # 建议稍微把图画大一点
    # 第一条线：原始数据 (Raw)
    plt.plot(return_list, label='Raw Returns', alpha=0.3, color='gray') 
    # 第二条线：平滑数据 (Average)
    plt.plot(return_list_avg50, label='Avg (50 eps)', color='red', linewidth=2)
    plt.title('DQN Baseline (CartPole-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.xlim(0, 500)  # 设置 x 轴范围 (最小值, 最大值)
    plt.ylim(0, 200)   # 设置 y 轴范围 (最小值, 最大值)
    plt.savefig(os.path.join(log_dir, 'dqn_baseline_result.png'))
    print(f"训练结束，结果已保存至 {log_dir}\n")

if __name__ == '__main__':
    train_dqn()
