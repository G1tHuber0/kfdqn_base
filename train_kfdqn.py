import gymnasium as gym
import numpy as np
import random
import time
import datetime
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from utils.replay_buffer import ReplayBuffer
from agents.kfdqn_agent import KFDQNAgent

def make_env(env_name="CartPole-v0"):
    # 创建环境
    env = gym.make(env_name)
    # 使用 .unwrapped 访问底层物理属性
    env.unwrapped.x_threshold = 2.4 
    env.unwrapped.theta_threshold_radians = 41.8 * (np.pi / 180)  # 转为弧度
    return env

def train_kfdqn():
    # 1. 初始化配置 (自动加载 KFDQN 参数)
    cfg = Config(algo="KFDQN")

    # 2. 准备 TensorBoard 和 日志路径
    print(f"\n{'='*60}")
    print(f"开始训练 KFDQN | 环境: {cfg.env_name} | 设备: {cfg.device}")
    print(f"{'='*60}\n")
    # --- TensorBoard 配置 ---
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results/KFDQN", f"KFDQN_{curr_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # 3. 环境与种子设置
    env = make_env(cfg.env_name)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 4. 初始化 Agent 和 Buffer
    agent = KFDQNAgent(cfg)
    buffer = ReplayBuffer(cfg.buffer_size)

    # 5. 训练循环变量
    return_list = []
    total_steps = 0
    last_log_time = time.time()
    last_log_total_steps = 0

    # 使用 tqdm 显示进度条
    with tqdm(total=cfg.episodes, desc="Training", unit="ep", dynamic_ncols=True, colour='red') as pbar:
        for ep in range(cfg.episodes):
            # [关键] 更新参数: Epsilon, 混合权重 m/n, 硬更新检查
            agent.update_parameters(ep)
            # 重置环境
            state, _ = env.reset(seed=cfg.seed + ep)
            done = False
            
            ep_return = 0.0
            ep_q_loss = 0.0
            ep_fuzzy_loss = 0.0
            updates = 0

            while not done:
                # [关键] 传入 episode_idx 以便 HYAS 策略判断阶段 (强制引导 vs 混合决策)
                action = agent.take_action(state, episode_idx=ep)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                buffer.add(state, action, reward, next_state, done)
                state = next_state
                ep_return += reward
                total_steps += 1

                # 经验回放训练
                if buffer.size() > cfg.minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = buffer.sample(cfg.batch_size)
                    # 构造字典传入 agent.update
                    transition_dict = {
                        "states": b_s,
                        "actions": b_a,
                        "rewards": b_r,
                        "next_states": b_ns,
                        "dones": b_d,
                    }
                    # [关键] 传入 episode_idx 以便判断是 监督学习 还是 混合TD学习
                    losses = agent.update(transition_dict, episode_idx=ep)
                    ep_q_loss += losses["q_loss"]
                    ep_fuzzy_loss += losses["fuzzy_loss"]
                    updates += 1

            # --- 本回合结束，记录数据 ---
            return_list.append(ep_return)

            # 计算最近50回合平均分 (论文常用指标)
            paper_avg = float(np.mean(return_list[-50:])) if len(return_list) >= 50 else float(np.mean(return_list))
            
            avg_q_loss = ep_q_loss / max(1, updates)
            avg_fuzzy_loss = ep_fuzzy_loss / max(1, updates)

            # TensorBoard 记录
            writer.add_scalar("Train/01_Episode_Reward", ep_return, ep)
            writer.add_scalar("Train/02_Avg_Reward_ep50", paper_avg, ep)
            writer.add_scalar("Train/03_Epsilon", agent.epsilon, ep)
            writer.add_scalar("Train/04_Q_Loss", avg_q_loss, ep)
            
            # KFDQN 特有指标
            writer.add_scalar("KFDQN/01_Parameter_m", agent.m, ep) # 观察 m 值的衰减
            writer.add_scalar("KFDQN/02_Fuzzy_Loss", avg_fuzzy_loss, ep) # 观察模糊系统的学习情况

            # 控制台输出 (每10轮更新一次详细信息)
            if (ep + 1) % 10 == 0:
                now = time.time()
                # 计算 SPS (Steps Per Second)
                sps = int((total_steps - last_log_total_steps) / max(1e-6, (now - last_log_time)))
                last_log_time = now
                last_log_total_steps = total_steps
                
                tqdm.write(f"Ep:{ep+1} | AvgRw:{paper_avg:.1f} | Eps:{agent.epsilon:.2f} | m:{agent.m:.2f} | SPS:{sps}")

            pbar.update(1)

    writer.close()
    env.close()
    
    # 6. 保存结果图 (跟其他基线保持一致)
    plt.figure()
    plt.plot(return_list)
    plt.title('KFDQN (CartPole-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.savefig(os.path.join(log_dir, 'kfdqn_result.png'))
    print(f"\n训练结束，结果图已保存至: {log_dir}")
    print("Done.")

if __name__ == "__main__":
    train_kfdqn()