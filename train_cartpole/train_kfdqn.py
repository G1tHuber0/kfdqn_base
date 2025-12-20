import datetime
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config
from utils.replay_buffer import ReplayBuffer
from agents.kfdqn_agent import KFDQNAgent
from utils.run_artifacts import save_run_config, save_metrics
from utils.metrics import compute_training_metrics
from utils.seeding import episode_seed, seed_everything
import warnings
# 仅忽略包含 "CartPole-v0" 文本的 DeprecationWarning
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
    # 创建环境
    env = gym.make(env_name)
    # 使用 .unwrapped 访问底层物理属性
    env.unwrapped.x_threshold = x_threshold
    return env

def train_kfdqn():
    # 1. 初始化配置 (自动加载 KFDQN 参数)
    cfg = Config(algo="KFDQN")
    cfg.ep_r = 50  # 强制引导阶段长度]

    # 2. 准备 TensorBoard 和 日志路径
    log_line(f"\n{'='*60}")
    log_line(f"开始训练 KFDQN | 环境: {cfg.env_name} | 设备: {cfg.device}")
    log_line(f"{'='*60}")
    log_line("查看训练过程数据，请终端运行: tensorboard --logdir=results_cartpole")
    log_line(f"{'='*60}\n")
    # --- TensorBoard 配置 ---
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results_cartpole/KFDQN", f"KFDQN_{curr_time}")
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

    # 3. 环境与种子设置
    base_seed = cfg.seed
    env = make_env(cfg.env_name, x_threshold=X_THRESHOLD)

    seed_everything(base_seed, env=env)

    # 4. 初始化 Agent 和 Buffer
    agent = KFDQNAgent(cfg)
    buffer = ReplayBuffer(cfg.buffer_size)

    # 5. 训练循环变量
    return_list = []
    total_steps = 0
    

    # 使用 tqdm 显示进度条
    tqdm_position = int(os.environ.get("TQDM_POSITION", "0"))
    with tqdm(total=cfg.episodes, desc="KFDQN", unit="ep", dynamic_ncols=True, colour='red', position=tqdm_position, leave=True) as pbar:
        for ep in range(cfg.episodes):
            # [关键] 更新参数: Epsilon, 混合权重 m/n, 硬更新检查
            agent.update_parameters(ep)
            # 重置环境
            state, _ = env.reset(seed=episode_seed(base_seed, ep))

            done = False
            ep_return = 0.0
            ep_q_loss = 0.0
            ep_fuzzy_loss = 0.0
            updates = 0
            count_af = 0
            count_hya = 0
            count_aq = 0
            aq_hya_rate = 0.0

            while not done:
                # [关键] 传入 episode_idx 以便 HYAS 策略判断阶段 (强制引导 vs 混合决策)
                action, action_type, a_q = agent.take_action(state, episode_idx=ep)
                if action_type=='a_f':
                    count_af += 1
                elif action_type=='hya':
                    count_hya += 1
                    if a_q==action:
                        count_aq += 1
                        
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                buffer.add(state, action, reward, next_state, done)
                state = next_state
                ep_return += reward
                total_steps += 1

                # 经验回放训练
                if buffer.size() >= cfg.minimal_size :
                    b_s, b_a, b_r, b_ns, b_d = buffer.sample(cfg.batch_size)
                    # 构造字典传入 agent.update
                    transition_dict = {
                        "states": b_s,
                        "actions": b_a,
                        "rewards": b_r,
                        "next_states": b_ns,
                        "dones": b_d,
                    }
                    # 传入 episode_idx 以便判断是 监督学习 还是 混合TD学习
                    losses = agent.update(transition_dict, episode_idx=ep)
                    ep_q_loss += losses["q_loss"]
                    ep_fuzzy_loss += losses["fuzzy_loss"]
                    updates += 1

            # --- 本回合结束，记录数据 ---
            if count_hya > 0:
                aq_hya_rate = count_aq / count_hya
            return_list.append(ep_return)

            avg_q_loss = ep_q_loss / max(1, updates)
            avg_fuzzy_loss = ep_fuzzy_loss / max(1, updates)

            # TensorBoard 记录
            writer.add_scalar("Train/01_Episode_Reward", ep_return, ep)
            writer.add_scalar("Train/02_Epsilon", agent.epsilon, ep)
            writer.add_scalar("Train/03_Q_Loss", avg_q_loss, ep)
            writer.add_scalar("actions/01_Count_a_f", count_af, ep)
            writer.add_scalar("actions/02_Count_hya", count_hya, ep)
            writer.add_scalar("actions/03_Aq_in_Hya_Rate", aq_hya_rate, ep)
            
            # KFDQN 特有指标
            writer.add_scalar("KFDQN/01_Parameter_m", agent.m, ep) # 观察 m 值的衰减
            writer.add_scalar("KFDQN/02_Fuzzy_Loss", avg_fuzzy_loss, ep) # 观察模糊系统的学习情况

            # 控制台输出 (每10轮更新一次详细信息)
            if (ep + 1) % 10 == 0:
                log_tqdm(f"Ep:{ep+1} | Reward:{ep_return:.1f} | Eps:{agent.epsilon:.2f} | m:{agent.m:.2f} | ")
            pbar.set_postfix({
                'total_steps': f"{total_steps}"
            })
            pbar.update(1)

    metrics = compute_training_metrics(return_list, None, success_threshold=200.0, cumulative_target=50000.0)
    save_metrics(log_dir, metrics)

    writer.close()
    env.close()
    
    # 6. 保存结果图
    plt.figure(figsize=(10, 6)) 
    # 第一条线：原始数据 (Raw)
    plt.plot(return_list,  color="#005d9b") 
    plt.title('KFDQN (CartPole-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    # --- 限制区域 ---
    plt.xlim(0, 500)
    plt.ylim(0, 205)
    plt.yticks([0, 50, 100, 150, 200])
    plt.grid(True, alpha=0.3) # 加上网格线更方便看读数
    plt.savefig(os.path.join(log_dir, 'kfdqn_result.png'))
    log_line(f"训练结束，结果图已保存至: {log_dir}\n")

if __name__ == "__main__":
    train_kfdqn()
