import datetime
import math
import os
import sys
import time
from pathlib import Path
import csv  # 需要导入 csv 库
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import envs_ros  # noqa: F401

from config import Config
from utils.replay_buffer import ReplayBuffer
from agents.kfdqn_agent import KFDQNAgent
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
    env = gym.make(env_name, render_mode=None)
    return env

def train_kfdqn():
    # 1. 初始化配置 (自动加载 KFDQN 参数)
    cfg = Config(algo="KFDQN", env_name=ENV_NAME)
    cfg.ep_r = 50  # 强制引导阶段长度]
    cfg.seed = 122
    try:
        cfg.episodes = int(os.environ.get("TRAIN_EP", cfg.episodes))
    except Exception:
        pass

    # 3. 环境与种子设置
    base_seed = cfg.seed 
    env = make_env(cfg.env_name)

    # 2. 准备 TensorBoard 和 日志路径
    log_line(f"\n{'='*60}")
    log_line(f"开始训练 KFDQN | 环境: {cfg.env_name} | 随机种子: {cfg.seed} | 设备: {cfg.device}")
    log_line(f"{'='*60}")
    log_line("查看训练过程数据，请终端运行: tensorboard --logdir=results_MountainCar")
    log_line(f"{'='*60}\n")
    # --- TensorBoard 配置 ---
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results_ObstacleAvoidROS/KFDQN", f"KFDQN_{curr_time}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_run_config(log_dir, cfg)
    writer = SummaryWriter(log_dir=log_dir)

    

    seed_everything(base_seed, env=env)

    # 4. 初始化 Agent 和 Buffer
    agent = KFDQNAgent(cfg)
    buffer = ReplayBuffer(cfg.buffer_size)

    # 5. 训练循环变量
    return_list = []
    total_steps = 0
    success_count = 0
    

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
            last_info = {}

            while not done:
                # [关键] 传入 episode_idx 以便 HYAS 策略判断阶段 (强制引导 vs 混合决策)
                action, action_type, a_q = agent.take_action(state, episode_idx=ep)
                if action_type=='a_f':
                    count_af += 1
                elif action_type=='hya':
                    count_hya += 1
                    if a_q==action:
                        count_aq += 1
                        
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                buffer.add(state, action, reward, next_state, done)
                state = next_state
                ep_return += reward
                total_steps += 1
                last_info = info

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
            if last_info.get("is_success"):
                success_count += 1
            success_rate = success_count / (ep + 1)

            avg_q_loss = ep_q_loss / max(1, updates)
            avg_fuzzy_loss = ep_fuzzy_loss / max(1, updates)

            # TensorBoard 记录
            writer.add_scalar("Train/01_Episode_Reward", ep_return, ep)
            writer.add_scalar("Train/02_Epsilon", agent.epsilon, ep)
            writer.add_scalar("Train/03_Q_Loss", avg_q_loss, ep)
            writer.add_scalar("actions/01_Count_a_f", count_af, ep)
            writer.add_scalar("actions/02_Count_hya", count_hya, ep)
            writer.add_scalar("actions/03_Aq_in_Hya_Rate", aq_hya_rate, ep)
            writer.add_scalar("Train/SuccessRate", success_rate, ep)
            
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

    raw_data_path = os.path.join(log_dir, "raw_episode_returns.csv")
    
    with open(raw_data_path, "w", newline='') as f:
        csv_writer = csv.writer(f)
        # 写入表头 (可选，方便 Origin 识别列名)
        csv_writer.writerow(["episode", "return"])
        # 写入数据
        for idx, val in enumerate(return_list):
            csv_writer.writerow([idx + 1, val])
            
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

    writer.close()
    env.close()
    
    # 6. 保存结果图
    plt.figure(figsize=(10, 6)) 
    # 第一条线：原始数据 (Raw)
    plt.plot(return_list,  color="#005d9b") 
    plt.title('KFDQN (MountainCar-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    # --- 限制区域 ---
    plt.xlim(0, cfg.episodes)
    plt.ylim(-200, 0)
    plt.yticks([-200, -150, -100, -50, 0])
    plt.grid(True, alpha=0.3) # 加上网格线更方便看读数
    plt.savefig(os.path.join(log_dir, 'kfdqn_result.png'))
    log_line(f"训练结束，结果图已保存至: {log_dir}\n")

if __name__ == "__main__":
    train_kfdqn()
