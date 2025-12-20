import gymnasium as gym
import matplotlib.pyplot as plt
import os
import datetime
import time 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Config
from agents.reinforce_agent import ReinforceAgent
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


def train_reinforce():
    cfg = Config(algo='Reinforce')
    
    # --- TensorBoard 配置 ---
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("results/Reinforce", f"Reinforce_{curr_time}")
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
    
    # 2. 准备 TensorBoard 和 日志路径
    log_line(f"\n{'='*60}")
    log_line(f"开始训练 REINFORCE | 环境: {cfg.env_name} | 设备: {cfg.device}")
    log_line(f"{'='*60}")
    log_line("查看训练过程数据，请终端运行: tensorboard --logdir=results")
    log_line(f"{'='*60}\n")
    
    base_seed = cfg.seed
    env = make_env(cfg.env_name, x_threshold=X_THRESHOLD)

    seed_everything(base_seed, env=env)
    
    agent = ReinforceAgent(cfg)
    return_list = []
    
    total_steps = 0 
    last_log_total_steps = 0    
    current_sps = 0             

    tqdm_position = int(os.environ.get("TQDM_POSITION", "0"))
    with tqdm(total=cfg.episodes, desc="Reinforce", unit="ep", dynamic_ncols=True, colour='blue', position=tqdm_position, leave=True) as pbar:      
        for i in range(cfg.episodes):
            agent.update_epsilon(i)
            state, _ = env.reset(seed=episode_seed(base_seed, i))
            
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
            
            # --- 数据记录 ---
            return_list.append(episode_return)
            
            # --- TensorBoard ---
            writer.add_scalar("Train/01_Episode_Reward", episode_return, i)
            writer.add_scalar("Train/02_Epsilon", agent.epsilon, i)
            writer.add_scalar("Train/03_Avg_Loss", loss, i)

            # --- 屏幕打印 ---
            if (i + 1) % 10 == 0:
                steps_diff = total_steps - last_log_total_steps
                last_log_total_steps = total_steps

                log_msg = (
                    f"Ep: {i+1}/{cfg.episodes} | "
                    f"Steps: {episode_steps} | " 
                    f"Loss: {loss:.3f}"
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
    plt.plot(return_list, color="#034fc2")
    plt.title('REINFORCE (CartPole-v0)')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.xlim(0, 500)
    plt.ylim(0, 205)
    plt.yticks([0, 50, 100, 150, 200])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, 'reinforce_result.png'))
    log_line(f"训练结束，结果已保存至 {log_dir}\n")

if __name__ == '__main__':
    train_reinforce()
