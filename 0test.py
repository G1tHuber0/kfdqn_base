import gymnasium as gym
import numpy as np
import math

def test_cartpole_env(env_name="CartPole-v0"):
    print(f"\n{'='*20} 测试环境: {env_name} {'='*20}")
    
    # 1. 加载环境 (render_mode="human" 可以让你看到动画，不想看这就删掉)
    try:
        env = gym.make(env_name, render_mode="rgb_array") 
    except:
        env = gym.make(env_name) # 兼容旧版 gym

    # ==========================================
    # Part 1: 静态空间检查 (Static Specs)
    # ==========================================
    print("\n[1. 动作空间 Action Space]")
    print(f"  类型: {env.action_space}")
    print(f"  动作数 (n): {env.action_space.n}")
    print(f"  含义: 0 -> 向左推, 1 -> 向右推")

    print("\n[2. 状态空间 Observation Space]")
    print(f"  类型: {env.observation_space}")
    print(f"  维度 (shape): {env.observation_space.shape}")
    
    # 获取理论上的最大最小值
    high = env.observation_space.high
    low = env.observation_space.low
    
    # 定义标准物理含义 (CartPole)
    names = ["小车位置 (x)", "小车速度 (x_dot)", "杆角度 (theta)", "杆角速度 (theta_dot)"]
    
    print("\n  状态变量详细范围:")
    for i in range(4):
        # 格式化输出，处理 inf
        h_str = f"{high[i]:.4f}" if high[i] < 1e10 else "Inf"
        l_str = f"{low[i]:.4f}" if low[i] > -1e10 else "-Inf"
        print(f"    Index {i} [{names[i]}]: 范围 [{l_str}, {h_str}]")

    # ==========================================
    # Part 2: 动态数值模拟 (Running Loop)
    # ==========================================
    print(f"\n{'='*20} 开始动态测试 (随机动作) {'='*20}")
    observation, info = env.reset(seed=42)
    
    print("初始状态 (Raw):", observation)
    
    # 运行 20 步看看数据长什么样
    for step in range(10):
        # 随机动作
        action = env.action_space.sample()
        
        # 执行一步
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 提取关键数据
        x, x_dot, theta, theta_dot = next_obs
        
        # === [关键验证] ===
        # 转换角度单位，帮你确认到底是不是弧度
        theta_deg = theta * (180 / math.pi)
        
        print(f"\nStep {step+1} | Action: {action} ({'Left' if action==0 else 'Right'})")
        print(f"  > Raw Obs (Gym输出): {next_obs}")
        print(f"  > 杆角度验证:")
        print(f"      弧度 (Raw): {theta:.6f} rad")
        print(f"      角度 (Deg): {theta_deg:.6f} deg")
        
        if terminated or truncated:
            print(f"\n*** 回合结束 (Step {step+1}) ***")
            break
            
        observation = next_obs

    env.close()
    print("\n测试结束。")

if __name__ == "__main__":
    test_cartpole_env()