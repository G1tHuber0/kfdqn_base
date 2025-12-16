# kfdqn_reproduction/utils/exploration.py

def get_linear_decay_epsilon(episode_idx, cfg):
    """
    计算线性衰减的 Epsilon 值
    逻辑：
    1. 在 decay_start 轮之前，保持 epsilon_start (预热)
    2. 之后在 decay_steps 轮内，线性衰减到 epsilon_end
    3. 之后保持 epsilon_end
    """
    # --- 修正点 1: 预热期应该返回 epsilon_start (1.0)，而不是 decay_start (50) ---
    if episode_idx < cfg.decay_start:
        return cfg.epsilon_start
    
    # --- 修正点 2: 计算已经衰减了多少步 ---
    steps_done = episode_idx - cfg.decay_start
    
    # --- 修正点 3: 防止除以0，且分母应该是 decay_steps (衰减持续时长) ---
    if cfg.decay_steps <= 0:
        return cfg.epsilon_end
        
    # --- 修正点 4: 计算进度 (0.0 -> 1.0) ---
    # 分母必须是 cfg.decay_steps，而不是 cfg.decay_start
    # 使用 min(1.0, ...) 确保进度不会超过 100%
    progress = min(1.0, steps_done / cfg.decay_steps)
    
    # 计算当前值
    epsilon_range = cfg.epsilon_start - cfg.epsilon_end
    current_epsilon = cfg.epsilon_start - (epsilon_range * progress)
    
    # 双重保险：确保不低于最小值
    return max(cfg.epsilon_end, current_epsilon)