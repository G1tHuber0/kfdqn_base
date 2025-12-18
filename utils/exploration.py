# kfdqn_reproduction/utils/exploration.py
# import math
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

# def get_linear_decay_epsilon(episode_idx, cfg):
#     """
#     计算指数衰减的 Epsilon 值
#     公式: epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-1. * steps / decay_rate)
#     """
#     # 1. 预热期 (Warmup): 与线性衰减保持一致
#     if episode_idx < cfg.decay_start:
#         return cfg.epsilon_start

#     # 2. 计算有效步数
#     steps_done = episode_idx - cfg.decay_start

#     # 3. 指数衰减计算
#     # 这里复用 cfg.decay_steps 作为衰减的时间常数 (scale)
#     # 如果 cfg.decay_steps 越大，衰减越慢；越小，衰减越快。
#     # 通常建议 cfg.decay_steps 设置为总回合数的 1/5 到 1/3 左右
    
#     # 防止除以 0
#     if cfg.decay_steps <= 0: 
#         return cfg.epsilon_end

#     # 核心公式
#     # math.exp(-x) 当 x=0 时为 1，当 x 很大时趋近于 0
#     decay_factor = math.exp(-1. * steps_done / cfg.decay_steps)
    
#     epsilon_range = cfg.epsilon_start - cfg.epsilon_end
#     current_epsilon = cfg.epsilon_end + (epsilon_range * decay_factor)

#     # 4. 指数衰减天然不会低于 epsilon_end，但加上 max 双重保险也没错
#     return max(cfg.epsilon_end, current_epsilon)