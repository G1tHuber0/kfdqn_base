import torch
import datetime
import os

class Config:
    def __init__(self, algo='DQN', env_name='CartPole-v0'):
        """
        :param algo: 'DQN', 'Double', 'Dueling', 'Reinforce', 'AC', 'KFDQN'
        """
        # --- 1. 环境与基础设置 ---
        self.algo = algo
        self.env_name = env_name
        # --- 自动适配环境参数 ---
        if "CartPole" in self.env_name:
            self.state_dim = 4
            self.action_dim = 2
            self.hidden_dim = 128
        elif "MountainCar" in self.env_name:
            self.state_dim = 2
            self.action_dim = 3
            self.hidden_dim = 128
        else:
            self.state_dim = 4
            self.action_dim = 2
            self.hidden_dim = 128
        # 默认随机种子可通过环境变量 TRAIN_SEED 覆盖（兼容单脚本运行与 run_all 批量运行）
        default_seed = 2
        try:
            self.seed = int(os.environ.get("TRAIN_SEED", str(default_seed)))
        except Exception:
            self.seed = default_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- 3. 训练通用参数 ---
        self.gamma = 0.98       
        self.episodes = 500     
        self.lr = 0.002   
        # 探索参数
        self.epsilon_start = 0.01
        self.epsilon_end = 0.01
        self.decay_start =0
        self.decay_steps = 0      
        # 梯度剪裁（<=0 或 None 不启用）
        self.grad_clip_norm = None
        # --- 4. 初始化特定算法参数 ---
        # 先给 KFDQN 特有参数赋默认值 None，防止报错
        self.h1 = None
        self.h2 = None
        self.m_base = None
        # 加载具体参数
        self._set_algo_specific_params()

    def _set_algo_specific_params(self):
        """根据不同算法调整参数"""
        
        # Group A: DQN 
        if self.algo in ['DQN', 'Double', 'Dueling']:
            self.buffer_size = 10000 
            self.minimal_size = 500  
            self.batch_size = 64     
            self.target_update = 10
            self.train_freq = 1
            self.gradient_steps = 1 
        # Group B: Reinforce
        elif self.algo == 'Reinforce':
            self.buffer_size = None
            self.minimal_size = None
            self.batch_size = None
            self.target_update = None
            self.epsilon_start = None
        # Group C: Actor-Critic
        elif self.algo == 'AC':
            self.buffer_size = None 
            self.minimal_size = 0
            self.batch_size = 1
            self.target_update = None
            self.epsilon_start = None
            self.lr_actor = 0.0005
            self.lr_critic = 0.002
        # Group D: KFDQN (Knowledge Guided)
        elif self.algo == 'KFDQN':

            self.use_hybrid_action = True
            self.use_hybrid_learning = True

            self.buffer_size = 10000
            self.minimal_size = 500
            self.batch_size = 64
            self.target_update = 10
            self.train_freq =1

            # 探索参数
            self.epsilon_start = 0.01
            self.epsilon_end = 0.01
            self.decay_start =0
            self.decay_steps = 0 

            # KFDQN 关键超参（按论文）
            self.h1 = 0.1
            self.h2 = 0.08
            # 监督阶段长度（论文描述常用 50 episodes）
            self.ep_r = 50
            # Algorithm 2: 每隔 C 回合同步一次 targetQ 和 kf_theta
            self.C_update = 10
            # Eq.(34): m = 0.35 + 0.6 * exp(-i)
            self.m_base = 0.35
            self.m_decay = 0.6
            self.m_tau = 100  # 严格复现：不使用 /m_tau
            # =========================
            # Fuzzy 学习率与动作强度（工程上常设小一些更稳）
            # =========================
            self.freeze_fuzzy_premise = True
            self.fuzzy_lr = 0.002
            if "MountainCar" in self.env_name:
                self.h1 = 0.4
                self.h2 = 0.6
                self.ep_r = 50
                self.m_base = 0.8
                self.m_decay = 0.2
