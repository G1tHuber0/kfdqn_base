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
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # # 自动生成实验路径
        # curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.exp_name = f"{self.env_name}_{self.algo}_{curr_time}"
        # self.result_path = os.path.join("results", self.algo)
        # self.log_path = os.path.join("results", "logs", self.algo, self.exp_name)
        
        # if not os.path.exists(self.result_path): os.makedirs(self.result_path)
        # if not os.path.exists(self.log_path): os.makedirs(self.log_path)

        # --- 2. 神经网络参数 ---
        self.state_dim = 4      
        self.action_dim = 2     
        self.hidden_dim = 128   
        
        # --- 3. 训练通用参数 ---
        self.gamma = 0.98       
        self.episodes = 500     
        self.lr = 0.002         
        
        # --- 4. 初始化特定算法参数 ---
        # 先给 KFDQN 特有参数赋默认值 None，防止报错
        self.h1 = None
        self.h2 = None
        self.m_base = None
        
        # 加载具体参数
        self._set_algo_specific_params()

    def _set_algo_specific_params(self):
        """根据不同算法调整参数"""
        
        # ==========================================
        # Group A: DQN 家族 (Off-Policy)
        # ==========================================
        if self.algo in ['DQN', 'Double', 'Dueling']:
            self.buffer_size = 10000 
            self.minimal_size = 500  
            self.batch_size = 64     
            self.target_update = 10
            
            # --- 探索参数 (统一命名) ---
            self.epsilon_start = 1.0  
            self.epsilon_end = 0.01   
            
            # [关键修改] DQN 也必须叫 decay_start/steps，否则 utils 报错
            self.decay_start = 50     # 对应原来的 episode_decay
            self.decay_steps = 100    # 对应原来的 epsilon_decay

        # ==========================================
        # Group B: Reinforce
        # ==========================================
        elif self.algo == 'Reinforce':
            self.buffer_size = None
            self.minimal_size = None
            self.batch_size = None
            self.target_update = None
            self.epsilon_start = None

        # ==========================================
        # Group C: Actor-Critic
        # ==========================================
        elif self.algo == 'AC':
            self.buffer_size = None 
            self.minimal_size = 0
            self.batch_size = 1
            self.target_update = None
            self.epsilon_start = None
            self.lr_actor = 0.0005
            self.lr_critic = 0.002

        # ==========================================
        # Group D: KFDQN (Knowledge Guided)
        # ==========================================
        elif self.algo == 'KFDQN':
            # =========================
            # Replay / Batch（论文 Table 4 同量级）
            # =========================
            self.buffer_size = 10000
            self.minimal_size = 500
            self.batch_size = 64
            # KFDQN 的 target 同步在 Agent 内用 C_update（episode级）实现
            # 为避免与 DQN 基类 step-based target_update 冲突，这里置 None（或不使用）
            self.target_update = None
            # 探索参数
            self.epsilon_start = 1.0
            self.epsilon_end = 0.01
            self.decay_start = 50
            self.decay_steps = 50
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
            self.m_tau = 1  # 严格复现：不使用 /m_tau

            # =========================
            # Fuzzy 学习率与动作强度（工程上常设小一些更稳）
            # =========================
            self.fuzzy_lr = 0.0002
            self.fuzzy_action_scale = 5.0
            # =========================
            # A-method scaler calibration / preprocessing
            # =========================
            self.fuzzy_calib_steps = 3000
            self.fuzzy_calib_q = 0.99
            # Fig.4 reference centers (用于分位数缩放对齐)
            self.fuzzy_pd_ref = 1.75
            self.fuzzy_pv_ref = 1.0
            # clip ranges（与你当前复现一致）
            self.fuzzy_cv_clip = 3.0
            self.fuzzy_pd_clip = 2.25
            self.fuzzy_pv_clip = 2.0
            self.fuzzy_cp_clip = 2.4