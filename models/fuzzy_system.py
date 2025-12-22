# models/fuzzy_system.py

import torch
import torch.nn as nn
import itertools

class CartPoleFuzzyConfig:
    """
    配置类：CartPole-v0 (Strict Paper Alignment)
    适用环境: CartPole-v0 (需在 train.py 中设置 x_threshold=2.4)
    """
    # ==========================================
    # 1. 输入变量定义 (Antecedents) - 论文 Fig.4
    # ==========================================
    # 状态顺序: [CartPos, CartVel, PoleAngle, PoleVel]
    
    ANTECEDENT_CENTERS = [
        [-2.4, 2.4],    # cp: 对应环境 +-2.4 限制
        [-3.0, 3.0],    # cv: 小车速度
        [-1.75, 1.75],  # pd: 杆角度
        [-1.0, 1.0]     # pv: 杆角速度
    ]
    
    ANTECEDENT_SIGMAS = [
        [1.5, 1.5],     # cp: 论文原参数
        [1.775, 1.775], # cv
        [0.75, 0.75],   # pd
        [0.625, 0.625]  # pv
    ]

    # ==========================================
    # 2. 输出动作定义 (Consequents) - 论文 Fig.5
    # ==========================================
    # 物理含义: +1.0 代表"支持/推荐", -1.0 代表"反对/抑制"
    
    ACTION_SUPPORT = 1.0    # Big (Blue Curve)
    ACTION_OPPOSE = -1.0    # Small (Red Curve)
    
    # 预处理参数
    POS_LIMIT = 2.4         # 位置截断值
    ANGLE_LIMIT = 2.4       # 角度截断值
    ANGVEL_LIMIT = 3.0      # 角速度截断值


class MountainCarFuzzyConfig:
    """
    配置类：MountainCar-v0 (Strict Paper Alignment - Table 7)
    Inputs: Position (2 sets), Velocity (3 sets) -> Total 6 Rules
    """
    # ==========================================
    # 1. 输入变量定义 (Antecedents)
    # ==========================================
    # Position (2 sets): [Left Side, Right Side]
    # 论文 Fig 15 暗示左侧中心偏左(-1.0)，右侧中心偏右(0.5)，以谷底-0.5为界
    ANTECEDENT_CENTERS = [
        [-1.0, 0],       # Position centers
        [-0.04, 0.0, 0.04] # Velocity centers: [Left(Neg), Stop(0), Right(Pos)]
    ]
    
    # 模糊集宽度 (Sigmas)
    ANTECEDENT_SIGMAS = [
        [0.5, 0.5],        # Position sigma
        [0.02, 0.02, 0.02] # Velocity sigma (中间的 Stop 窄一点)
    ]

    # ==========================================
    # 2. 输出动作定义 (Consequents)
    # ==========================================
    # 动作: 0=Left, 1=None, 2=Right
    # 论文 Fig 16 动作模糊集中心为 -1, 0, 1
    # 考虑到 Softmax，建议放大这些值以增强引导
    ACTION_SUPPORT = 1.0   # 强力推荐 (对应 1.0)
    ACTION_OPPOSE = -1.0   # 强力抑制 (对应 -1.0)

    # 预处理截断
    POS_LIMIT = 1.2
    VEL_LIMIT = 0.07


class FuzzySystem(nn.Module):
    """
    KFDQN 模糊逻辑控制器
    核心功能: 将环境状态映射为动作推荐分数 (Logits)
    """
    def __init__(self, device, env_name: str = "CartPole-v0"):
        super(FuzzySystem, self).__init__()
        self.device = device
        self.env_name = env_name
        self.is_mountaincar = "MountainCar" in env_name

        if self.is_mountaincar:
            self.cfg_cls = MountainCarFuzzyConfig
            self.num_inputs = 2
            self.num_rules = 6  # MountainCar: 2 Pos * 3 Vel = 6 Rules
            self.action_dim = 3
        else:
            self.cfg_cls = CartPoleFuzzyConfig
            self.num_inputs = 4
            self.num_rules = 16 # CartPole: 2^4 = 16 Rules
            self.action_dim = 2
        
        # 1. 初始化模糊集参数
        self._init_fuzzy_sets()

        # 2. 定义缩放系数 (Preprocess Scaling)
        # 将 Gym 微小的数值放大，使其能落在模糊集的有效区间内
        if self.is_mountaincar:
            # Velocity range ~0.07, need scale up to match sigma ~0.02 distribution
            self.scales = torch.tensor([1.0, 1.0], device=device)
        else:
            self.scales = torch.tensor([1.0, 1.0, 2.4, 1/3], device=device)
        
        # 3. 初始化规则库权重
        self.rule_weights = nn.Parameter(torch.zeros(self.num_rules, self.action_dim).to(device))
        self._build_rule_base()

    def _init_fuzzy_sets(self):
        # 注意：MountainCar 的 Centers/Sigmas 维度是不规则的 (2 vs 3)，
        # 所以对于 MountainCar 我们分别存储，对于 CartPole 我们统一存储。
        
        if self.is_mountaincar:
            # 分别注册，方便 forward 中处理不同维度
            self.pos_centers = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_CENTERS[0], device=self.device))
            self.pos_sigmas = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_SIGMAS[0], device=self.device))
            self.vel_centers = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_CENTERS[1], device=self.device))
            self.vel_sigmas = nn.Parameter(torch.tensor(self.cfg_cls.ANTECEDENT_SIGMAS[1], device=self.device))
            
            # 占位符，防止调用出错 (CartPole logic won't use these)
            self.centers = None 
            self.sigmas = None
        else:
            self.centers = nn.Parameter(
                torch.tensor(self.cfg_cls.ANTECEDENT_CENTERS, dtype=torch.float32).to(self.device)
            )
            self.sigmas = nn.Parameter(
                torch.tensor(self.cfg_cls.ANTECEDENT_SIGMAS, dtype=torch.float32).to(self.device)
            )

    def preprocess(self, state):
        """
        数据预处理流水线: Scaling -> Clamping
        """
        # 1. 缩放
        scaled_state = state * self.scales
        
        # 2. 截断 (避免数值越界导致 Gaussian 输出为 0)
        processed = scaled_state.clone()
        if self.is_mountaincar:
            processed[:, 0] = torch.clamp(processed[:, 0], -self.cfg_cls.POS_LIMIT, self.cfg_cls.POS_LIMIT)
            processed[:, 1] = torch.clamp(processed[:, 1], -self.cfg_cls.VEL_LIMIT, self.cfg_cls.VEL_LIMIT)
        else:
            processed[:, 0] = torch.clamp(processed[:, 0], -self.cfg_cls.POS_LIMIT, self.cfg_cls.POS_LIMIT)
            processed[:, 2] = torch.clamp(processed[:, 2], -self.cfg_cls.ANGLE_LIMIT, self.cfg_cls.ANGLE_LIMIT)
            processed[:, 3] = torch.clamp(processed[:, 3], -self.cfg_cls.ANGVEL_LIMIT, self.cfg_cls.ANGVEL_LIMIT)
        
        return processed

    def _build_rule_base(self):
        # ==========================================
        # Branch 1: MountainCar (6 Rules, Table 7)
        # ==========================================
        if self.is_mountaincar:
            SUPPORT = self.cfg_cls.ACTION_SUPPORT
            OPPOSE = self.cfg_cls.ACTION_OPPOSE
            
            # 初始化所有权重为 OPPOSE (抑制)
            nn.init.constant_(self.rule_weights, OPPOSE)

            # 辅助函数: 设置某条规则的推荐动作
            # pos_i: 0=Left Side, 1=Right Side
            # vel_i: 0=Left(Neg), 1=Stop, 2=Right(Pos)
            # action: 0=Left, 1=None, 2=Right
            def set_rule(pos_i, vel_i, action):
                rule_idx = pos_i * 3 + vel_i # Flatten index for 2x3 grid
                self.rule_weights[rule_idx, action] = SUPPORT

            with torch.no_grad():
                # Rule 1: Left Side + Moving Left -> Push Left (0)
                set_rule(0, 0, 0)
                # Rule 2: Left Side + Stop -> Push Right (2) (Key: Gain Momentum)
                set_rule(0, 1, 2)
                # Rule 3: Left Side + Moving Right -> Push Right (2)
                set_rule(0, 2, 2)
                
                # Rule 4: Right Side + Moving Left -> Push Left (0)
                set_rule(1, 0, 0)
                # Rule 5: Right Side + Stop -> Push Left (0) (Key: Gain Momentum)
                set_rule(1, 1, 0)
                # Rule 6: Right Side + Moving Right -> Push Right (2)
                set_rule(1, 2, 2)
            return

        # ==========================================
        # Branch 2: CartPole (16 Rules, Table 3 Extended)
        # ==========================================
        SUPPORT = self.cfg_cls.ACTION_SUPPORT
        OPPOSE = self.cfg_cls.ACTION_OPPOSE

        combinations = list(itertools.product([0, 1], repeat=4)) # (cp, cv, pd, pv)

        # (pd, pv) -> action mapping (0=left, 1=right)
        action_map = {
            (0, 0): 0,  # left,  left  -> move left
            (0, 1): 1,  # left,  right -> move right
            (1, 1): 1,  # right, right -> move right
            (1, 0): 0,  # right, left  -> move left
        }

        with torch.no_grad():
            for i, (cp, cv, pd, pv) in enumerate(combinations):
                a = action_map[(pd, pv)]  # Determine action based on Pole

                self.rule_weights[i, 0] = SUPPORT if a == 0 else OPPOSE
                self.rule_weights[i, 1] = SUPPORT if a == 1 else OPPOSE

    def gaussian(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def forward(self, state):
        # 0. 维度与预处理
        if state.dim() == 1:
            state = state.unsqueeze(0)
        batch_size = state.shape[0] 
        x_in = self.preprocess(state)
        x = x_in.unsqueeze(2)

        # ==========================================
        # Branch 1: MountainCar Inference (2 Pos sets x 3 Vel sets)
        # ==========================================
        if self.is_mountaincar:
            # 分别提取位置和速度
            pos = x[:, 0, :] # [B, 1]
            vel = x[:, 1, :] # [B, 1]
            
            # 计算隶属度
            mu_pos = self.gaussian(pos, self.pos_centers, self.pos_sigmas) # [B, 2]
            mu_vel = self.gaussian(vel, self.vel_centers, self.vel_sigmas) # [B, 3]
            
            # 规则推理: 外积 [B, 2, 1] * [B, 1, 3] -> [B, 2, 3]
            firing = torch.bmm(mu_pos.unsqueeze(2), mu_vel.unsqueeze(1))
            firing = firing.view(batch_size, -1) # Flatten to [B, 6]
            
        # ==========================================
        # Branch 2: CartPole Inference (Standard 2x2x2x2)
        # ==========================================
        else:
            # 统一计算高斯隶属度 [B, 4, 2]
            mu = self.gaussian(x, self.centers, self.sigmas)
            
            m_cp, m_cv = mu[:, 0, :], mu[:, 1, :]
            m_pd, m_pv = mu[:, 2, :], mu[:, 3, :]   
            
            # 两两组合
            layer1 = torch.bmm(m_cp.unsqueeze(2), m_cv.unsqueeze(1)).view(batch_size, -1) # [B, 4]
            layer2 = torch.bmm(m_pd.unsqueeze(2), m_pv.unsqueeze(1)).view(batch_size, -1) # [B, 4]
            
            # 最终 16 条规则激活度
            firing = torch.bmm(layer1.unsqueeze(2), layer2.unsqueeze(1)).view(batch_size, -1)

        # 3. 归一化 (Normalization)
        norm = firing / (torch.sum(firing, dim=1, keepdim=True) + 1e-6)
        
        # 4. 解模糊 (Defuzzification)
        # MountainCar: [B, 6] x [6, 3] -> [B, 3]
        # CartPole:    [B, 16] x [16, 2] -> [B, 2]
        output = torch.matmul(norm, self.rule_weights)
        
        return output