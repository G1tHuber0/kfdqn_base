import torch
import torch.nn as nn
import itertools

class FuzzyConfig:
    """
    配置类：严格对齐论文参数 (Strict Paper Alignment)
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
    # 基于数据拟合: Big中心在 1.0, Small中心在 -1.0
    # 物理含义: +1.0 代表"支持/推荐", -1.0 代表"反对/抑制"
    
    ACTION_SUPPORT = 1.0    # Big (Blue Curve)
    ACTION_OPPOSE = -1.0    # Small (Red Curve)
    
    # 预处理参数
    POS_LIMIT = 2.4         # 位置截断值
    ANGLE_LIMIT = 2.4       # 角度截断值
    ANGVEL_LIMIT = 3.0        # 角速度截断值

class FuzzySystem(nn.Module):
    """
    KFDQN 模糊逻辑控制器
    核心功能: 将环境状态映射为动作推荐分数 (Logits)
    """
    def __init__(self, device):
        super(FuzzySystem, self).__init__()
        self.device = device
        
        # 1. 初始化模糊集参数
        self._init_fuzzy_sets()

        # 2. 定义缩放系数 (Preprocess Scaling)
        #  将Gym 微小的弧度值放大，使其能落在模糊集的有效区间内
        self.scales = torch.tensor([1.0, 1.0, 2.4, 1/3], device=device)
        
        # 3. 初始化规则库权重
        self.rule_weights = nn.Parameter(torch.zeros(16, 2).to(device))
        self._build_rule_base()

    def _init_fuzzy_sets(self):
        self.centers = nn.Parameter(
            torch.tensor(FuzzyConfig.ANTECEDENT_CENTERS, dtype=torch.float32).to(self.device)
        )
        self.sigmas = nn.Parameter(
            torch.tensor(FuzzyConfig.ANTECEDENT_SIGMAS, dtype=torch.float32).to(self.device)
        )

    def preprocess(self, state):
        """
        数据预处理流水线:
        Raw State -> [Scaling] -> [Clamping] -> Fuzzy Input
        """
        # 1. 缩放 (主要针对 Angle 和 AngVel)
        scaled_state = state * self.scales
        
        # 2. 截断 (适配环境 2.4 的限制)
        # clone() 防止原地修改影响外部数据
        processed = scaled_state.clone()
        processed[:, 0] = torch.clamp(processed[:, 0], -FuzzyConfig.POS_LIMIT, FuzzyConfig.POS_LIMIT)
        processed[:, 2] = torch.clamp(processed[:, 2], -FuzzyConfig.ANGLE_LIMIT, FuzzyConfig.ANGLE_LIMIT)
        processed[:, 3] = torch.clamp(processed[:, 3], -FuzzyConfig.ANGVEL_LIMIT, FuzzyConfig.ANGVEL_LIMIT)
        
        return processed

    def _build_rule_base(self):
        """
        构建基于人类直觉的模糊规则库 (16条生存法则)
        输入状态: 
            cp (Cart Pos): 0=偏左, 1=偏右
            cv (Cart Vel): 0=向左, 1=向右
            pd (Pole Ang): 0=向左, 1=向右
            pv (Pole Vel): 0=往左倒, 1=往右倒
        
        输出动作:
            Action 0: 全力推左 (Force Left)
            Action 1: 全力推右 (Force Right)
        """
        # 生成所有可能的状态组合 (2^4 = 16种)
        # 顺序: CartPos, CartVel, PoleDeg, PoleVel
        combinations = list(itertools.product([0, 1], repeat=4))
        
        SUPPORT = FuzzyConfig.ACTION_SUPPORT # 建议做 (+1.0)
        OPPOSE = FuzzyConfig.ACTION_OPPOSE   # 强烈反对 (-1.0)

        with torch.no_grad():
            for i, (cp, cv, pd, pv) in enumerate(combinations):
                
                # 默认两个动作都反对，下面根据规则择优录取
                weight_left = OPPOSE  # Action 0
                weight_right = OPPOSE # Action 1
                # ==========================================
                # 阶段一：生存本能 (Pole Safety First)
                # ==========================================
                # [直觉 1] 杆子向左歪，且正在加速向左倒 -> 极度危险！
                # 不管车在哪，必须向左追，去接住杆子。
                if pd == 0 and pv == 0:
                    weight_left = SUPPORT  # 必须推左
                    weight_right = OPPOSE

                # [直觉 2] 杆子向右歪，且正在加速向右倒 -> 极度危险！
                # 必须向右追。
                elif pd == 1 and pv == 1:
                    weight_left = OPPOSE
                    weight_right = SUPPORT # 必须推右

                # ==========================================
                # 阶段二：精细微调 (Stabilization & Wall Avoidance)
                # ==========================================
                # 走到这里，说明 pd != pv，杆子正在往回摆 (Self-correcting)。
                # 这时候杆子暂时安全，我们把注意力转移到"车的位置"上。
                
                else: 
                    # --- 情况 A: 杆子向左歪(0)，但正在往右甩(1) ---
                    # 正常思路: 我们应该推右(Action 1)，帮杆子回正，顺便把车带回中间。
                    if pd == 0 and pv == 1:
                        # 但是！如果车已经在最右边(1)而且还在向右跑(1) -> 撞墙警报！
                        if cp == 1 and cv == 1:
                            weight_left = SUPPORT  # [反直觉] 必须推左刹车，保住车
                        else:
                            weight_right = SUPPORT # 正常情况：推右，帮杆子立起来

                    # --- 情况 B: 杆子向右歪(1)，但正在往左甩(0) ---
                    # 正常思路: 我们应该推左(Action 0)，帮杆子回正。
                    elif pd == 1 and pv == 0:
                        # 但是！如果车已经在最左边(0)而且还在向左跑(0) -> 撞墙警报！
                        if cp == 0 and cv == 0:
                            weight_right = SUPPORT # [反直觉] 必须推右刹车，保住车
                        else:
                            weight_left = SUPPORT  # 正常情况：推左，帮杆子立起来

                # ==========================================
                # 写入权重表
                # ==========================================
                self.rule_weights[i, 0] = weight_left
                self.rule_weights[i, 1] = weight_right
    def gaussian(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def forward(self, state):
        # 0. 维度与预处理
        if state.dim() == 1: state = state.unsqueeze(0)
        batch_size = state.shape[0] 
        # 缩放与截断
        x_in = self.preprocess(state)
        x = x_in.unsqueeze(2) # [B, 4, 1]
        # 1. 模糊化 (Fuzzification)
        mu = self.gaussian(x, self.centers, self.sigmas)
        # 2. 推理 (Inference)
        m_cp, m_cv = mu[:, 0, :], mu[:, 1, :]
        m_pd, m_pv = mu[:, 2, :], mu[:, 3, :]  
        layer1 = torch.bmm(m_cp.unsqueeze(2), m_cv.unsqueeze(1)).view(batch_size, -1)
        layer2 = torch.bmm(m_pd.unsqueeze(2), m_pv.unsqueeze(1)).view(batch_size, -1)
        # 计算 16 条规则的激活度
        firing = torch.bmm(layer1.unsqueeze(2), layer2.unsqueeze(1)).view(batch_size, -1)
        # 3. 归一化 (Normalization)
        norm = firing / (torch.sum(firing, dim=1, keepdim=True) + 1e-6)
        # 4. 解模糊 (Defuzzification)
        # Output Range: approx [-1.0, 1.0]
        output = torch.matmul(norm, self.rule_weights)
        return output