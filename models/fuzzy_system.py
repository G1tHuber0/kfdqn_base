import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class FuzzySystem(nn.Module):
    def __init__(self, device):
        super(FuzzySystem, self).__init__()
        self.device = device
        
        # ====================================================
        # 1. 严格使用你提供的论文参数 (不可更改)
        # ====================================================
        # 状态顺序: [CartPos, CartVel, PoleAngle, PoleVel]
        
        # 你的参数:
        mus_data = [
            [-2.4, 2.4],    # Position (x)
            [-3.0, 3.0],    # Velocity (x_dot)
            [-1.75, 1.75],  # Angle (theta) - 论文值
            [-1.0, 1.0]     # AngVel (theta_dot)
        ]
        
        sigmas_data = [
            [1.5, 1.5],       # Position
            [1.775, 1.775],   # Velocity
            [0.75, 0.75],     # Angle
            [0.625, 0.625]    # AngVel
        ]

        self.centers = nn.Parameter(torch.tensor(mus_data, dtype=torch.float32).to(device))
        self.sigmas = nn.Parameter(torch.tensor(sigmas_data, dtype=torch.float32).to(device))

        # ====================================================
        # 2. 规则权重初始化 (Consequent Parameters)
        # ====================================================
        # 生成 16 条规则 (2^4)
        self.rule_weights = nn.Parameter(torch.zeros(16, 2).to(device))
        
        # 使用强化的物理逻辑初始化，以适配较宽的模糊集参数
        self._initialize_physics_knowledge()

    def _initialize_physics_knowledge(self):
        """
        初始化 16 条规则的权重。
        由于 Angle 的中心(1.75)远大于失效角度(0.2)，
        我们需要非常敏感的权重来捕捉微小的变化。
        """
        # 生成所有组合: 0=Left/Negative, 1=Right/Positive
        combinations = list(itertools.product([0, 1], repeat=4))
        
        with torch.no_grad():
            for i, (pos, vel, angle, ang_vel) in enumerate(combinations):
                # 将 0/1 映射为物理符号: 0 -> -1 (Left), 1 -> +1 (Right)
                s_pos = -1.0 if pos == 0 else 1.0
                s_vel = -1.0 if vel == 0 else 1.0
                s_ang = -1.0 if angle == 0 else 1.0
                s_ang_vel = -1.0 if ang_vel == 0 else 1.0
                
                # --- PD 控制打分逻辑 ---
                # 核心逻辑：Force ~ Kp * Angle + Kd * AngVel
                # 因为模糊集很宽，我们需要加大 AngVel 的权重来预测趋势
                
                # 权重分配：
                # AngVel (5.0): 预测趋势，防止倒塌的最关键因素
                # Angle (4.0): 当前倾斜程度
                # Pos (1.0): 稍微回中，但不能影响平衡
                
                # 计算“向右推”的必要性分值
                # 注意：Pos 的符号是反的。如果车在右边(s_pos=1)，我们希望往左推(-1)来回中。
                score = (4.0 * s_ang) + (5.0 * s_ang_vel) - (1.0 * s_pos) - (0.5 * s_vel)
                
                # --- 赋值权重 ---
                # 使用较大的 Scale (5.0) 来使得 Softmax 后的输出接近 One-hot
                # 这样即使隶属度差异很小，动作选择也会很果断
                scale = 5.0 
                
                if score > 0:
                    # 建议向右 (Action 1)
                    self.rule_weights[i, 0] = -1.0 * scale # 抑制 Left
                    self.rule_weights[i, 1] = 1.0 * scale  # 激活 Right
                else:
                    # 建议向左 (Action 0)
                    self.rule_weights[i, 0] = 1.0 * scale  # 激活 Left
                    self.rule_weights[i, 1] = -1.0 * scale # 抑制 Right

    def gaussian(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def forward(self, state):
        batch_size = state.shape[0]
        x = state.unsqueeze(2) 
        
        # 1. 模糊化 (使用固定参数)
        mu = self.gaussian(x, self.centers, self.sigmas)
        
        # 2. 全排列规则推理 (16 Rules)
        m0 = mu[:, 0, :] # Pos
        m1 = mu[:, 1, :] # Vel
        m2 = mu[:, 2, :] # Angle
        m3 = mu[:, 3, :] # AngVel
        
        # 计算所有组合的激活强度
        # [Batch, 4]
        layer1 = torch.bmm(m0.unsqueeze(2), m1.unsqueeze(1)).view(batch_size, -1) 
        layer2 = torch.bmm(m2.unsqueeze(2), m3.unsqueeze(1)).view(batch_size, -1)
        
        # [Batch, 16]
        firing_strengths = torch.bmm(layer1.unsqueeze(2), layer2.unsqueeze(1)).view(batch_size, -1)
        
        # 3. 归一化
        norm_strengths = firing_strengths / (torch.sum(firing_strengths, dim=1, keepdim=True) + 1e-6)
        
        # 4. 去模糊化
        # [Batch, 16] x [16, 2] -> [Batch, 2]
        output = torch.matmul(norm_strengths, self.rule_weights)
        
        return output