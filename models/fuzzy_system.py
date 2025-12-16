import torch
import torch.nn as nn
import itertools


class FuzzySystem(nn.Module):
    """
    语义知识模糊系统（CartPole-v0，对齐论文写法）
    - 输入状态: s = [cp, cv, pd, pv] = [CartPos, CartVel, PoleAngle, PoleVel]
    - 每个输入变量 2 个高斯模糊集: Left / Right
    - 规则数: 2^4 = 16（四维全排列）
    - 输出: 对离散动作空间 A={PushLeft, PushRight} 的“分数/logits”，覆盖整个动作空间

    关键点：
    1) Fig.4 的前件参数（centers/sigmas）固定（buffer），不训练。
    2) A-method 标定得到的 theta_scale/theta_dot_scale + clip 在 preprocess() 中完成，
       保证任何调用 forward() 的地方输入尺度一致。
    3) 16 条语义规则用 if/elif 逐条写死（你要求的直观 IF-THEN）。
    """

    def __init__(self, device, action_scale: float = 5.0):
        super().__init__()
        self.device = device
        self.action_scale = float(action_scale)

        # ====================================================
        # 1) Fig.4 模糊集参数（固定）
        # ====================================================
        # 每维两个模糊集: [Left, Right]
        mus_data = [
            [-2.4, 2.4],    # cp
            [-3.0, 3.0],    # cv
            [-1.75, 1.75],  # pd
            [-1.0, 1.0],    # pv
        ]
        sigmas_data = [
            [1.5, 1.5],        # cp
            [1.775, 1.775],    # cv
            [0.75, 0.75],      # pd
            [0.625, 0.625],    # pv
        ]
        self.register_buffer("centers", torch.tensor(mus_data, dtype=torch.float32, device=device))
        self.register_buffer("sigmas", torch.tensor(sigmas_data, dtype=torch.float32, device=device))

        # ====================================================
        # 2) A-method 标定缩放 + clip（训练前由 agent 写入）
        # ====================================================
        self.register_buffer("theta_scale", torch.tensor(1.0, dtype=torch.float32, device=device))
        self.register_buffer("theta_dot_scale", torch.tensor(1.0, dtype=torch.float32, device=device))

        self.register_buffer("cp_clip", torch.tensor(2.4, dtype=torch.float32, device=device))
        self.register_buffer("cv_clip", torch.tensor(3.0, dtype=torch.float32, device=device))
        self.register_buffer("pd_clip", torch.tensor(2.25, dtype=torch.float32, device=device))
        self.register_buffer("pv_clip", torch.tensor(2.0, dtype=torch.float32, device=device))

        # ====================================================
        # 3) 后件参数（可训练）：16 条规则 × 2 个动作分数（logits）
        # ====================================================
        # rule_weights[k] = 第 k 条规则对动作 [A0=PushLeft, A1=PushRight] 的 logits
        self.rule_weights = nn.Parameter(torch.zeros(16, 2, dtype=torch.float32, device=device))

        # 用 if/elif 逐条写 16 条 IF-THEN 初始化
        self._init_rule_weights_by_if()

    def set_scaler(
        self,
        theta_scale: float,
        theta_dot_scale: float,
        fuzzy_cv_clip: float = 3.0,
        fuzzy_pd_clip: float = 2.25,
        fuzzy_pv_clip: float = 2.0,
        fuzzy_cp_clip: float = 2.4,
    ):
        """写入 A-method 标定参数（guide 和 learn 两个模糊系统都必须调用）。"""
        self.theta_scale.fill_(float(theta_scale))
        self.theta_dot_scale.fill_(float(theta_dot_scale))
        self.cv_clip.fill_(float(fuzzy_cv_clip))
        self.pd_clip.fill_(float(fuzzy_pd_clip))
        self.pv_clip.fill_(float(fuzzy_pv_clip))
        self.cp_clip.fill_(float(fuzzy_cp_clip))

    def preprocess(self, state: torch.Tensor) -> torch.Tensor:
        """
        raw state -> Fig.4 对齐尺度（A-method）
        state: [B,4] or [4]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        s = state.to(self.device, dtype=torch.float32)

        cp = s[:, 0].clamp(-self.cp_clip.item(), self.cp_clip.item())
        cv = s[:, 1].clamp(-self.cv_clip.item(), self.cv_clip.item())
        pd = (s[:, 2] * self.theta_scale).clamp(-self.pd_clip.item(), self.pd_clip.item())
        pv = (s[:, 3] * self.theta_dot_scale).clamp(-self.pv_clip.item(), self.pv_clip.item())

        return torch.stack([cp, cv, pd, pv], dim=1)

    @staticmethod
    def _gaussian(x, mu, sigma):
        """高斯隶属度函数：μ(x)=exp(-0.5*((x-mu)/sigma)^2)"""
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def _init_rule_weights_by_if(self):
        """
        16 条语义规则（if 语句直写）。

        规则枚举顺序：
          combos = itertools.product([0,1], repeat=4)
          (cp, cv, pd, pv) 字典序，pv 变化最快
        其中 0=Left，1=Right

        论文语义（倒立摆）核心：
          - IF pole angle (pd) is Left  THEN push left
          - IF pole angle (pd) is Right THEN push right

        同时满足论文对规则输出的要求：每条规则对整个动作空间输出分数：
          - 推左：a0 Large, a1 Small
          - 推右：a0 Small, a1 Large
        用 logits ±S 表示 Large/Small，S=action_scale
        """
        L, R = 0, 1
        S = self.action_scale

        def set_push_left(i: int):
            self.rule_weights[i, 0] = +S  # A0=PushLeft Large
            self.rule_weights[i, 1] = -S  # A1=PushRight Small

        def set_push_right(i: int):
            self.rule_weights[i, 0] = -S  # A0 Small
            self.rule_weights[i, 1] = +S  # A1 Large

        combos = list(itertools.product([0, 1], repeat=4))

        with torch.no_grad():
            for i, (cp, cv, pd, pv) in enumerate(combos):

                # -------- pd=Left -> PushLeft (8条) --------
                if (cp == L and cv == L and pd == L and pv == L):
                    set_push_left(i)
                elif (cp == L and cv == L and pd == L and pv == R):
                    set_push_left(i)
                elif (cp == L and cv == R and pd == L and pv == L):
                    set_push_left(i)
                elif (cp == L and cv == R and pd == L and pv == R):
                    set_push_left(i)
                elif (cp == R and cv == L and pd == L and pv == L):
                    set_push_left(i)
                elif (cp == R and cv == L and pd == L and pv == R):
                    set_push_left(i)
                elif (cp == R and cv == R and pd == L and pv == L):
                    set_push_left(i)
                elif (cp == R and cv == R and pd == L and pv == R):
                    set_push_left(i)

                # -------- pd=Right -> PushRight (8条) --------
                elif (cp == L and cv == L and pd == R and pv == L):
                    set_push_right(i)
                elif (cp == L and cv == L and pd == R and pv == R):
                    set_push_right(i)
                elif (cp == L and cv == R and pd == R and pv == L):
                    set_push_right(i)
                elif (cp == L and cv == R and pd == R and pv == R):
                    set_push_right(i)
                elif (cp == R and cv == L and pd == R and pv == L):
                    set_push_right(i)
                elif (cp == R and cv == L and pd == R and pv == R):
                    set_push_right(i)
                elif (cp == R and cv == R and pd == R and pv == L):
                    set_push_right(i)
                elif (cp == R and cv == R and pd == R and pv == R):
                    set_push_right(i)
                else:
                    # 理论不会发生：16 全排列已覆盖
                    set_push_left(i)

    def forward(self, state: torch.Tensor, already_preprocessed: bool = False) -> torch.Tensor:
        """
        语义模糊系统输出：对两个动作的分数/logits（覆盖整个动作空间）
        """
        if not already_preprocessed:
            state = self.preprocess(state)

        B = state.shape[0]
        x = state.unsqueeze(2)  # [B,4,1]

        # fuzzification: [B,4,2]
        mu = self._gaussian(x, self.centers, self.sigmas)

        m0 = mu[:, 0, :]  # cp [B,2]
        m1 = mu[:, 1, :]  # cv
        m2 = mu[:, 2, :]  # pd
        m3 = mu[:, 3, :]  # pv

        # rule firing strength by product t-norm (乘积算子)
        layer1 = torch.bmm(m0.unsqueeze(2), m1.unsqueeze(1)).view(B, -1)  # [B,4]
        layer2 = torch.bmm(m2.unsqueeze(2), m3.unsqueeze(1)).view(B, -1)  # [B,4]
        firing = torch.bmm(layer1.unsqueeze(2), layer2.unsqueeze(1)).view(B, -1)  # [B,16]

        # normalize firing strengths
        norm = firing / (firing.sum(dim=1, keepdim=True) + 1e-6)

        # defuzzification: center-average style in vector form (论文写法 kf_theta(s)=theta*xi(s))
        out = norm @ self.rule_weights  # [B,2]
        return out
