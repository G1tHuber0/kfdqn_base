import math
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.dqn_agent import DQNAgent
from models.fuzzy_system import FuzzySystem
from utils.exploration import get_linear_decay_epsilon


class KFDQNAgent(DQNAgent):
    """
    符合论文原文的 KFDQN 实现 (针对 CartPole):
    - HYAS (算法 1): 使用模糊动作 a_f 进行探索；早期回合强制使用 a_f；后期使用混合动作。
    - 双模糊系统 (章节 4.2.2): kf_theta (指导/Guide) 和 kf_theta_minus (学习/Learn) 分离，避免训练不稳定。
    - 两阶段 Q 学习 (算法 2):
        * episode < ep_r (ept): 监督学习损失 (公式 19)，用于模仿模糊系统的指导。
        * episode >= ep_r: 混合 TD 目标 (公式 18)。
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # --- 模糊系统初始化 ---
        # kf_theta: 指导模糊系统 (用于动作选择 & 生成标签) - 对应论文中的 Mentor
        self.fuzzy_guide = FuzzySystem(self.device).to(self.device)
        # kf_theta_minus: 学习模糊系统 (通过 Replay Buffer 的数据进行交叉熵更新) - 对应论文中的 Student
        self.fuzzy_learn = FuzzySystem(self.device).to(self.device)
        # 初始化时，让学习网络与指导网络参数同步
        self.fuzzy_learn.load_state_dict(self.fuzzy_guide.state_dict())

        # 如果遵循通常的模糊系统解释，只更新规则权重（后件参数），
        # 而冻结隶属度参数（中心/宽度，即前件参数）。
        freeze_premise = getattr(cfg, "freeze_fuzzy_premise", True)
        if freeze_premise:
            # 冻结指导网络的前件参数
            for name, p in self.fuzzy_guide.named_parameters():
                if "rule_weights" not in name:
                    p.requires_grad_(False)
            # 冻结学习网络的前件参数
            for name, p in self.fuzzy_learn.named_parameters():
                if "rule_weights" not in name:
                    p.requires_grad_(False)

        # 设置模糊系统的优化器
        fuzzy_lr = getattr(cfg, "fuzzy_lr", cfg.lr)
        self.fuzzy_optimizer = optim.Adam(
            [p for p in self.fuzzy_learn.parameters() if p.requires_grad],
            lr=fuzzy_lr
        )

        # --- 混合目标权重 ---
        self.m = 1.0 # DQN 权重
        self.n = 0.0 # Fuzzy 权重

        # 内部计数器
        self._episode_idx = 0

    def _hard_update_targets(self):
        """硬更新目标 Q 网络和模糊指导系统 (算法 2)。"""
        # 更新 Target Q Network
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        # 将"学习好的模糊参数"复制给"指导模糊系统"
        self.fuzzy_guide.load_state_dict(self.fuzzy_learn.state_dict())

    def update_parameters(self, episode_idx: int):
        """每回合调用一次：更新 epsilon，更新 m/n 权重，以及执行周期性硬更新。"""
        self._episode_idx = episode_idx

        # 更新 epsilon (论文中使用 HYAS 来避免早期的纯随机探索)
        self.epsilon = get_linear_decay_epsilon(episode_idx, self.cfg)

        # 公式 (34): m = 0.35 + 0.6 * exp(-i)
        # 如果希望衰减得慢一点，可以在 config 设置 m_tau，计算 exp(-i/m_tau)
        m_tau = getattr(self.cfg, "m_tau", None)
        if m_tau is None:
            expo = -float(episode_idx)
        else:
            expo = -float(episode_idx) / float(m_tau)

        # 计算动态权重 m 和 n
        self.m = float(self.cfg.m_base + self.cfg.m_decay * math.exp(expo))
        self.m = max(0.0, min(1.0, self.m)) # 确保在 [0, 1] 之间
        self.n = 1.0 - self.m

        # 算法 2: 每隔 C 回合更新一次 Target 网络
        C = getattr(self.cfg, "C_update", 10)
        if episode_idx > 0 and (episode_idx % C == 0):
            self._hard_update_targets()

    @torch.no_grad()
    def take_action(self, state, episode_idx: Optional[int] = None) -> int:
        """HYAS 混合动作选择策略 (算法 1)。"""
        if episode_idx is None:
            episode_idx = self._episode_idx

        state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)

        # 获取 Q 值和模糊系统输出
        q_values = self.q_net(state_t)
        fuzzy_logits = self.fuzzy_guide(state_t)

        # 模糊系统的推荐动作
        a_f = int(fuzzy_logits.argmax(dim=1).item())

        # 算法 1 逻辑: 
        # 如果 p < epsilon (探索) -> 使用 a_f (知识引导探索)
        # 或者 如果 episode < ept (早期阶段) -> 强制使用 a_f
        # 否则 -> 使用混合动作
        if (np.random.rand() < self.epsilon) or (episode_idx < self.cfg.ep_r):
            return a_f

        # 混合动作 (公式 16): argmax(h1 * softmax(kf) + h2 * softmax(Q))
        f_score = F.softmax(fuzzy_logits, dim=1)
        q_score = F.softmax(q_values, dim=1)
        hybrid_score = self.cfg.h1 * f_score + self.cfg.h2 * q_score
        return int(hybrid_score.argmax(dim=1).item())

    def update(self, transition_dict: Dict[str, Any], episode_idx: Optional[int] = None) -> Dict[str, float]:
        """
        更新网络参数。
        返回 loss 字典用于日志记录: {'q_loss':..., 'fuzzy_loss':...}
        """
        if episode_idx is None:
            episode_idx = self._episode_idx

        # 数据准备
        states = torch.tensor(transition_dict["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32, device=self.device).view(-1, 1)

        # ========= 第一部分: Q 网络更新 =========
        if episode_idx < self.cfg.ep_r:
            # 阶段 1：监督学习 (公式 19)
            # 此时我们不信任 Q 值，而是让 Q 网络去模仿模糊系统的输出
            with torch.no_grad():
                a_f_labels = self.fuzzy_guide(states).argmax(dim=1)  # [B]
            q_logits = self.q_net(states)  # [B, A]
            q_loss = F.cross_entropy(q_logits, a_f_labels)
        else:
            # 阶段 2：混合 TD 学习 (公式 18)
            q_sa = self.q_net(states).gather(1, actions)

            with torch.no_grad():
                # DQN 部分的目标值: max Q_target
                max_next = self.target_q_net(next_states).max(dim=1)[0].view(-1, 1)

                # Fuzzy 部分的目标值: Q(s', a_f)
                # 获取指导模糊系统对下一状态的推荐动作 a_f(s')
                a_f_next = self.fuzzy_guide(next_states).argmax(dim=1).view(-1, 1)

                # 重要细节: 论文此处通常使用 Online Q Network 来评估模糊动作的价值，
                # 而上面的 max 项使用的是 Target Q Network。
                q_fuzzy_next = self.q_net(next_states).gather(1, a_f_next)

                # 混合目标值: m * DQN目标 + n * Fuzzy目标
                hybrid_next = self.m * max_next + self.n * q_fuzzy_next
                q_target = rewards + self.cfg.gamma * hybrid_next * (1.0 - dones)

            q_loss = F.mse_loss(q_sa, q_target)

        # 反向传播更新 Q 网络
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # ========= 第二部分: 知识更新 (公式 13) =========
        # 利用 Replay Buffer 中的真实状态-动作数据来更新“学习模糊系统”
        # 这是一种自模仿或行为克隆，使模糊系统适应实际产生的有效策略
        fuzzy_logits_learn = self.fuzzy_learn(states)
        fuzzy_loss = F.cross_entropy(fuzzy_logits_learn, actions.squeeze(1))
        
        self.fuzzy_optimizer.zero_grad()
        fuzzy_loss.backward()
        self.fuzzy_optimizer.step()

        return {"q_loss": float(q_loss.item()), "fuzzy_loss": float(fuzzy_loss.item())}