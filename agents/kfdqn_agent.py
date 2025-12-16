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
    贴合论文结构的 KFDQN (CartPole-v0)：
    - HYAS: 探索时用模糊动作 a_f；前 ep_r 回合强制用 a_f；之后用混合动作。
    - 双模糊系统: fuzzy_guide(kf_theta) / fuzzy_learn(kf_theta_minus)
    - 两阶段 Q 学习：
        * episode < ep_r：监督学习（CE）模仿 fuzzy_guide（Eq.19）
        * episode >= ep_r：混合 TD 目标（Eq.18）
    - 知识更新（Eq.13）：用 replay 中 (s,a) 对训练 fuzzy_learn
    - 周期性硬同步（Algorithm 2）：同时同步 target Q 与 fuzzy_guide
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # A-method 标定比例（由 calibrate_fuzzy_scaler 写入）
        self.theta_scale = 1.0
        self.theta_dot_scale = 1.0

        # 两套模糊系统
        self.fuzzy_guide = FuzzySystem(self.device, action_scale=getattr(cfg, "fuzzy_action_scale", 5.0)).to(self.device)
        self.fuzzy_learn = FuzzySystem(self.device, action_scale=getattr(cfg, "fuzzy_action_scale", 5.0)).to(self.device)
        self.fuzzy_learn.load_state_dict(self.fuzzy_guide.state_dict())

        # 只优化后件参数 rule_weights（符合“固定模糊集，学习规则参数”的复现思路）
        fuzzy_lr = getattr(cfg, "fuzzy_lr", cfg.lr)
        self.fuzzy_optimizer = optim.Adam([self.fuzzy_learn.rule_weights], lr=fuzzy_lr)

        # Eq.(34) 的混合权重
        self.m = 1.0  # DQN 部分权重
        self.n = 0.0  # Fuzzy 部分权重

        self._episode_idx = 0

    def _hard_update_targets(self):
        """Algorithm 2：硬同步 target Q 和 fuzzy_guide。"""
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.fuzzy_guide.load_state_dict(self.fuzzy_learn.state_dict())

    def update_parameters(self, episode_idx: int):
        """每回合更新：epsilon、m/n、以及是否硬同步。"""
        self._episode_idx = episode_idx

        # epsilon（论文未给清晰 schedule，你目前的线性退火可用）
        self.epsilon = get_linear_decay_epsilon(episode_idx, self.cfg)

        # Eq.(34): m = 0.35 + 0.6 * exp(-i)（严格复现默认不做 /m_tau）
        m_tau = getattr(self.cfg, "m_tau", None)
        expo = -float(episode_idx) if m_tau is None else (-float(episode_idx) / float(m_tau))
        self.m = float(self.cfg.m_base + self.cfg.m_decay * math.exp(expo))
        self.m = max(0.0, min(1.0, self.m))
        self.n = 1.0 - self.m

        # Algorithm 2：每隔 C_update 回合硬同步一次
        C = getattr(self.cfg, "C_update", 10)
        if episode_idx > 0 and (episode_idx % C == 0):
            self._hard_update_targets()

    @torch.no_grad()
    def take_action(self, state, episode_idx: Optional[int] = None) -> int:
        """HYAS 混合动作选择（Algorithm 1，Eq.16）。"""
        if episode_idx is None:
            episode_idx = self._episode_idx

        # 统一输入尺度：先用 fuzzy_guide.preprocess 做 Fig.4 尺度映射
        s_raw = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
        s = self.fuzzy_guide.preprocess(s_raw)

        q_values = self.q_net(s)
        fuzzy_logits = self.fuzzy_guide(s, already_preprocessed=True)

        a_f = int(fuzzy_logits.argmax(dim=1).item())

        # 探索 or 早期阶段：使用模糊动作 a_f
        if (np.random.rand() < self.epsilon) or (episode_idx < self.cfg.ep_r):
            return a_f

        # 混合动作：argmax(h1*softmax(kf) + h2*softmax(Q))
        f_score = F.softmax(fuzzy_logits, dim=1)
        q_score = F.softmax(q_values, dim=1)
        hybrid = self.cfg.h1 * f_score + self.cfg.h2 * q_score
        return int(hybrid.argmax(dim=1).item())

    def update(self, transition_dict: Dict[str, Any], episode_idx: Optional[int] = None) -> Dict[str, float]:
        """更新 Q 与 fuzzy_learn。"""
        if episode_idx is None:
            episode_idx = self._episode_idx

        states_raw = torch.tensor(transition_dict["states"], dtype=torch.float32, device=self.device)
        next_states_raw = torch.tensor(transition_dict["next_states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32, device=self.device).view(-1, 1)

        # 统一尺度：Q 和 fuzzy 都吃同一份 preprocess 后的状态
        states = self.fuzzy_guide.preprocess(states_raw)
        next_states = self.fuzzy_guide.preprocess(next_states_raw)

        # =========================
        # (1) Q 网络更新
        # =========================
        if episode_idx < self.cfg.ep_r:
            # 阶段1：监督学习（Eq.19）—— Q 模仿 fuzzy_guide 给的标签
            with torch.no_grad():
                a_f_labels = self.fuzzy_guide(states, already_preprocessed=True).argmax(dim=1)
            q_logits = self.q_net(states)
            q_loss = F.cross_entropy(q_logits, a_f_labels)
        else:
            # 阶段2：混合 TD（Eq.18）
            q_sa = self.q_net(states).gather(1, actions)

            with torch.no_grad():
                # DQN 部分：max_a Q_target(s',a)
                max_next = self.target_q_net(next_states).max(dim=1)[0].view(-1, 1)

                # Fuzzy 部分：Q_online(s', a_f(s'))
                a_f_next = self.fuzzy_guide(next_states, already_preprocessed=True).argmax(dim=1).view(-1, 1)
                q_fuzzy_next = self.q_net(next_states).gather(1, a_f_next)

                hybrid_next = self.m * max_next + self.n * q_fuzzy_next
                q_target = rewards + self.cfg.gamma * hybrid_next * (1.0 - dones)

            q_loss = F.mse_loss(q_sa, q_target)

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # =========================
        # (2) 知识更新（Eq.13）：训练 fuzzy_learn 去拟合 replay 中的 (s,a)
        # =========================
        fuzzy_logits_learn = self.fuzzy_learn(states, already_preprocessed=True)
        fuzzy_loss = F.cross_entropy(fuzzy_logits_learn, actions.squeeze(1))

        self.fuzzy_optimizer.zero_grad()
        fuzzy_loss.backward()
        self.fuzzy_optimizer.step()

        return {"q_loss": float(q_loss.item()), "fuzzy_loss": float(fuzzy_loss.item())}

    def calibrate_fuzzy_scaler(self, env, steps: int, q: float):
        """
        A-method：用随机交互采样 raw theta/theta_dot 的分位数，把它们映射到 Fig.4 的参考尺度。
        标定完成后写入 fuzzy_guide / fuzzy_learn，保证两套系统尺度一致。
        """
        thetas, theta_dots = [], []

        obs, _ = env.reset(seed=self.cfg.seed + 123)
        for _ in range(steps):
            a = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(a)
            thetas.append(abs(obs[2]))
            theta_dots.append(abs(obs[3]))
            if terminated or truncated:
                obs, _ = env.reset()

        q_theta = float(np.quantile(thetas, q))
        q_theta_dot = float(np.quantile(theta_dots, q))
        eps = 1e-6

        # 对齐到 Fig.4 的中心量级（pd_ref/pv_ref）
        self.theta_scale = float(self.cfg.fuzzy_pd_ref / max(q_theta, eps))
        self.theta_dot_scale = float(self.cfg.fuzzy_pv_ref / max(q_theta_dot, eps))

        # 写入两套模糊系统
        self.fuzzy_guide.set_scaler(
            theta_scale=self.theta_scale,
            theta_dot_scale=self.theta_dot_scale,
            fuzzy_cv_clip=self.cfg.fuzzy_cv_clip,
            fuzzy_pd_clip=self.cfg.fuzzy_pd_clip,
            fuzzy_pv_clip=self.cfg.fuzzy_pv_clip,
            fuzzy_cp_clip=getattr(self.cfg, "fuzzy_cp_clip", 2.4),
        )
        self.fuzzy_learn.set_scaler(
            theta_scale=self.theta_scale,
            theta_dot_scale=self.theta_dot_scale,
            fuzzy_cv_clip=self.cfg.fuzzy_cv_clip,
            fuzzy_pd_clip=self.cfg.fuzzy_pd_clip,
            fuzzy_pv_clip=self.cfg.fuzzy_pv_clip,
            fuzzy_cp_clip=getattr(self.cfg, "fuzzy_cp_clip", 2.4),
        )

        return {
            "q_theta": q_theta,
            "q_theta_dot": q_theta_dot,
            "theta_scale": self.theta_scale,
            "theta_dot_scale": self.theta_dot_scale,
        }
