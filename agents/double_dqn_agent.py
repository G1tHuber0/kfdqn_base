import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.networks import QNet
from utils.exploration import get_linear_decay_epsilon

class DoubleDQNAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.device = cfg.device
        
        self.q_net = QNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_q_net = QNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.count = 0
        self.epsilon = cfg.epsilon_start

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            return self.q_net(state).argmax().item()

    def update_epsilon(self, episode_idx):
        self.epsilon = get_linear_decay_epsilon(episode_idx, self.cfg)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        # --- Double DQN 核心差异 ---
        with torch.no_grad():
            # 1. 使用【当前网络】确定最大动作
            max_action = self.q_net(next_states).argmax(1).view(-1, 1)
            # 2. 使用【目标网络】计算该动作的价值
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
            q_targets = rewards + self.cfg.gamma * max_next_q_values * (1 - dones)
        # -------------------------

        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.cfg.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        
        return loss.item()