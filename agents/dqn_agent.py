import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.networks import QNet
from utils.exploration import get_linear_decay_epsilon

class DQNAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.device = cfg.device
        
        # 初始化 Q 网络和目标网络
        self.q_net = QNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_q_net = QNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.update_steps = 0
        self.epsilon = cfg.epsilon_start
    
    def take_action(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            return self.q_net(state_tensor).argmax().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # 当前 Q 值
        q_values = self.q_net(states).gather(1, actions)
        # 目标 Q 值 (公式 8: r + gamma * max Q(s'))
        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.cfg.gamma * max_next_q_values * (1 - dones)

        # Loss 计算
        loss = torch.mean(F.mse_loss(q_values, q_targets))

        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_steps += 1
        if self.update_steps % self.cfg.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
            
        return loss.item()

    def update_epsilon(self, episode_idx):
        self.epsilon = get_linear_decay_epsilon(episode_idx, self.cfg)