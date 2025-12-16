import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.networks import DuelingQNet # 引用不同网络
from utils.exploration import get_linear_decay_epsilon

class DuelingDQNAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.device = cfg.device
        
        # 使用 DuelingQNet
        self.q_net = DuelingQNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.target_q_net = DuelingQNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
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

        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.cfg.gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.cfg.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        return loss.item()