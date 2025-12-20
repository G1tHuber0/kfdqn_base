import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.networks import PolicyNet, ValueNet

class ACAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        
        self.actor = PolicyNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.critic = ValueNet(cfg.state_dim, cfg.hidden_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.lr_actor) # AC Actor 学习率通常低一些
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic) # Critic 学习率
        self.epsilon = 0.0

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update_epsilon(self, i):
        pass

    def update(self, transition_dict):
        # AC 即使是 On-policy，如果传入的是一个 batch，也可以进行 batch 更新
        # 这里为了兼容性，处理 batch 数据
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 1. 计算 TD Target
        td_target = rewards + self.cfg.gamma * self.critic(next_states) * (1 - dones)
        
        # 2. 计算 TD Error
        td_error = td_target - self.critic(states)
        
        # 3. Critic Loss (MSE)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        
        # 4. Actor Loss
        probs = self.actor(states)
        action_dist = torch.distributions.Categorical(probs)
        log_probs = action_dist.log_prob(actions.squeeze())
        
        # Policy Gradient: -log_prob * td_error
        actor_loss = torch.mean(-log_probs * td_error.detach().squeeze())
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        return critic_loss.item() # 记录 critic loss