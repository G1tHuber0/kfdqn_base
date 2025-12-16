import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.networks import PolicyNet

class ReinforceAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.policy_net = PolicyNet(cfg.state_dim, cfg.hidden_dim, cfg.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.epsilon = 0.0 # 占位，REINFORCE 不需要 epsilon

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        # 根据概率采样
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update_epsilon(self, i):
        pass

    def update(self, transition_dict):
        # REINFORCE 是回合更新，这里假设 transition_dict 包含了一个完整回合的数据
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        
        # 逆序计算回报 G_t
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor(np.array([state_list[i]]), dtype=torch.float).to(self.device)
            action = torch.tensor(np.array([action_list[i]]), dtype=torch.long).to(self.device)
            
            G = self.cfg.gamma * G + reward
            
            # 计算 log_prob
            probs = self.policy_net(state)
            action_dist = torch.distributions.Categorical(probs)
            log_prob = action_dist.log_prob(action)
            
            # Loss = -log_prob * G
            loss = -log_prob * G
            loss.backward() # 累积梯度
            
        self.optimizer.step()
        return loss.item() # 返回最后一步的 loss 仅作记录