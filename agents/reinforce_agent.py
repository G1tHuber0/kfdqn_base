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
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        self.optimizer.zero_grad()
        
        # --- 第一步：计算并收集所有的 G_t ---
        G = 0
        G_list = []
        # 逆序计算，注意我们要把计算出来的 G 存起来
        for r in reversed(reward_list):
            G = self.cfg.gamma * G + r
            G_list.insert(0, G) # 插入到最前面，恢复正序

        # 转换为 Tensor
        G_tensor = torch.tensor(G_list, dtype=torch.float).to(self.device)

        # 回报处理
        if len(G_tensor) > 1: # 防止单步回合导致方差为0
            # G_tensor = (G_tensor - G_tensor.mean()) / (G_tensor.std() + 1e-9)
            G_tensor = G_tensor - G_tensor.mean()
        
        # --- 第二步：批量计算 Loss ---
        # 我们可以把 state_list 堆叠起来一次性计算，效率更高
        
        # 1. 转换数据
        states = torch.tensor(np.array(state_list), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(action_list), dtype=torch.long).view(-1, 1).to(self.device)
        
        # 2. 前向传播 (一次性算出所有步骤的概率)
        probs = self.policy_net(states)
        action_dist = torch.distributions.Categorical(probs)
        log_probs = action_dist.log_prob(actions.squeeze())
        
        # 3. 计算 Loss (Vectorized)
        # loss = - sum( log_prob * Normalized_G )
        loss = -torch.sum(log_probs * G_tensor)
        
        # 4. 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item() / len(action_list)