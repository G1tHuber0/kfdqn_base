import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Q 网络 (DQN)
class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        # Table 4: Input:4 -> hidden:128 -> Output:2
        self._use_list = isinstance(hidden_dim, (list, tuple))
        if self._use_list:
            hidden_sizes = list(hidden_dim)
            layers = []
            in_dim = state_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.feature = nn.Sequential(*layers)
            self.fc_out = nn.Linear(in_dim, action_dim)
        else:
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if self._use_list:
            x = self.feature(x)
            return self.fc_out(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
# 2. Dueling Q 网络
class DuelingQNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DuelingQNet, self).__init__()
        self._use_list = isinstance(hidden_dim, (list, tuple))
        if self._use_list:
            hidden_sizes = list(hidden_dim)
            layers = []
            in_dim = state_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.feature = nn.Sequential(*layers)
            # 状态价值流 V(s)
            self.fc_v = nn.Linear(in_dim, 1)
            # 动作优势流 A(s, a)
            self.fc_a = nn.Linear(in_dim, action_dim)
        else:
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            
            # 状态价值流 V(s)
            self.fc_v = nn.Linear(hidden_dim, 1)
            # 动作优势流 A(s, a)
            self.fc_a = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if self._use_list:
            x = self.feature(x)
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        
        v = self.fc_v(x)
        a = self.fc_a(x)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return v + (a - a.mean(dim=1, keepdim=True))
    
# 3. 策略网络 (用于 REINFORCE, AC-Actor)
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

# 4. 价值网络 (用于 AC-Critic)
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
