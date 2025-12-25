import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Q 网络 (DQN) - [已适配多层]
class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self._use_list = isinstance(hidden_dim, (list, tuple))
        
        if self._use_list:
            # 动态构建多层
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
            # 原始单层逻辑
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if self._use_list:
            x = self.feature(x)
            return self.fc_out(x)
        else:
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    
# 2. Dueling Q 网络 - [已适配多层]
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
            
            # 接入通过多层后的特征
            self.fc_v = nn.Linear(in_dim, 1)
            self.fc_a = nn.Linear(in_dim, action_dim)
        else:
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            
            self.fc_v = nn.Linear(hidden_dim, 1)
            self.fc_a = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if self._use_list:
            x = self.feature(x)
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        
        v = self.fc_v(x)
        a = self.fc_a(x)
        return v + (a - a.mean(dim=1, keepdim=True))
    
# 3. 策略网络 (PolicyNet) - [本次修改：适配多层]
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self._use_list = isinstance(hidden_dim, (list, tuple))
        
        if self._use_list:
            # --- 新增：列表处理逻辑 ---
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
            # --- 原始：单层逻辑 ---
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if self._use_list:
            x = self.feature(x)
            # 注意：PolicyNet 输出需要 Softmax
            return F.softmax(self.fc_out(x), dim=1)
        else:
            x = F.relu(self.fc1(x))
            return F.softmax(self.fc2(x), dim=1)

# 4. 价值网络 (ValueNet) - [本次修改：适配多层]
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self._use_list = isinstance(hidden_dim, (list, tuple))
        
        if self._use_list:
            # --- 新增：列表处理逻辑 ---
            hidden_sizes = list(hidden_dim)
            layers = []
            in_dim = state_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.feature = nn.Sequential(*layers)
            self.fc_out = nn.Linear(in_dim, 1) # 输出维度为 1
        else:
            # --- 原始：单层逻辑 ---
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self._use_list:
            x = self.feature(x)
            return self.fc_out(x)
        else:
            x = F.relu(self.fc1(x))
            return self.fc2(x)