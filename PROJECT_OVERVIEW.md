# KFDQN Base 项目说明

## 项目概览
- 针对 `CartPole-v0` 的强化学习实验仓库，涵盖 DQN 系列（基础/Double/Dueling）、REINFORCE、Actor-Critic 以及知识引导的 KFDQN。
- 每个算法对应独立的训练脚本与 Agent 类，TensorBoard 日志与训练曲线保存在 `results/`。
- 代码使用 `gymnasium`（部分兼容旧版 `gym`）、PyTorch、NumPy、matplotlib、tqdm；`sb3dqn.py` 额外依赖 `stable-baselines3` 作为对照基线。

## 目录结构（核心）
- `config.py`：统一的超参配置，`Config(algo=...)` 会按算法切换默认参数（buffer、batch、epsilon 线性衰减、学习率等），并为 KFDQN 提供模糊系统超参（h1/h2/m_base/C_update 等）。
- `agents/`  
  - `dqn_agent.py`：标准 DQN，目标网络软更新周期由 `target_update` 控制。  
  - `double_dqn_agent.py`：动作由当前网络选，价值由目标网络评估的 Double DQN。  
  - `dueling_dqn_agent.py`：使用 `DuelingQNet`，将价值/优势分支拼合。  
  - `kfdqn_agent.py`：KFDQN 核心，包含两阶段训练（前期监督学习模糊策略，后期混合 TD），HYAS 混合动作选择、动态权重 m/n、模糊系统双网络（guide/learn）与周期性硬更新。  
  - `reinforce_agent.py`：回合式 REINFORCE，批量计算 G_t 并标准化。  
  - `ac_agent.py`：简单 Actor-Critic（On-policy 批处理），分 actor/critic 优化器。  
  - `__init__.py`：空，占位。
- `models/`  
  - `networks.py`：`QNet`、`DuelingQNet`、`PolicyNet`、`ValueNet`。  
  - `fuzzy_system.py`：CartPole 先验模糊控制器（高斯隶属函数、16 条规则、支持/反对权重初始化、前件冻结选项）。
- `utils/`  
  - `replay_buffer.py`：简单 deque 经验回放。  
  - `exploration.py`：线性 epsilon 衰减逻辑（预热、衰减步数、下限保护）。
- 训练脚本  
  - `train_dqn.py`、`train_double_dqn.py`、`train_dueling_dqn.py`：三种 DQN 变体，统一记录日志/图片。  
  - `train_kfdqn.py`：KFDQN 训练入口，记录模糊相关指标（m、fuzzy_loss、HYAS 行为计数）。  
  - `train_reinforce.py`、`train_ac.py`：策略梯度 / Actor-Critic 入口。  
  - `sb3dqn.py`：Stable-Baselines3 DQN 对照实验。  
  - `0test.py`：CartPole 环境动作/状态空间与角度单位的快速自测。
- `results/`：运行后生成的 TensorBoard 日志与 PNG 曲线（按算法分类）。

## 运行示例
先安装依赖（如未创建虚拟环境，可参考下方依赖列表）：
```bash
pip install torch gymnasium numpy matplotlib tqdm tensorboard
# SB3 基线（如需运行 sb3dqn）：
pip install stable-baselines3
```

运行任一训练脚本（默认 CartPole-v0，500 回合）：
```bash
python train_dqn.py
python train_double_dqn.py
python train_dueling_dqn.py
python train_reinforce.py
python train_ac.py
python train_kfdqn.py
```

查看 TensorBoard 日志：
```bash
tensorboard --logdir results
```

绘图与日志会保存在 `results/<Algorithm>/<Algorithm>_<时间戳>/`，训练结束会输出一张 PNG。

## 关键行为与差异点
- **探索**：DQN 系列使用线性 epsilon 衰减（`decay_start/decay_steps`）；REINFORCE/AC 固定 0。  
- **KFDQN 特性**：  
  - 前 `ep_r` 回合监督模仿模糊策略；之后使用混合 TD 目标（m*maxQ + n*Q(s', a_f)）。  
  - HYAS 动作选择：前期强制模糊动作，后期按 h1/h2 加权的标准化 softmax 融合。  
  - 双模糊系统（guide/learn），学习网络周期性硬更新同步到指导网络；可冻结前件参数，仅训练规则权重。  
  - 动态权重 m 依赖 `m_base/m_decay/m_tau`，C 间隔硬更新 target Q 与模糊参数。
- **Actor-Critic**：按批大小或回合结束触发更新，使用 TD 目标与优势加权的策略梯度。

## 依赖清单（主要）
- Python 3.x
- PyTorch
- gymnasium（或兼容旧版 gym，`sb3dqn.py` 用 gym）
- numpy, matplotlib, tqdm, tensorboard
- stable-baselines3（仅运行 `sb3dqn.py` 时需要）

## 快速排查提示
- CartPole 阈值在各训练脚本中手动设置（位置 2.4、角度约 3.2°/6.4°），若更换环境需同步调整。  
- 训练初期若 loss 为 0，确认 `minimal_size` 与 `train_freq`，以及 epsilon 预热设置。  
- KFDQN 如出现数值异常，可尝试降低 `fuzzy_lr` 或关闭 `freeze_fuzzy_premise=False` 让前件参与训练。
