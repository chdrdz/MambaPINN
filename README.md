[README.md](https://github.com/user-attachments/files/33521623/README.md)
# Mamba-PINN for Battery RUL/SOH Prediction

## 项目概述

本项目融合 **Mamba 状态空间模型**和 **Physics-Informed Neural Networks (PINN)** 用于电池剩余寿命 (RUL) 和健康状态 (SOH) 预测。

**数据集策略**：主实验在 MIT 数据集(124 块电池)上完成训练、验证和测试,后续在 NASA 和 CALCE 数据集上做跨数据集鲁棒性验证和少样本迁移实验,因此项目命名与代码结构保持数据集无关。

**核心创新**：
- Mamba 提供强大的序列建模能力，捕捉电池退化的长期依赖
- PINN 约束通过 Verhulst 退化方程提供物理可解释性
- AdpBal (Adaptive Balancing) 损失函数自动平衡数据驱动和物理约束
- 大规模数据集验证 + 跨数据集鲁棒性测试 + 少样本迁移学习

**目标**：发表 SCI-TOP 期刊论文（Journal of Power Sources / Applied Energy / IEEE Trans. on Industrial Electronics）

---

## 技术架构

### 整体框架

```
MIT 数据集 (124 块 LFP 电池，Severson et al.)
    ↓
容量退化曲线提取 + 归一化 (SOH = C / C_rated) + 滑动窗口构造
    ↓
输入：[SOH 序列] → Mamba Backbone → 预测：SOH
    ↓
损失函数 = loss_pred + λ_physics * loss_physics + λ_boundary * loss_boundary
    ↓
主实验：MIT 内部划分 (train 80 / valid 20 / test 24 块)
    ↓
跨数据集验证：MIT 训练 → NASA / CALCE 零样本或少样本迁移
    ↓
评价指标：RMSE, MAE, MAPE, R²
```

### 模型结构

#### 1. Mamba Backbone

```python
class MambaPINN(nn.Module):
    def __init__(self, d_model, n_layers, window_size):
        # 输入投影：每个 cycle 的 SOH 标量 → d_model
        self.input_proj = nn.Linear(1, d_model)
        
        # 官方 mamba-ssm 的 Mamba-1 Block，expand=4 保持与旧简易实现相同的内部展开容量。
        self.backbone = MambaBackbone(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            backend="official_mamba1",
            expand_factor=4,
        )
        
        # 输出投影：最后一个时间步 hidden + 归一化 cycle 时间 → SOH
        self.head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
```

**输入形状**：`(batch_size, window_size)`
- `batch_size`：样本数
- `window_size`：滑动窗口大小（例如 16 个 cycle）
- 模型内部会转换为 `(batch_size, window_size, 1)`，再投影到 `(batch_size, window_size, d_model)`

**输出形状**：`(batch_size, 1)`

**Mamba 核心原理**：
- 使用官方 `mamba-ssm` 的 Mamba-1 实现及其 Linux CUDA 选择性扫描路径
- 项目固定 `expand=4`，保持与旧简易实现一致的内部展开容量
- 输入输出契约保持为 `(batch, sequence_length, d_model)`
- 训练结果按 `results/mambapinn_official_mamba1_<dataset>_w<window>/` 保存，与旧简易实现目录物理隔离

#### 2. PINN 物理约束

**Verhulst 退化方程**：

```
dC/dt = r * (C - C_min) * (1 - (C - C_min) / (C_max - C_min))
```

其中：
- `C`：当前容量
- `r`：退化速率（可学习参数）
- `C_max`：初始最大容量（可学习参数）
- `C_min`：最小容量阈值（可学习参数）

**物理损失计算**：

```python
def physics_loss(model, inputs, predictions):
    # 开启自动求导
    inputs.requires_grad_(True)
    
    # 计算预测容量对时间的导数
    dC_dt = torch.autograd.grad(
        predictions, inputs,
        grad_outputs=torch.ones_like(predictions),
        create_graph=True
    )[0]
    
    # Verhulst 方程右端项
    G = r * (predictions - C_min) * (1 - (predictions - C_min) / (C_max - C_min))
    
    # 物理残差：dC/dt 应该等于 G
    physics_residual = dC_dt - G
    
    return torch.mean(physics_residual ** 2)
```

#### 3. 损失函数

**Baseline（纯数据驱动）**：
```
loss = MSE(pred, true)
```

**Sum（简单求和）**：
```
loss = loss_pred + loss_physics + loss_boundary
```

**AdpBal（自适应平衡，推荐）**：
```
loss = exp(-log_σ_u) * loss_pred 
     + exp(-log_σ_f) * loss_physics 
     + exp(-log_σ_b) * loss_boundary
     + log_σ_u + log_σ_f + log_σ_b
```

其中 `log_σ_u`, `log_σ_f`, `log_σ_b` 是可训练参数，自动学习三个损失项的相对权重。

---

## 数据处理

### 数据集

#### MIT 数据集（主实验）

**规模**：124 块 LFP 电池 (Severson et al., Nature Energy 2019)

**原始数据格式**：`.pkl` 或 `.mat` 文件
- `cycle`：循环编号
- `Qd`：每个 cycle 的放电容量
- `IR`：内阻
- 充电曲线特征 (电压、电流、温度等)

**数据划分策略**：

```python
# MIT 内部按电池编号划分 (或按容量分层抽样)
train_cells: 1-80    (80 块, 约 65%)
valid_cells: 81-100  (20 块, 约 16%)
test_cells: 101-124  (24 块, 约 19%)
```

**预处理**：
1. 提取容量退化序列
2. 归一化为 SOH: `SOH = Capacity / Rated_Capacity`
3. MIT 主实验默认 `Rated_Capacity = 1.1 Ah`，配置在 `config.yaml` 的 `rated_capacity.mit`
4. 滑动窗口构造训练样本 (window_size=16)

**最终指标**：在 test_cells 上的 RMSE / MAE / MAPE / R²

---

#### NASA 数据集（跨数据集鲁棒性验证）

**4 块电池**：B0005, B0006, B0007, B0018

**原始数据格式**：`.mat` 文件
- `cycle`：循环编号
- `Capacity`：每个 cycle 的放电容量
- `Voltage_measured`：电压曲线
- `Current_measured`：电流曲线
- `Temperature_measured`：温度

**验证策略**：

1. **零样本迁移**：MIT 训练的模型直接在 NASA 4 块电池上测试,不做任何微调
2. **少样本迁移**：
   - MIT 预训练 → 用 B0005/B0006 微调 → B0007/B0018 测试
   - 或用不同数量样本 (10/20/50 个 cycle) 微调,观察性能曲线

**容量配置**：NASA B0005/B0006/B0007/B0018 默认 `Rated_Capacity = 2.0 Ah`，配置在 `config.yaml` 的 `rated_capacity.nasa`。

**目的**：验证模型在小样本、不同化学体系 (NMC vs LFP) 下的泛化能力

---

#### CALCE 数据集（补充验证）

**规模**：多组不同工况电池 (CS2 系列等)

**验证策略**：MIT 训练 → CALCE 零样本测试

**容量配置**：CALCE 不应直接复用 MIT/NASA 的额定容量。当前 `config.yaml` 先保留 `rated_capacity.calce: 1.1` 作为占位，后续应按你实际下载的 CALCE 电池型号调整。

**目的**：验证模型对不同充放电倍率、温度条件的适应性

---

### 数据划分策略总结

**主实验 (MIT)**：
- 训练集：80 块电池 (65%)
- 验证集：20 块电池 (16%)
- 测试集：24 块电池 (19%)
- 评价指标：RMSE, MAE, MAPE, R² (在测试集上)

**跨数据集验证 (NASA / CALCE)**：
- 零样本：MIT 训练模型 → 直接在 NASA/CALCE 上测试
- 少样本：MIT 预训练 → NASA 微调 (10/20/50 样本) → 测试
- 评价指标：性能衰减率、迁移效率

---

#### MIT 与 CALCE 数据集（泛化性验证）

各数据集的额定容量、EOL 阈值不同,需在 `config.yaml` 中分别配置 `C_max`、`C_min` 等物理参数初始值。由于使用 SOH 归一化后物理参数统一,迁移时只需调整额定容量。

### 滑动窗口构造

```python
def build_sliding_windows(capacity_sequence, window_size=16):
    """
    输入：[C1, C2, C3, ..., C_n]
    输出：
        X: [[C1...C16], [C2...C17], ..., [C_{n-16}...C_{n-1}]]
        y: [C17, C18, ..., C_n]
    """
    X, y = [], []
    for i in range(len(capacity_sequence) - window_size):
        X.append(capacity_sequence[i:i+window_size])
        y.append(capacity_sequence[i+window_size])
    return np.array(X), np.array(y)
```

### 数据标准化

```python
# 对训练集计算统计量
mean_train = train_data.mean()
std_train = train_data.std()

# 所有 split 使用训练集统计量
train_norm = (train_data - mean_train) / std_train
valid_norm = (valid_data - mean_train) / std_train
test_norm = (test_data - mean_train) / std_train
```

---

## 实验设计

### 超参数配置

**模型超参**（MIT 大样本配置）：
```python
WINDOW_SIZE = 16          # 滑动窗口大小
D_MODEL = 64              # Mamba 隐藏维度(先用64,后续可消融128)
N_LAYERS = 2              # Mamba 层数
BATCH_SIZE = 64           # MIT样本多,batch可以大一些
EPOCHS = 500              # 训练轮数
LR = 0.001                # 学习率
WEIGHT_DECAY = 1e-5       # L2 正则化
```

**物理参数初始化**（归一化后通用值）：
```python
r = 0.01                  # 退化速率
C_max = 1.0               # 归一化后SOH最大值
C_min = 0.7               # EOL阈值(70% SOH)
```

### 训练流程

#### 主实验 (MIT)

1. **数据加载**：读取 MIT .pkl 文件，提取容量序列并归一化为 SOH
2. **数据划分**：按 80/10/10 划分 train/valid/test (80/20/24 块电池)
3. **窗口构造**：滑动窗口生成训练样本
4. **标准化**：基于训练集统计量归一化
5. **模型初始化**：Mamba + PINN 参数
6. **训练循环**：
   ```python
   for epoch in range(EPOCHS):
       for batch in train_loader:
           # 前向传播
           pred = model(batch_x)
           
           # 计算损失
           loss_pred = MSE(pred, batch_y)
           loss_phys = physics_loss(model, batch_x, pred)
           loss = adaptive_balance(loss_pred, loss_phys, log_sigma_u, log_sigma_f)
           
           # 反向传播
           loss.backward()
           optimizer.step()
       
       # 验证集评估
       valid_rmse = evaluate(model, valid_loader)
   ```
7. **测试评估**：在 test_cells 上计算 RMSE、MAE 等指标

#### 跨数据集验证 (NASA/CALCE)

**零样本迁移**：
```python
# 加载 MIT 训练好的模型
model.load_state_dict(torch.load('mit_best.pth'))

# 直接在 NASA 上测试
nasa_loader = load_nasa_data()
test_rmse = evaluate(model, nasa_loader)
```

**少样本微调**：
```python
# MIT 预训练模型
model.load_state_dict(torch.load('mit_best.pth'))

# 冻结 Mamba backbone，只微调输出层
for param in model.mamba.parameters():
    param.requires_grad = False

# 用 NASA 少量样本微调
finetune(model, nasa_train_loader, epochs=50)

# 在 NASA 测试集评估
test_rmse = evaluate(model, nasa_test_loader)
```

### 评价指标

```python
RMSE = sqrt(mean((pred - true)^2))
MAE = mean(|pred - true|)
MAPE = mean(|pred - true| / true) * 100
R² = 1 - sum((pred - true)^2) / sum((true - mean(true))^2)
RE = |pred_EOL - true_EOL| / true_EOL  # End-of-Life 预测误差
```

---

## 实现路线图

### 阶段 1：最小可复现版本（2-3 周）

**目标**：跑通 Mamba + PINN，验证技术可行性

**任务**：
1. ✅ 创建项目目录
2. ✅ 编写技术文档
3. ⏳ 实现 MIT 数据加载模块
4. ⏳ 实现 Mamba-PINN 模型
5. ⏳ 实现损失函数（Baseline + AdpBal）
6. ⏳ 实现 MIT 训练脚本
7. ⏳ 验证收敛性和初步结果

### 阶段 2：Baseline 对比实验（2-3 周）

**Baseline 模型**（参考 SOHbenchmark）：
- LSTM-PINN
- GRU-PINN
- Transformer-PINN
- 纯 Mamba（无 PINN）
- 纯 PINN（MLP backbone）

### 阶段 3：跨数据集验证与消融实验（2-3 周）

**跨数据集验证**：
1. MIT 训练 → NASA 零样本测试
2. MIT 训练 → NASA 少样本微调 (10/20/50 样本)
3. MIT 训练 → CALCE 零样本测试

**消融实验**：
1. Mamba + Verhulst PINN + AdpBal (完整模型)
2. Mamba + Baseline Loss (去除物理约束)
3. Mamba + Sum Loss (去除自适应平衡)
4. 不同窗口大小的影响 (8/16/32)
5. 不同模型规模的影响 (d_model=32/64/128)

### 阶段 4：论文撰写（3-4 周）

1. 结果分析与可视化
2. 鲁棒性测试（噪声、少样本）
3. 可解释性分析（物理参数学习、注意力权重）
4. 论文撰写与投稿

---

## 代码结构

```
20260615_MambaPINN_BatteryRUL/
├── README.md                    # 本文档
├── requirements.txt             # 依赖包
├── config.yaml                  # 配置文件
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   │   ├── mit/                # MIT .pkl 文件
│   │   ├── nasa/               # NASA .mat 文件
│   │   └── calce/              # CALCE 数据
│   └── processed/               # 处理后的数据
│       ├── mit_soh.pkl
│       └── nasa_soh.pkl
├── datasets/                    # 数据集类
│   ├── __init__.py
│   ├── mit_dataset.py          # MIT 数据加载
│   ├── nasa_dataset.py         # NASA 数据加载
│   └── calce_dataset.py        # CALCE 数据加载
├── models/                      # 模型定义
│   ├── __init__.py
│   ├── mamba.py                # 官方 Mamba-1 适配层
│   └── mamba_pinn.py           # MambaPINN 主模型
├── losses/                      # 损失函数
│   ├── __init__.py
│   ├── pinn_loss.py            # Verhulst 物理损失
│   └── adaptive_balance.py     # AdpBal 损失
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── metrics.py              # RMSE/MAE/MAPE/R²
│   ├── visualization.py        # 绘图函数
│   └── logger.py               # 日志记录
├── scripts/run_mit_mambapinn.py # MIT 主实验训练脚本
├── evaluate.py                  # 评估脚本
├── cross_dataset_eval.py        # 跨数据集零样本测试
├── few_shot_finetune.py         # 少样本微调脚本
├── results/                     # 结果保存
│   ├── checkpoints/            # 模型权重
│   ├── logs/                   # TensorBoard 日志
│   ├── metrics/                # 评价指标 CSV
│   └── figures/                # 可视化图表
└── notebooks/                   # Jupyter 实验
    ├── 01_explore_mit.ipynb    # MIT 数据探索
    ├── 02_verify_verhulst.ipynb # 物理方程验证
    └── 03_visualize_results.ipynb # 结果可视化
```

---

## 关键实现细节

### 1. Mamba 输入形状处理

Mamba 期望输入 `(B, L, D)`，当前实现的处理流程是：
- `B`：batch_size
- `L`：滑动窗口长度，即 `window_size`
- `D`：每个 cycle 的输入维度，SOH 单变量时为 1，投影后为 `d_model`

```python
# 输入：(batch_size, window_size)
seq = x.unsqueeze(-1)       # → (batch_size, window_size, 1)
h = self.input_proj(seq)    # → (batch_size, window_size, d_model)
h = self.backbone(h)        # → (batch_size, window_size, d_model)
h = self.norm(h[:, -1, :])  # → (batch_size, d_model)
pred = self.head(torch.cat([h, t], dim=-1))  # → (batch_size, 1)
```

### 2. PINN 梯度计算

```python
# 必须在前向传播时保留计算图
inputs.requires_grad_(True)
predictions = model(inputs)

# 使用 autograd 计算导数
grad_outputs = torch.ones_like(predictions)
dC_dt = torch.autograd.grad(
    predictions, inputs,
    grad_outputs=grad_outputs,
    create_graph=True,      # 保留计算图，支持二阶导数
    retain_graph=True,      # 多次反向传播
    only_inputs=True
)[0]
```

### 3. 自适应权重参数化

```python
# log_sigma 作为可训练参数
log_sigma_u = nn.Parameter(torch.randn(()))
log_sigma_f = nn.Parameter(torch.randn(()))

# 权重 = exp(-log_sigma)，自动满足正数约束
weight_u = torch.exp(-log_sigma_u)
weight_f = torch.exp(-log_sigma_f)

# 完整损失（含正则项）
loss = weight_u * loss_pred + weight_f * loss_physics + log_sigma_u + log_sigma_f
```

---

## 依赖环境

- Linux 或 WSL2
- NVIDIA GPU 与 CUDA 可用的 PyTorch
- Python 3.10+
- 官方 Mamba-1：`mamba-ssm==2.2.4`、`causal-conv1d==1.4.0`

完整依赖见 `requirements.txt`。当前项目的验证基线为 PyTorch `2.6.0+cu124` / CUDA `12.4`。

官方 Mamba-1 的 CUDA 扩展必须在 Linux 环境运行。所有官方 Mamba-1 实验固定 `expand_factor: 4`，以保持与旧简易实现的内部展开容量一致；旧简易实现 checkpoint 与新模型不兼容，需重新训练。

---

## 源项目参考

### MambaLithium
- **路径**：`d:\MyPaperCode\20260608_MambaLithium-main`
- **关键文件**：`mamba.py`, `pscan.py`, `main.py`
- **借鉴**：Mamba 模型结构、并行 scan 实现

### PINN-Battery-Prognostics
- **路径**：`d:\MyPaperCode\20260601_PINN-Battery-Prognostics-main_NPU`
- **关键文件**：`functions.py`, `Experiments/4 Presentation/RUL Prognostics/*.py`
- **借鉴**：Verhulst 方程、DeepHPM 架构、AdpBal 损失函数

### SOHbenchmark
- **路径**：`d:\MyPaperCode\20260518_SOHbenchmark-main_ByXJT`
- **关键文件**：`nets/Model.py`, `dataloader/*.py`
- **借鉴**：标准化实验流程、公平对比超参、评价指标计算

---

## 快速开始

```bash
# 1. 在 Linux / WSL2 中启用项目虚拟环境
source .venv/bin/activate

# 2. 仅在新环境中安装依赖
pip install -r requirements.txt --no-build-isolation

# 3. 将 MIT 原始数据放入 data/raw/mit/

# 4. 运行 MIT 主实验（窗口为 8）
python scripts/run_mit_mambapinn.py --config configs/config_w8.yaml

# 训练结果不会写入旧简易实现目录，而是保存在：
# results/mambapinn_official_mamba1_mit_w8/checkpoint/best_model.pt
# results/mambapinn_official_mamba1_mit_w8/checkpoint/history.json
# results/mambapinn_official_mamba1_mit_w8/metrics/metrics.json
# results/mambapinn_official_mamba1_mit_w8/metrics/test_predictions.csv
```

---

## 论文写作提示

### 核心卖点

1. **创新性**：首次将 Mamba SSM 与 PINN 结合用于电池预测
2. **性能提升**：相比 LSTM/Transformer 等 baseline 的 RMSE 降低 X%
3. **物理可解释性**：学习到的 Verhulst 参数符合电池退化物理规律
4. **自适应平衡**：AdpBal 自动权衡数据驱动和物理约束

### 实验对比表格模板

| 模型 | RMSE (↓) | MAE (↓) | MAPE (%) (↓) | R² (↑) | 参数量 |
|------|---------|---------|-------------|--------|--------|
| LSTM | X.XX±X.XX | X.XX±X.XX | X.XX±X.XX | 0.XX±0.XX | XXK |
| GRU | X.XX±X.XX | X.XX±X.XX | X.XX±X.XX | 0.XX±0.XX | XXK |
| Transformer | X.XX±X.XX | X.XX±X.XX | X.XX±X.XX | 0.XX±0.XX | XXK |
| 纯 Mamba | X.XX±X.XX | X.XX±X.XX | X.XX±X.XX | 0.XX±0.XX | XXK |
| 纯 PINN | X.XX±X.XX | X.XX±X.XX | X.XX±X.XX | 0.XX±0.XX | XXK |
| **Mamba-PINN (Ours)** | **X.XX±X.XX** | **X.XX±X.XX** | **X.XX±X.XX** | **0.XX±0.XX** | **XXK** |

### 可视化建议

1. **容量退化曲线对比图**：真实值 vs 预测值
2. **物理参数收敛曲线**：r, C_max, C_min 训练过程
3. **自适应权重演化**：λ_u, λ_f 随 epoch 变化
4. **消融实验柱状图**：各组件贡献度
5. **跨数据集迁移性能**：MIT → NASA/CALCE 零样本 vs 少样本对比
6. **少样本学习曲线**：微调样本数 vs 测试 RMSE

---

## 常见问题

### Q1: Mamba 和 LSTM 的核心区别？
A: Mamba 基于状态空间模型 (SSM)，使用 selective mechanism 让参数随输入动态变化，理论上能更好捕捉长期依赖且计算效率更高（线性复杂度 vs LSTM 的二次复杂度）。

### Q2: 为什么选择 Verhulst 方程？
A: Verhulst 是经典的逻辑增长模型，符合电池容量"先慢后快再慢"的 S 型退化曲线。相比线性退化或指数退化，Verhulst 更贴近实际物理过程。

### Q3: AdpBal 相比固定权重的优势？
A: 固定权重需要手动调参，且不同数据集/实验最优权重不同。AdpBal 通过可训练参数自动学习最优平衡，避免了繁琐的网格搜索。

### Q4: 为什么用 MIT 做主实验而不是 NASA？
A: MIT 有 124 块电池，样本量充足，能充分发挥 Mamba 的长序列建模能力。NASA 只有 4 块电池，样本量不足以支撑大规模训练，但适合作为跨数据集鲁棒性验证。

### Q5: 如何处理不同数据集容量量纲不同？
A: 统一归一化为 SOH (SOH = Capacity / Rated_Capacity)，MIT(1.1Ah)、NASA(2.0Ah)、CALCE 等数据集都映射到 [0,1] 范围，物理参数 C_max=1.0, C_min=0.7 通用。

### Q6: 跨数据集迁移时需要重新训练吗？
A: 不需要。零样本迁移直接用 MIT 训练的模型在 NASA 上测试。如果性能不理想，可以用少量 NASA 样本微调输出层，通常 50 个样本就能显著提升。

---

## 更新日志

- **2026-06-15**：项目初始化，完成技术文档
- **待完成**：数据加载模块、模型实现、训练脚本

---

## 联系方式

**项目负责人**：[您的姓名]  
**邮箱**：[您的邮箱]  
**GitHub**：[项目仓库链接]

---

## 许可证

本项目仅用于学术研究，请勿用于商业用途。

引用源项目：
- MambaLithium: [论文链接]
- PINN-Battery-Prognostics: [论文链接]
- SOHbenchmark: Wang et al. (2024)
