# Lotka-Volterra 实验文档

本文档详细介绍了 Lotka-Volterra 种群模型推断实验的各个环节，包括数据生成机制、先验设置、真实参数、实验模型配置（含 Refine+ 策略）、实验运行流程以及 SMMD 的序贯学习（Sequential Learning）架构。

## 1. Data Generation Process (Gillespie Method)

Lotka-Volterra 模型的模拟采用 **Gillespie 算法（Stochastic Simulation Algorithm, SSA）** 实现，这是一个用于模拟随机化学反应系统的精确方法。在我们的代码中（`data_generation.py`），该过程被向量化以支持批量模拟。

### 反应动力学
模型包含两个物种：猎物（Prey, $Y$）和捕食者（Predator, $X$）。系统演化由以下四个反应通道控制：
1.  **Predator Born (繁殖)**: $X + Y \xrightarrow{\theta_1} 2X + Y$ （捕食者吃掉猎物后繁殖）
2.  **Predator Die (死亡)**: $X \xrightarrow{\theta_2} \emptyset$ （捕食者自然死亡）
3.  **Prey Born (繁殖)**: $Y \xrightarrow{\theta_3} 2Y$ （猎物自然繁殖）
4.  **Prey Die (被捕食)**: $X + Y \xrightarrow{\theta_4} X$ （猎物被捕食者吃掉，不导致捕食者立即繁殖，通常合并在反应1中，但在本代码实现中作为独立参数控制交互消耗）
    *   *注：代码实现中 `rates_params` 对应 `[c1, c2, c3, c4]`，分别对应上述反应速率系数。*

### 模拟步骤
对于每一组参数 $\theta$：
1.  **初始化**：设置初始种群数量 $X_0=50, Y_0=100$。
2.  **速率计算**：在当前状态下，计算所有可能反应的速率（Propensity functions）。
    *   $r_1 = c_1 \cdot X \cdot Y$
    *   $r_2 = c_2 \cdot X$
    *   $r_3 = c_3 \cdot Y$
    *   $r_4 = c_4 \cdot X \cdot Y$
    *   $R_{total} = \sum r_i$
3.  **时间推进**：采样下一个反应发生的时间间隔 $\tau \sim \text{Exp}(R_{total})$。
4.  **反应选择**：根据 $r_i / R_{total}$ 的概率选择发生的反应，并更新 $X, Y$ 的数量。
5.  **观测记录**：在固定的时间点记录种群数量（由于 Gillespie 是连续时间的，通常需要插值或在特定时间点采样）。我们的实现记录了固定时间步长的状态序列。

## 2. Prior (Vague/Uninformative)

我们使用了对数空间上的均匀分布作为先验（Vague Prior）：
*   **分布**: $\log(\theta_i) \sim U(-5, 2)$ for $i=1, \dots, 4$
*   **范围**: 对应参数实际值范围约为 $[0.0067, 7.389]$。

### 局限性说明
使用这种宽泛的无信息先验（Vague Prior）会带来一个显著问题：
> "For vague prior, many of the ‘true’ parameters considered corresponded to uninteresting models, where both populations died out quickly, or the prey population diverge."

即在先验空间中采样的参数，很大一部分会导致种群迅速灭绝（趋于0）或者猎物数量指数级爆炸（Diverge）。这使得许多训练数据包含的是缺乏信息的“平坦”或“爆炸”轨迹，增加了推断的难度，也体现了 **Refine+** 或 **Sequential Learning** 策略的重要性，即引导模型关注那些能产生“存活且波动”轨迹的参数区域。

## 3. 真实参数设置 (Ground Truth)

为了评估模型性能，我们设定了一组真实的参数值用于生成观测数据（Observation）：

*   **Log Scale Values**: $\theta_{true} = \log([0.01, 0.5, 1.0, 0.01])$
*   **Linear Scale Values**:
    *   $\theta_1 \text{ (Predator Born)} = 0.01$
    *   $\theta_2 \text{ (Predator Die)} = 0.5$
    *   $\theta_3 \text{ (Prey Born)} = 1.0$
    *   $\theta_4 \text{ (Prey Die)} = 0.01$

该组参数产生的数据展现了经典的 Lotka-Volterra 周期性震荡特征。

## 4. 实验模型与 Refine+ 策略

### 支持模型
实验支持多种神经推断方法，通过 `config.json` 配置：
*   **SMMD**: Score-based Maximum Mean Discrepancy (基于分数的最大均值差异)
*   **MMD**: Maximum Mean Discrepancy (最大均值差异)
*   **BayesFlow**: 基于流的生成模型
*   **SNPE**: Sequential Neural Posterior Estimation (via `sbi` package)

### Refine+ 训练策略
针对 SMMD 和 MMD 模型，我们引入了 "Refine+" 步骤来提升后验推断的精度。这是一种简单的序贯改进策略：

1.  **初始训练**: 使用先验 $\theta \sim p(\theta)$ 生成的数据集进行第一轮训练。
2.  **Refine 阶段**:
    *   **混合采样**: 构造一个新的训练集，数据来源两部分：
        *   50% 来自先验 $p(\theta)$。
        *   50% 来自当前模型的后验预测 $\hat{p}(\theta|x_{obs})$。
    *   **重要性加权 (Weighting)**: 为了纠正采样偏差，我们需要对来自后验的样本进行加权。权重计算公式为：
        $$ w(\theta) = \frac{p(\theta)}{0.5 \cdot p_{KDE}(\theta) + 0.5 \cdot p(\theta)} $$
        其中 $p_{KDE}(\theta)$ 是对当前混合样本拟合的核密度估计（KDE）。这本质上是将采样分布视为 Mixture Distribution 的重要性采样。
    *   **再训练**: 使用加权后的混合数据集对模型进行微调（Fine-tuning）或重训练。

此步骤旨在让模型在真实观测值 $x_{obs}$ 附近的高概率区域获得更多的训练样本，同时保持一定的先验探索能力。

## 5. run_experiment 流程

`run_experiment.py` 是实验的主入口，其标准执行流程如下：

1.  **配置加载**: 读取 `config.json`，初始化参数（学习率、网络结构、Refine 轮数等）。
2.  **任务初始化**: 实例化 `LVTask`，加载或生成真实观测数据 $x_{obs}$。
3.  **基准训练 (Base Training)**:
    *   从先验生成数据集 $\mathcal{D} = \{(\theta_i, x_i)\}_{i=1}^N$。
    *   训练选定的模型（SMMD, BayesFlow 等）。
    *   保存第一阶段模型权重。
4.  **Refine+ 训练 (Optional)**:
    *   如果配置开启且模型支持（SMMD/MMD），执行上述 "Refine+" 循环。
    *   每一轮 Refine 后更新模型权重。
5.  **评估与绘图**:
    *   **后验采样**: 针对 $x_{obs}$ 采样后验分布。
    *   **可视化**: 调用 plotting 工具绘制 Pair Plot（参数相关性）和 Trace Plot（后验预测轨迹 vs 真实轨迹）。
    *   **指标计算**: 计算 C2ST (Classifier 2-Sample Test) 等指标评估后验质量。

## 6. SMMD Sequential Learning 详细步骤与 Scaffold

`SMMD_sequentiallearning.py` 实现了一个更纯粹的序贯神经后验估计（SNPE-style）流程，专门针对 SMMD 进行了适配。

### 核心步骤 (Algorithm Steps)

该过程分多轮（Rounds）进行：

1.  **Round 1 (Initialization)**:
    *   从先验 $p(\theta)$ 采样 $\theta$ 并模拟 $x$。
    *   权重 $w=1$。
    *   训练初始模型 $q_1(\theta|x)$。

2.  **Round $r > 1$ (Sequential Refinement)**:
    *   **后验采样**: 利用上一轮的模型 $q_{r-1}(\theta|x_{obs})$ 生成新的参数候选 $\theta_{new}$。
    *   **数据聚合**: 将 $\theta_{new}$ 及其模拟结果 $x_{new}$ 加入数据池（Data Pool）。这意味着训练集大小随轮数增加（Doubling data strategy）。
    *   **密度估计与加权**:
        *   使用 KDE (Kernel Density Estimation) 拟合当前数据池中的 $\theta$ 分布，估计采样密度 $q_{proposal}(\theta)$。
        *   计算重要性权重：$w(\theta) \propto \frac{p(\theta)}{q_{proposal}(\theta)}$。
        *   *注：为了数值稳定性，权重通常会归一化。*
    *   **加权训练**: 使用带权重的损失函数训练模型 $q_r(\theta|x)$。

### Scaffold (脚手架/基础设施)

为了保证序贯学习的稳定性，我们实现了一套完善的训练脚手架（Scaffold）：

*   **验证集机制 (Validation Set)**: 每一轮训练数据都会划分出验证集（Validation Split），用于监控过拟合。
*   **早停策略 (Early Stopping)**:
    *   监控验证集损失（Validation Loss）。
    *   如果在 `patience` 个 Epoch 内验证集损失未下降，则提前终止当前轮次的训练。
    *   这防止了模型在后期轮次中对特定采样偏差过拟合。
*   **学习率调度 (Scheduler)**: 使用 Cosine Decay 或 ReduceLROnPlateau 动态调整学习率。
*   **权重裁剪 (Weight Clipping)**: 防止 KDE 估计导致的极端权重值破坏梯度稳定性。

此流程使得 SMMD 能够像 SNPE 一样，通过多轮迭代自动聚焦到真实后验的高概率区域，从而显著提升样本效率。
