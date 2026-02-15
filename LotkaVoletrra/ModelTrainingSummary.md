## Lotka–Volterra 模型结构与训练总结（SMMD / MMD / BayesFlow）

本文总结 Lotka–Volterra 实验中三个主力方法（SMMD、MMD、BayesFlow）的
Summary Network 结构、生成网络结构（SMMD/MMD），以及训练时的损失函数和优化器设置。

---

## 1. Summary Network 结构与参数设置

### 1.1 SMMD / MMD：TimeSeriesSummaryNet（PyTorch）

位置：`models/smmd.py` 中的 `TimeSeriesSummaryNet` 与 `SMMD_Model`。

- **输入输出**
  - 输入：`x` 形状为 `(batch, n_points, d_x)`，其中 `n_points = n_time_steps`，`d_x` 是观测维度。
  - 输出：时间序列 summary 向量，维度为 `summary_dim`（在 `config.json` 中配置，当前为 8）。
  - 在 `SMMD_Model` 中：
    - `self.T = SummaryNet(n_points=n, input_dim=d_x, output_dim=summary_dim)`。
    - `SummaryNet` 实际别名为 `TimeSeriesSummaryNet`。

- **网络结构**
  - 先将输入从 `(batch, seq_len, input_dim)` 变为 `(batch, input_dim, seq_len)` 以适配 `Conv1d`：
    - `x = x.permute(0, 2, 1)`
  - 卷积 + 池化 + 全局池化：
    - Block 1：
      - `Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=10, padding=2)`
      - `ReLU`
      - `MaxPool1d(kernel_size=2)`
    - Block 2：
      - `Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=10, padding=2)`
      - `ReLU`
    - 全局平均池化：
      - `AdaptiveAvgPool1d(1)` → 输出形状 `(batch, hidden_dim*2, 1)`
  - 展平 + 归一化 + 全连接：
    - 展平到 `(batch, hidden_dim*2)`。
    - `RMSNorm(hidden_dim*2)`：对卷积特征做 RMS 归一化。
    - `Linear(hidden_dim*2, output_dim)`：输出 summary 向量。

- **超参数**
  - `hidden_dim = 64`（在实现中固定）。
  - `output_dim = summary_dim`，从 `config.json` 中读取（目前为 8）。

MMD 使用的 `MMD_Model` 结构上直接继承自 `SMMD_Model`，只是在训练时使用不同的损失（欧式 MMD），因此 Summary Network 与 SMMD 完全一致。

### 1.2 BayesFlow：TimeSeriesSummary（Keras）

位置：`models/bayesflow_net.py` 中的 `TimeSeriesSummary` 和 `build_bayesflow_model`。

- **输入输出**
  - 输入：`x` 形状 `(batch, time_steps, d_x)`。
  - 输出：summary 向量，维度为 `summary_dim`（同样从 `config.json` 读取）。
  - 在 `build_bayesflow_model` 中：
    - `summary_net = TimeSeriesSummary(input_dim=d_x, output_dim=summary_dim)`。

- **网络结构（Keras 实现）**
  - `Conv1D(filters=hidden_dim, kernel_size=5, padding="same", activation="relu")`
  - `MaxPooling1D(pool_size=2)`
  - `Conv1D(filters=hidden_dim*2, kernel_size=5, padding="same", activation="relu")`
  - `GlobalAveragePooling1D()`
  - `Dense(output_dim)`

- **超参数**
  - `hidden_dim = 64`。
  - `output_dim = summary_dim`，与 SMMD/MMD 对齐（目前为 8）。

### 小结

- SMMD / MMD 与 BayesFlow 都使用 1D CNN + 全局池化的 TimeSeries summary 思路。
- 区别在于：
  - SMMD / MMD 使用 PyTorch + `Conv1d` + `AdaptiveAvgPool1d` + `RMSNorm`。
  - BayesFlow 使用 Keras + `Conv1D` + `GlobalAveragePooling1D`。
- 两者在接口上都实现从 `(batch, time_steps, d_x)` 到 `(batch, summary_dim)` 的映射，便于对比实验。

---

## 2. SMMD / MMD 的生成网络结构与 RMSNorm

### 2.1 生成网络 G 的结构

位置：`models/smmd.py` 中的 `Generator`。

- **输入输出**
  - 输入：
    - `z`：噪声 / 潜变量，形状 `(batch, M, z_dim)`，其中 `z_dim = d`（参数维度）。
    - `stats`：Summary Network 输出，形状 `(batch, stats_dim)`，其中 `stats_dim = summary_dim`。
  - 在调用时会将 `stats` 与 `z` 拼接之后送入一个多层 MLP，输出 `theta_fake`（生成的参数样本）。

- **网络结构**
  - 先拼接并展平：
    - 输入维度：`input_dim = z_dim + stats_dim`。
  - 多层全连接 + RMSNorm + ReLU：
    - `Linear(input_dim, hidden_dim)` → `RMSNorm(hidden_dim)` → `ReLU`
    - `Linear(hidden_dim, hidden_dim)` → `RMSNorm(hidden_dim)` → `ReLU`
    - `Linear(hidden_dim, hidden_dim)` → `RMSNorm(hidden_dim)` → `ReLU`
    - `Linear(hidden_dim, hidden_dim)` → `RMSNorm(hidden_dim)` → `ReLU`
  - 输出层：
    - `Linear(hidden_dim, out_dim)`，其中 `out_dim = d`（参数维度）。

- **超参数**
  - `hidden_dim = 64`（固定）。
  - `z_dim = d`，`stats_dim = summary_dim`。

MMD 的 `MMD_Model` 直接继承 `SMMD_Model`：

- 位置：`models/mmd.py` 中：
  - `class MMD_Model(SMMD_Model): pass`
- 因此 MMD 的生成网络结构与 SMMD 完全相同。

### 2.2 RMSNorm 的作用与添加方式

RMSNorm 是一种基于均方根的归一化方式，类似 LayerNorm 但对均值不做偏移校正，适合稳定深层 MLP 的训练。

在当前实现中，RMSNorm 出现的地方主要有：

- TimeSeriesSummaryNet 的卷积特征输出（`post_conv_norm`）。
- Generator 每一层线性层后面。

**如何在新的网络结构中添加 RMSNorm：**

- 典型模式：
  - `Linear(...)` → `RMSNorm(hidden_dim)` → `ReLU`。
- 如果你在 Generator 中增加了一层：
  - 只需要增加：
    - `layers.append(nn.Linear(hidden_dim, hidden_dim))`
    - `layers.append(RMSNorm(hidden_dim))`
    - `layers.append(nn.ReLU())`
- 对于 Summary Network，如果你在卷积后还想加一层全连接，也可以在该层输出后接一个 `RMSNorm` 再接非线性激活。

**原则：**

- RMSNorm 用于稳定特征分布，尤其是在多层 MLP 或深层卷积堆叠时，可以减小梯度爆炸 / 消失风险。
- 在 Lotka–Volterra 的实现中，RMSNorm 主要用在：
  - CNN 结束后的高维特征上；
  - 多层全连接生成网络内部，每层线性之后。

---

## 3. 训练中的损失函数与优化器设置

### 3.1 SMMD / MMD 的损失

训练入口：`run_experiment.py` 中的 `train_smmd_mmd`。

#### 3.1.1 MMD / SMMD 损失

在每个 batch 内：

- 从真实参数 `theta_batch` 得到一批真实样本；
- 通过生成网络得到 fake 参数 `theta_fake`，形状 `(batch, M, d)`；
- 计算对应的 MMD 损失：
  - SMMD：
    - 使用 `sliced_mmd_loss(theta_batch, theta_fake, num_slices=L, n_time_steps=n_time_steps)`。
    - 通过随机投影到多个一维方向、再在这些投影上计算高斯核 MMD。
    - 带宽 `bandwidth = 5.0 / n_time_steps`。
  - MMD：
    - 使用 `mmd_loss(theta_batch, theta_fake, n_time_steps=n_time_steps)`。
    - 欧式 MMD，使用一组固定带宽的高斯核。

#### 3.1.2 L1 正则项的添加方式

在 MMD / SMMD 损失的基础上，还对生成网络 G 的参数加了一个 L1 正则项：

- 位置：`train_smmd_mmd` 内部。
- 具体做法：

  - 首先计算 MMD 或 sliced MMD 损失 `loss_mmd`。
  - 然后如果 `model` 有属性 `G`（即存在生成网络）：
    - 遍历 `model.G.parameters()`，累加 `|param|` 的和，得到 `l1_loss`。
    - 使用全局超参数 `L1_LAMBDA`（在文件顶部定义，当前为 `1e-4`）加权：
      - `loss = loss_mmd + L1_LAMBDA * l1_loss`。

**要点：**

- L1 正则只作用在生成网络 G 的参数上，不作用于 Summary Network；
- `L1_LAMBDA` 控制稀疏约束的强度，可以在 `run_experiment.py` 顶部调节。

### 3.2 优化器与学习率设置

#### 3.2.1 SMMD / MMD（PyTorch）

- 优化器：`torch.optim.AdamW`。
- 学习率：
  - 全局从 `config.json` 读取 `learning_rate`：
    - 当前配置为 `1e-3`。
  - 在 `run_experiment.py` 顶部：
    - `LEARNING_RATE = CONFIG.get("learning_rate", 3e-4)`。
  - 在 `train_smmd_mmd` 中：
    - `optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)`。
- 学习率调度：
  - `scheduler = get_scheduler(optimizer, epochs)`。
  - 在每一 epoch 结束后调用 `scheduler.step()`。

#### 3.2.2 BayesFlow（Keras）

- 在 `train_bayesflow` 中：
  - 通过 `build_bayesflow_model` 构建 `amortized_posterior`（BayesFlow 的 ContinuousApproximator）。
  - 使用 Keras 的 `Adam` 优化器，学习率同样使用 `LEARNING_RATE`：
    - `optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)`。
  - 损失函数由 BayesFlow 内部定义，通常是基于负对数似然或基于流模型的标准训练目标。
- 训练 loop：
  - 手动遍历 `train_loader`，把数据送入 BayesFlow 的 `train_step` 或自定义训练函数；
  - 训练轮数 `epochs` 从 `config.json` 中的 `models_config["bayesflow"]["epochs"]` 读取（当前为 200）。

### 3.3 如何在现有框架下扩展损失 / 优化器

如果你想进一步修改损失或优化器，推荐的扩展方式是：

- 在 `train_smmd_mmd` 中：
  - 以当前的 MMD（或 SMMD）损失为基础，逐项加上新的正则或辅助损失：
    - 例如增加一个 L2 正则，只需在 L1 部分旁边增加：
      - 遍历参数，累加 `param.pow(2).sum()`，用新的系数加权；
  - 保持总损失仍然是一个标量 `loss`，然后正常 `backward()` 和 `step()`。
- 在 `config.json` 中增加新的超参数字段（例如 `l1_lambda`、`l2_lambda`），在 `run_experiment.py` 顶部读取，以便统一控制不同实验的配置。
- 对 BayesFlow，如要更改优化器或学习率，集中修改 `train_bayesflow` 中的 `optimizer` 构造即可。

---

## 4. 训练配置概览（来自 config.json）

在 `LotkaVoletrra/config.json` 中，目前与这三种方法相关的主要训练配置为：

- 全局：
  - `n_time_steps`: 151
  - `batch_size`: 128
  - `learning_rate`: `1e-3`
  - `n_samples_posterior`: 1000
  - `smmd_mmd_config`：
    - `M`: 50（生成网络每个 batch 的采样数量）
    - `L`: 20（sliced MMD 的随机投影数量）

- 每个模型：
  - SMMD：
    - `epochs`: 200
    - `summary_dim`: 8
    - `refined_mode`: 1
  - MMD：
    - `epochs`: 200
    - `summary_dim`: 8
  - BayesFlow：
    - `epochs`: 200
    - `summary_dim`: 8

这些配置与上面的网络结构和训练 loop 直接对应，可以通过修改 `config.json` 来做系统性的实验（例如增大 `summary_dim`、改变 `M` 和 `L`，或者调节 `learning_rate` 和 `epochs`）。

---

## 5. 评估指标、逐轮存储与结果文件结构

本节总结 Lotka–Volterra 实验中使用的 metric、逐轮结果的存储结构、绘图方式，
以及实验结束后的总体指标与结果文件组织方式，便于在其他 benchmark 中复用。

### 5.1 评估指标定义

评估函数位置：`utilities.py` 中的 `compute_metrics`。

对给定的近似后验样本 `samples`（形状 `(n_samples, d)`）和真实参数 `theta_true`（形状 `(d,)`），
计算如下指标：

- `bias_l2`：
  - 标量；
  - 定义为后验均值与真实参数之差的 L2 范数，即  
    `bias_vec = mean(samples, axis=0) - theta_true`，`bias_l2 = ||bias_vec||_2`。
- `bias_vec`：
  - 形状 `(d,)`；
  - 每个参数维度上的偏差。
- `hdi_lower` / `hdi_upper`：
  - 形状 `(d,)`；
  - 每个参数维度上 95% HDI 的下界和上界，使用分位数 `[0.025, 0.975]`。
- `hdi_length`：
  - 形状 `(d,)`；
  - `hdi_upper - hdi_lower`，即每个参数维度上 95% HDI 的长度。
- `coverage`：
  - 形状 `(d,)`，元素为 `0` 或 `1`；
  - 表示真实参数是否落在对应维度的 HDI 内。

在训练过程中，这些指标既用于单轮评估，也被进一步汇总为 per-round 表格和最终的 `final_summary.csv`。

### 5.2 逐轮结果的存储结构与绘图

对于每个模型 `model_type`、每轮 `round_id`，`run_single_experiment` 会执行：

1. 训练模型；
2. 对观测 `x_obs` 抽样得到初始后验样本 `posterior_samples`；
3. 计算初始后验的指标 `metrics_initial`；
4. 对 SMMD / MMD / BayesFlow 执行 Refine+（以及之后的 MCMC）得到精炼后的后验样本和指标。

对应的存储与绘图结构如下。

#### 5.2.1 每轮每模型的样本与单模型图像

根目录：`results/models/{model_type}/round_{round_id}`。

- 初始后验：
  - 目录：`initial/`
    - `posterior_samples.npy`：初始后验样本，形状 `(n_samples, d)`。
    - `posterior_plot.png`：该模型在该轮的初始后验可视化。
- Refine+ 后验（仅 SMMD / MMD / BayesFlow）：
  - 目录：`refineplus/`
    - `posterior_samples.npy`：Refine+ 后验样本。
    - `posterior_plot.png`：Refine+ 后验的可视化。
- 初始 vs Refine+ 对比图：
  - 目录：`comparison/`
  - 内容：利用 `plot_refinement_comparison`，将 `Initial` 和 `Refine+`（如果存在 MCMC，则为最终 Refine+ + MCMC）放在同一张图上进行比较（如一维边缘、二维等高线对比等）。

对于没有 Refine+ 的方法（如 DNNABC、NPE），只有 `initial/` 子目录，不会生成 `refineplus/` 与 `comparison/`。

#### 5.2.2 跨模型的对比图

在多模型多轮实验的外层循环中，会额外保存跨模型的比较图。

- 根目录：`results/comparisons/round_{round_id}`。
- 文件：
  - `all_methods_initial_posterior.png`：
    - 聚合该轮所有模型的初始后验样本；
    - 展示不同方法对真实参数的初始拟合情况。
  - `all_methods_with_refineplus_posterior.png`：
    - 聚合该轮所有模型的初始后验样本与 Refine+ / MCMC 后的最终后验；
    - 用于比较 Refine+ 带来的改进（仅对具备 Refine+ 的模型有对应曲线）。

#### 5.2.3 每轮的表格存储

在主循环结束后，对 `model_results_table` 中的每个模型，保存 per-round 原始表和精简的 per-round metric 表：

- 原始 per-round 表（调试用）：
  - 路径：`results/tables/{model_name}_results.csv`。
  - 列大致包括：
    - 基本信息：`round`, `status`, `stage`（`initial` 或 `refined_mcmc` 等）；
    - 当轮指标：`bias_l2`, `hdi_length`, `coverage`, `bias_vec`, `hdi_lower`, `hdi_upper`；
    - 时间信息：`time_initial`, `time_refined_plus`, `time_mcmc`, `training_time`, `sampling_time`；
    - 对于有 Refine+ 的模型，还包含：
      - `bias_l2_initial`, `hdi_length_initial`, `coverage_initial`；
      - `bias_l2_refineplus`, `hdi_length_refineplus`, `coverage_refineplus`。

- 精简 per-round metric 表：
  - 路径：`results/models/{model_name}/{model_name}_per_round_metrics.csv`。
  - 每行对应一轮（包括失败轮，失败时数值为 NaN），主要列为：
    - `round`：轮次；
    - `Bias`：该轮 initial posterior 的 `bias_l2`；
    - `RefinePlus_Bias`：该轮 Refine+ posterior 的 `bias_l2`（无 Refine+ 的方法为 NaN）。
    - 对于每个参数维度 `j = 1..d`：
      - `HDI_Len_Param{j}`：initial posterior 的 95% HDI length；
      - `Coverage_Param{j}`：initial posterior 的 coverage；
      - `RefinePlus_HDI_Len_Param{j}`：Refine+ posterior 的 95% HDI length；
      - `RefinePlus_Coverage_Param{j}`：Refine+ posterior 的 coverage。

这样可以直接用一张 per-model CSV 分析每轮每个参数的收敛情况，同时保证 pre Refine+ 和 Refine+ 指标在列上紧挨着出现。

### 5.3 实验结束后的总体指标（final_summary）

所有轮次结束后，会对每个模型在所有成功轮上进行统计，并写入：

- 路径：`results/final_summary.csv`。
- 每行对应一个模型，列的排列方式刻意将 pre Refine+ 与 Refine+ 指标放在一起，便于对比：

- 基本信息：
  - `Model`：模型名称（如 `smmd`, `mmd`, `bayesflow`, `dnnabc`, `npe`）。
- Bias 汇总：
  - `Bias_Mean`：所有轮次 initial posterior 的 `bias_l2` 平均值；
  - `RefinePlus_Bias_Mean`：所有轮次 Refine+ posterior 的 `bias_l2` 平均值（无 Refine+ 时为 NaN）；
  - `Bias_Median`：initial posterior 的 `bias_l2` 中位数；
  - `RefinePlus_Bias_Median`：Refine+ posterior 的 `bias_l2` 中位数。
- 时间指标：
  - `Avg_Time_Initial`：所有轮次 `seconds_initial` 的平均（格式化为 `HH:MM:SS`）；
  - `Avg_Time_RefinedPlus`：所有轮次 `seconds_refined_plus` 的平均；
  - `Avg_Time_MCMC`：所有轮次 `seconds_mcmc` 的平均。
- 参数维度上的 HDI 与 Coverage：
  - 对于每个参数维度 `j = 1..d`，有四列依次相邻：
    - `HDI_Len_Param{j}_Mean`：initial posterior 在所有轮次上该维度的平均 HDI length；
    - `Coverage_Param{j}_Mean`：initial posterior 在所有轮次上该维度的平均 coverage；
    - `RefinePlus_HDI_Len_Param{j}_Mean`：Refine+ posterior 的平均 HDI length；
    - `RefinePlus_Coverage_Param{j}_Mean`：Refine+ posterior 的平均 coverage。

这样，针对每个参数维度，可以直接按照行内顺序读取 pre Refine+ 与 Refine+ 的 HDI 长度和覆盖率，方便横向比较不同模型、纵向比较 Refine+ 改进效果。

### 5.4 结果文件夹整体结构示意

Lotka–Volterra benchmark 的结果主要保存在 `LotkaVoletrra/results/` 目录，典型结构如下（略去部分细节）：

```text
results/
  final_summary.csv                    # 跨模型总体指标汇总

  tables/                              # 每模型逐轮原始表
    smmd_results.csv
    mmd_results.csv
    bayesflow_results.csv
    dnnabc_results.csv
    npe_results.csv

  models/
    smmd/
      smmd_per_round_metrics.csv       # 按轮的精简 metric 表
      round_1/
        initial/
          posterior_samples.npy
          posterior_plot.png
        refineplus/
          posterior_samples.npy
          posterior_plot.png
        comparison/
          ...                          # 初始 vs Refine+ 对比图
      round_2/
        ...

    mmd/
      mmd_per_round_metrics.csv
      round_k/...

    bayesflow/
      bayesflow_per_round_metrics.csv
      round_k/...

    dnnabc/
      dnnabc_per_round_metrics.csv     # Refine+ 相关列为 NaN
      round_k/initial/...

    npe/
      npe_per_round_metrics.csv        # Refine+ 相关列为 NaN
      round_k/initial/...

  comparisons/
    round_1/
      all_methods_initial_posterior.png
      all_methods_with_refineplus_posterior.png
    round_2/
      ...
```

在迁移到其他 benchmark 时，只要保持类似的 metric 命名、per-round 存储与 `final_summary.csv` 列布局，
就可以用相同的 prompt 模板来描述、解析和比较不同问题上的 SMMD / MMD / BayesFlow 表现。

