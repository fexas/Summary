# 修改建议

1. 模型网络的初始化用N(0,0.2^2)。并与合适处添加RMSNorm（SummaryNetwork和Generative Network中）。保持SMMD、MMD和Bayesflow及DNNABC使用的SummaryNetwork在概念结构上一样（代码实现可以不一样）。
2. （SMMD，MMD）对训练使用的损失函数增加基于Generative Network的参数的l1正则项，正则系数为1e-4。将优化器修改为AdamW，初始学习率修改为1e-3。
3. SMMD和MMD所使用的generative network的中间层从原先的两个神经元数量为128的两个中间层修改为64个神经元的3个中间层。

## 实验结果保存

 实验结果的保存请采取如下结构：
对于每个模型 `model_type`、每轮 `round_id`，`run_single_experiment` 会执行：

1. 训练模型；
2. 对观测 `x_obs` 抽样得到初始后验样本 `posterior_samples`；
3. 计算初始后验的指标 `metrics_initial`；
4. 对 SMMD / MMD / BayesFlow 执行 Refine（ABC-MCMC）得到精炼后的后验样本和指标。

(经过ABC-MCMC refine得到的后验，在作图或者记录的时候，label都用Refine)

对应的存储与绘图结构如下。

#### 5.2.1 每轮每模型的样本与单模型图像

根目录：`results/models/{model_type}/round_{round_id}`。

- 初始后验：
  - 目录：`initial/`
    - `posterior_samples.npy`：初始后验样本，形状 `(n_samples, d)`。
    - `posterior_plot.png`：该模型在该轮的初始后验可视化。
- Refine后验（仅 SMMD / MMD / BayesFlow）：
  - 目录：`refine/`
    - `posterior_samples.npy`：Refine后验样本。
    - `posterior_plot.png`：Refine后验的可视化。
- 初始 vs Refine对比图：
  - 目录：`comparison/`
  - 内容：利用 `plot_refinement_comparison`，将 `Initial` 和 `Refine+`（如果存在 MCMC，则为最终 Refine+ MCMC）放在同一张图上进行比较（如一维边缘、二维等高线对比等）。

对于没有 Refine的方法（如 DNNABC、NPE），只有 `initial/` 子目录，不会生成 `refine/` 与 `comparison/`。

#### 5.2.2 跨模型的对比图

在多模型多轮实验的外层循环中，会额外保存跨模型的比较图。

- 根目录：`results/comparisons/round_{round_id}`。
- 文件：
  - `all_methods_initial_posterior.png`：
    - 聚合该轮所有模型的初始后验样本；
    - 展示不同方法对真实参数的初始拟合情况。
  - `all_methods_with_refine_posterior.png`：
    - 聚合该轮所有模型的初始后验样本与 Refine/ MCMC 后的最终后验；
    - 用于比较 Refine带来的改进（仅对具备 Refine的模型有对应曲线）。

#### 5.2.3 每轮的表格存储

在主循环结束后，对 `model_results_table` 中的每个模型，保存 per-round 原始表和精简的 per-round metric 表：

- 原始 per-round 表（调试用）：
  - 路径：`results/tables/{model_name}_results.csv`。
  - 列大致包括：
    - 基本信息：`round`, `status`, `stage`（`initial` 或 `refined_mcmc` 等）；
    - 当轮指标
    - 时间信息：`time_initial`, `time_refined_plus`, `time_mcmc`, `training_time`, `sampling_time`；
    - 对于有 Refine 的模型，还包含refine后的评价指标。

- 精简 per-round metric 表：
  - 路径：`results/models/{model_name}/{model_name}_per_round_metrics.csv`。
  - 每行对应一轮（包括失败轮，失败时数值为 NaN），主要列为：
    - `round`：轮次；
    - 指标
    - refine 后得到的指标

这样可以直接用一张 per-model CSV 分析，同时保证 pre Refine和 Refine指标在列上紧挨着出现。

### 5.3 实验结束后的总体指标（final_summary）

所有轮次结束后，会对每个模型在所有成功轮上进行统计，并写入：

- 路径：`results/final_summary.csv`。
- 每行对应一个模型，列的排列方式刻意将 pre Refine与 Refine指标放在一起，便于对比：

- 基本信息：
  - `Model`：模型名称（如 `smmd`, `mmd`, `bayesflow`, `dnnabc`, `npe`）。
- 指标
- 时间指标：
  - `Avg_Time_Initial`：所有轮次 `seconds_initial` 的平均（格式化为 `HH:MM:SS`）；
  - `Avg_Time_Refined`：所有轮次做ABC-MCMC 后的平均（格式化为 `HH:MM:SS`）。



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
        refine/
          posterior_samples.npy
          posterior_plot.png
        comparison/
          ...                          # 初始 vs Refine对比图
      round_2/
        ...

    mmd/
      mmd_per_round_metrics.csv
      round_k/...

    bayesflow/
      bayesflow_per_round_metrics.csv
      round_k/...

    dnnabc/
      dnnabc_per_round_metrics.csv     # Refine相关列为 NaN
      round_k/initial/...

    npe/
      npe_per_round_metrics.csv        # Refine相关列为 NaN
      round_k/initial/...

  comparisons/
    round_1/
      all_methods_initial_posterior.png
      all_methods_with_refine_posterior.png
    round_2/
      ...
```

在迁移到其他 benchmark 时，只要保持类似的 metric 命名、per-round 存储与 `final_summary.csv` 列布局，
就可以用相同的 prompt 模板来描述、解析和比较不同问题上的 SMMD / MMD / BayesFlow 表现。

