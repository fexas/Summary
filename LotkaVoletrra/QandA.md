# Q and A

Q1: SNPE 和 Bayesflow 会显式或者隐式的约束generative model的output吗？比如说prior的范围是[-5,5]，在SNPE和Bayesflow除开Summary network的后半部分，是否会有一些trick来确保output在[-5,5]之间？

**A1:**

针对我们当前的代码实现（Lotka-Volterra 任务），两者的处理方式不同：

1.  **SNPE (基于 `sbi` 包)**:
    *   **机制**: **隐式约束 (Rejection Sampling)**。
    *   **解释**: 在 `models/sbi_wrappers.py` 中，我们定义了 `BoxUniform` Prior (范围 `[-5, 2]`) 并传递给了 `SNPE` 和 `build_posterior`。
    *   `sbi` 训练的神经密度估计器（Normalizing Flow）本身是定义在整个实数域 $\mathbb{R}^d$ 上的，因此模型输出的原始样本可能会超出 `[-5, 2]` 的范围（即 "Leakage"）。
    *   但是，当我们调用 `posterior.sample()` 时，`sbi` 默认会使用 **Rejection Sampling**（拒绝采样）。它会从流模型中采样，然后检查样本是否在 Prior 的支持集（Support）内。如果样本超出范围（例如 -5.1），该样本会被丢弃并重新采样，直到满足 Prior 约束。
    *   **代码体现**: `posterior = inference.build_posterior(density_estimator)`。`sbi` 的 `DirectPosterior` 类在采样时会自动利用我们传入的 `prior` 进行检查。

2.  **BayesFlow**:
    *   **机制**: **无显式约束 (当前实现)**。
    *   **解释**: 在 `models/bayesflow_net.py` 中，我们使用的是标准的 `bf.networks.CouplingFlow`。
    *   Normalizing Flow 通常构建在 $\mathbb{R}^d$ 上的微分同胚映射。在该代码实现中，没有添加额外的 Bijector（如 `Sigmoid` 或 `Softplus` 结合缩放平移）来将输出空间强制映射到 `[-5, 2]`。
    *   **结果**: BayesFlow 完全依靠数据驱动来学习分布。如果训练得当，后验概率质量（Probability Mass）会集中在 `[-5, 2]` 之间，但从理论上讲，模型**可以**输出超出此范围的值。
    *   **改进方法**: 若要强制约束，通常需要在 Flow 的构建中加入一个边界转换层（Bounded Bijector），或者在采样后手动进行截断/拒绝采样。

**总结**:
*   **SNPE**: 生成模型本身无约束，但通过 `sbi` 库的后处理（采样阶段）强制保证输出在 Prior 范围内。
*   **BayesFlow**: 当前代码中模型输出无硬性约束，依赖模型学习到的分布集中度。
  
Q1.2： q请问`sbi` 库的后处理（采样阶段）强制保证输出在 Prior 范围内是怎么实现的？

**A1.2:**

`sbi` 库通过 **Rejection Sampling (拒绝采样)** 机制来实现这一约束。

具体流程如下：
1.  **构建后验对象**: 当调用 `inference.build_posterior(density_estimator)` 时，`sbi` 会将训练好的密度估计器（Flow）与之前定义的 `prior` 绑定，生成一个 `DirectPosterior` 对象。
2.  **采样过程 (`sample`)**:
    *   用户调用 `posterior.sample(num_samples)`。
    *   `sbi` 首先从神经密度估计器（Flow）中快速生成一批候选样本（Proposal Samples）。由于 Flow 定义在整个实数域，这些样本可能包含超出 Prior 范围的值（Leakage）。
    *   **有效性检查**: `sbi` 会调用 `prior.log_prob(samples)` 来计算这些候选样本在 Prior 下的对数概率。对于超出 Prior 支持集（Support）范围的样本，`log_prob` 会返回 `-inf`。
    *   **拒绝与重采**: `sbi` 保留那些 `log_prob > -inf` 的样本。对于被拒绝（丢弃）的样本数量，`sbi` 会自动计算并补齐，再次从 Flow 中采样，直到收集到足够数量（`num_samples`）的有效样本。
    *   **警告**: 如果 Flow 的“泄露”（Leakage）非常严重（即大量样本落在 Prior 范围外），`sbi` 会发出警告（`Sampling is very slow...`），提示后验分布与 Prior 差异过大或模型未训练好。

Q2： SNPE的训练过程是什么样的？（每轮使用多少个训练样本，训练多少个epochs, 使用什么优化器，对应的batch是多少，怎么决定学习率，怎么决定要不要早停？）

A2:
本项目中使用 `sbi` 库实现的 Sequential Neural Posterior Estimation (SNPE-A) 方法。
- **每轮训练样本数**: 1000 (`config.json` 中配置 `"sims_per_round": 1000`)。SNPE 是一种序贯方法，每轮会从当前的 proposal (上一轮的后验分布) 中采样 1000 个新的 $\theta$ 并生成对应的观测数据 $x$，并将这些新数据加入到之前累积的数据集中一起用于训练。
- **训练 Epochs**: 每轮最大训练 1000 个 epochs (`config.json` 中配置 `"epochs": 1000`)。
- **优化器**: 使用 `sbi` 默认的 Adam 优化器。
- **Batch Size**: 50 (`sbi` 默认值)。
- **学习率**: 5e-4 (`sbi` 默认值)。
- **早停机制 (Early Stopping)**: 启用。`sbi` 的 `train` 方法默认会将所有累积的模拟数据按 9:1 划分为训练集和验证集 (`validation_fraction=0.1`)。如果验证集上的 loss 在连续 20 个 epochs (`stop_after_epochs=20`) 内没有改善，训练将提前停止，以防止过拟合。

