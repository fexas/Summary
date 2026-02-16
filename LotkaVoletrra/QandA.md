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


3. Bayesflow 和 NPE在非summary的generative network应该都是多个复合层将标准正态分布映射到期望分布。请问他们的结构是怎么样的？比如有多少层，每一个基础层对应的又是一个什么样的神经网络，参数是多少？

**A3:**

这里先把“summary network”和“generative / inference network（flow 部分）”分开说明，并严格按照当前 Lotka–Volterra 代码实现来回答。

---

**1. BayesFlow 中的 generative（后验）网络结构**

相关代码：`models/bayesflow_net.py` 中的 `build_bayesflow_model`。

- **Flow 类型**
  - 使用的是 BayesFlow 自带的  
    `bf.networks.CouplingFlow` 作为可逆网络（inference / generative network）。
  - 调用方式：
    - `inference_net = bf.networks.CouplingFlow(num_params=d, num_coupling_layers=4, coupling_settings={"dense_args": dict(units=64, activation="relu")})`

- **整体结构（非 summary 部分，只看 flow）**
  - 输入空间：标准正态噪声 $z \sim \mathcal{N}(0, I_d)$
  - 输出空间：参数 $\theta \in \mathbb{R}^d$
  - Flow 由 **4 个 coupling layer** 级联组成：
    - `num_coupling_layers = 4`
  - 每个 coupling layer 内部使用掩码把维度拆成两部分，一部分保持不变，另一部分通过条件网络做仿射（或更一般）变换：
    - “保持不变”的子向量直接穿过
    - “被变换”的子向量：  
      $y = x \odot \exp(s(\cdot)) + t(\cdot)$  
      其中 $s(\cdot), t(\cdot)$ 由一个小的 MLP 产生

- **每个 coupling layer 的基础网络（`coupling_settings`）**
  - `coupling_settings={"dense_args": dict(units=64, activation="relu")}`
  - 在 BayesFlow 的实现里，这表示：
    - 每个 coupling layer 的内部 MLP 使用 Dense 层，宽度为 64，激活为 ReLU
    - 典型形式类似：
      - Dense(64, activation="relu")
      - Dense(64, activation="relu")
      - Dense(2 * d) 或类似输出（拆分为 scale 和 shift）
  - 精确的层数和输出维度由 BayesFlow 内部默认设置决定，但可以肯定的是：
    - **隐藏层宽度固定为 64**
    - **激活函数统一为 ReLU**
    - **每个 coupling 层都有自己的独立 MLP 参数**

- **参数规模（数量级）**
  - 设参数维度为 `d=4`（Lotka–Volterra 中 alpha, beta, gamma, delta），则：
    - 单个 coupling 的 MLP 参数量约为：
      - 例如两层 64 宽度 + 输出层：  
        - 4 → 64: 约 4×64 + 64 ≈ 320  
        - 64 → 64: 约 64×64 + 64 ≈ 4,160  
        - 64 → 2d (scale+shift): 约 64×8 + 8 ≈ 520  
        - 合计 ≈ 5,000 左右参数 / coupler（数量级）
    - 4 个 coupling 层，总参数 ≈ 2 万数量级
  - 加上 summary network 的参数（TimeSeriesSummary 的卷积和全连接），整个 BayesFlow 模型在 10^4–10^5 量级，是一个中等规模网络。

---

**2. NPE / SNPE 在本项目中的 generative 网络结构**

相关代码：`models/sbi_wrappers.py` 中的 `run_sbi_model`。

- **Flow 类型**
  - 使用 `sbi.inference` 中的 NPE / NPE_A / NPE_B / SNPE（具体类由版本决定）。
  - 在当前 Lotka–Volterra 实现里：
    - NPE / SNPE 默认使用 **Masked Autoregressive Flow (MAF)** 或 **Neural Spline Flow (NSF)** 一类的正规化流，具体类型和层数来自 `sbi` 的默认配置。
  - 代码中我们只显式指定了：
    - `inference = NPE_A(...)` 或 `NPE_B(...)` 或 `NPE(...)`
  - 没有手动传 `density_estimator`，因此：
    - `sbi` 内部会根据 `NPE` 的默认设置构建一组 flow 层。

- **NPE 的网络配置（默认，概念层面）**
  - `sbi` 的 NPE 默认使用：
    - 多个 flow blocks（例如 5–10 个层），每层是一个自回归或耦合型变换
    - 每层内部一个小的 MLP 作为条件网络（类似 BayesFlow coupling 中的 MLP）
  - 典型的默认超参（不是在你代码里写死，而是 `sbi` 默认）大致是：
    - **隐藏宽度**：50–100
    - **隐藏层数**：2–3 层
    - **激活**：ReLU 或 Tanh
  - 具体的数字可以通过在运行时打印 `density_estimator` 的 `network` 结构得到，但从代码上看，我们没有覆盖这些默认设置。

- **本项目对 NPE 的特别之处**
  - 输入并不是原始时间序列，而是我们自己定义的 **Summary Statistics**：
    - 在 `calculate_summary_statistics` 中，我们把长度 151、维度 2 的轨迹压缩成：
      - 均值（2 维）
      - log 方差（2 维）
      - 自相关（lag=1, lag=2，各 2 维）
      - 交叉相关（1 维）
      - 合计 9 维 summary
  - Flow 的输入就是这个 9 维 summary，而输出是 4 维参数 $\theta$。
  - 因此：
    - NPE 部分的“generative network = Normalizing Flow”
    - 结构上仍然是“多层 flow + 每层内部的 MLP”，只是输入和输出维度分别是 9 和 4。

- **参数数量级**
  - 以典型配置估算（假设）：
    - 5 个 flow 层
    - 每层 MLP 宽度 50，2 层
  - 单层参数量与 BayesFlow 的估算类似，也是几千数量级 / 层；
  - 总体在 10^4–10^5 范围，与 BayesFlow 的 CouplingFlow 同一数量级。

---

**3. 对比总结：BayesFlow vs NPE 的 generative 网络**

- **共同点**
  - 都是：  
    **标准正态 $z \sim \mathcal{N}(0, I)$ → 若干可逆变换（flow blocks）→ 参数 $\theta$**
  - 每个 flow block 内都有一个小型 MLP（Dense 层）作为条件网络。
  - 参数规模都是 10^4–10^5 数量级的“中等”模型，而非超大网络。

- **差异**
  - **实现框架**：
    - BayesFlow：基于 Keras + 自己的 `bf.networks.CouplingFlow`。
    - NPE：基于 PyTorch + `sbi` 内部的 flow（MAF/NSF 等）。
  - **你代码中显式控制的部分**：
    - BayesFlow：
      - 明确指定 `num_coupling_layers=4`
      - coupling 的隐藏宽度 = 64，激活 = ReLU
    - NPE：
      - 使用 `sbi` 默认 flow 架构，未在代码中手动设定层数/宽度
      - 若要完全精确控制，需要在 `sbi` 中传入自定义 `density_estimator` 构造函数

如果你希望，我可以在运行时打印一次 `density_estimator` 的 `network` 结构，把 NPE 的具体层数、每层隐藏维度、激活函数等精确列出来，再补充一段更细致的结构表格到 Q&A 中。***


Q4. 为什么在cuda训练中会出现如下报错，是分母出的问题吗？抽出来approximate posterior中的samples得到的 prob 接近0？
Total MCMC steps: 30 (Burn-in: 29, Sampling: 1)
/mnt/Summary/LotkaVoletrra/utilities.py:65: RuntimeWarning: overflow encountered in divide
  ratio = np.divide(proposed_prob, current_prob, out=np.zeros_like(current_prob), where=current_prob!=0)

**A4:**

这个报错本质上不是「分母直接为 0」，而是「在概率空间做除法导致数值溢出」。

- 在 `utilities.py` 的 `run_mcmc_refinement` 里，原始实现是：
  - 先在概率空间计算
    - `current_prob = exp(log_prior) * likelihood`
    - `proposed_prob = exp(log_prior') * likelihood'`
  - 再做
    - `ratio = proposed_prob / current_prob`
  - 如果：
    - `current_prob` 非常接近 0（但不等于 0），例如 `1e-320`；
    - `proposed_prob` 相对没有那么小（例如 `1e-5`）；
    - 那么 `ratio ≈ 1e315`，已经接近甚至超过 `float64` 的上限，就会触发 `overflow encountered in divide`。
- 在 cuda 训练下，由于：
  - Summary network 和 generative network 的输出尺度略有不同；
  - 统计量维度较高、`epsilon` 较小时，距离平方 `dist_sq` 会很大；
  - 于是近似似然 `likelihood = exp(-dist_sq/(2*eps^2))` 会在 float32 上迅速下溢到 0（或极小值），
  - 再乘上 `exp(log_prior)`，得到的 `current_prob`/`proposed_prob` 非常极端，除法就更容易溢出。

要解决这个问题，正确做法是**完全在 log-space 里做 MH 接受率计算**，而不是先回到概率空间再相除：

- 记
  - `log π(θ) = log_prior_fn(θ)`
  - `ℓ(θ) = approximate_likelihood_core(...)` 返回的是近似似然本身
  - 则联合「目标密度」的 log 形式是
    - `log p(θ | x_obs) ∝ log π(θ) + log ℓ(θ)`
- 在代码里改成：
  - 初始：
    - `current_log_prior = log_prior_fn(current_theta)`
    - `current_likelihood = likelihood_fn(current_theta, x_obs_stats)`
    - `current_log_likelihood = log(current_likelihood + 1e-300)`
    - `current_log_prob = current_log_prior + current_log_likelihood`
  - 提案：
    - `proposed_log_prior = log_prior_fn(proposed_theta)`
    - `proposed_likelihood = likelihood_fn(proposed_theta, x_obs_stats)`
    - `proposed_log_likelihood = log(proposed_likelihood + 1e-300)`
    - `proposed_log_prob = proposed_log_prior + proposed_log_likelihood`
  - 接受率：
    - `log_ratio = proposed_log_prob - current_log_prob`
    - 抽 `u ~ U(0,1)`，计算 `log_u = log(u)`（`log_u <= 0`）
    - 接受条件写成
      - `accept_mask = log_u < log_ratio`
- 这个写法等价于原来的
  - `ratio = proposed_prob / current_prob`
  - `accept_prob = min(1, ratio)`
  - `u < accept_prob`
  但整个过程完全避免了在概率空间里做除法和 `exp(·)` 的大数，从而自然消除了 `overflow encountered in divide` 这个 warning。

目前在 `LotkaVoletrra/utilities.py` 中，我已经把 `run_mcmc_refinement` 改成了这样的 log-space 实现：

- 不再显式维护 `current_prob`、`proposed_prob`；
- 全程用 `current_log_prob`、`proposed_log_prob` 和 `log_ratio` 做 MH 接受率；
- 对极小似然加了一个 `1e-300` 的下限，避免 `log(0)` 直接变成 `-inf` 造成后续 NaN。

因此：

- 你在 cuda 训练（或者 MPS、CPU）下再跑 Refine / Refine+ 时，应该不会再看到这条 `overflow encountered in divide` 的报错；
- 即便 approximate posterior 的样本在某些区域概率极小（近似 0），MCMC 也会通过 log-space 的接受率计算稳定地处理这些情况，而不是在概率空间溢出。

Q4.1 为什么还会有这种问题？是哪一部分计算导致的这个问题？
/mnt/Summary/LotkaVoletrra/utilities.py:43: RuntimeWarning: divide by zero encountered in log
  current_log_likelihood = np.log(current_likelihood + 1e-300)
Total MCMC steps: 30 (Burn-in: 29, Sampling: 1)
/mnt/Summary/LotkaVoletrra/utilities.py:63: RuntimeWarning: divide by zero encountered in log
  proposed_log_likelihood = np.log(proposed_likelihood + 1e-300)
/mnt/Summary/LotkaVoletrra/utilities.py:67: RuntimeWarning: invalid value encountered in subtract
  log_ratio = proposed_log_prob - current_log_prob

**A4.1:**

这些 warning 的根本原因是：**近似似然 `likelihood_fn` 在数值上经常返回精确的 0（或极小值），然后在 log 空间里直接参与计算**，导致：

- 对 `current_likelihood` / `proposed_likelihood` 做 `np.log` 时触发
  - `divide by zero encountered in log`（`log(0)`）
- 这些 `-inf` 再参与减法
  - `log_ratio = proposed_log_prob - current_log_prob`
  - 当两边都是 `-inf` 时，数值上是 `(-inf) - (-inf)`，得 `NaN`，于是出现
    - `invalid value encountered in subtract`

为什么会出现大量 `likelihood == 0`？

- 近似似然的定义是：
  - `likelihood = exp(-dist^2 / (2 * eps^2))`
  - 当 summary 距离 `dist` 很大，或者带宽 `eps` 很小的时候：
    - `dist^2 / (2 * eps^2)` 会非常大
    - 在 float32 / float64 下 `exp(-非常大)` 会**下溢到 0**
- 在 cuda/MPS 或不同设备上，SummaryNetwork 的输出尺度和 eps 的估计略有变化，更容易把 `likelihood` 推到 0。

之前我加的 `+1e-300` 只是试图在 log 之前拉起一个下限，但在你那次运行中还有两点导致 warning 仍然出现：

1. `likelihood_fn` 返回的数组里存在非有限值（比如 NaN），`np.log(NaN + 1e-300)` 仍然是 NaN，不会被简单的 `+1e-300` 修复；
2. 就算 log 本身不报错，如果我们允许 `log_prob` 里保留很多 `-inf`，在做差时也会出现 `(-inf) - (-inf)` 型的无定义操作。

为彻底解决这些 warning，我在 `run_mcmc_refinement` 做了更严格的数值防护：

- 对 current/proposed likelihood，先做显式裁剪：

  ```python
  current_likelihood = likelihood_fn(current_theta, x_obs_stats)
  current_likelihood_safe = np.clip(current_likelihood, 1e-300, None)
  current_log_likelihood = np.log(current_likelihood_safe)
  ...
  proposed_likelihood = likelihood_fn(proposed_theta, x_obs_stats)
  proposed_likelihood_safe = np.clip(proposed_likelihood, 1e-300, None)
  proposed_log_likelihood = np.log(proposed_likelihood_safe)
  ```

  - `np.clip(·, 1e-300, None)` 保证传给 `log` 的值永远在 `[1e-300, +∞)`，不会再出现 `log(0)`；
  - 同时也把任何负值（如果上游产生了轻微的数值噪音）拉到一个极小正数，避免 `log(负数)` 直接变成 NaN。

- 之后仍然在 log-space 里做 MH 接受率：

  ```python
  proposed_log_prob = proposed_log_prior + proposed_log_likelihood
  log_ratio = proposed_log_prob - current_log_prob
  log_u = np.log(np.random.rand(n_chains))
  accept_mask = log_u < log_ratio
  ```

这样一来：

- `np.log` 再也不会接触到 0 或负数，不会触发 “divide by zero encountered in log”；
- `log_prob` 中不会再出现由 `log(0)` 直接产生的 `-inf`/NaN，大幅减少 `log_ratio` 中的 `invalid`；
- 即便在 cuda/MPS 训练、极端 eps 和距离组合下，MCMC 也能稳定工作（只是接受率可能会很低，但那是统计上的问题，而不是数值崩溃）。

总结一下：

- Q4 的 overflow 是「概率空间除法」导致的溢出；
- Q4.1 的 warning 是「log 空间中仍然对 0 或非有限值取 log」导致的；
- 现在 `run_mcmc_refinement` 通过 log-space + `clip` 两步，把这两类问题都从数值层面消掉了。
