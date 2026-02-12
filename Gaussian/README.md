# Gaussian Experiment Documentation

本文档详细介绍了 Gaussian 实验的各个环节，包括数据生成机制、先验设置、真实参数、实验模型以及后验推断与 Refinement 流程。

## 1. Data Generation Process

Gaussian 任务是一个经典的基准测试问题，旨在测试模型推断多维高斯分布参数的能力。为了增加难度，我们将生成的 2D 数据投影到了 3D 球面上。

### 模拟器 (Simulator)

数据生成分为两个步骤：

1.  **2D Gaussian Sampling**:
    首先生成标准的二维高斯数据 $(X, Y)$。对于每一组参数 $\theta$：
    $$
    \begin{pmatrix} X \\ Y \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} m_0 \\ m_1 \end{pmatrix}, \begin{pmatrix} s_0^2 & r s_0 s_1 \\ r s_0 s_1 & s_1^2 \end{pmatrix} \right)
    $$
    其中参数 $\theta$ 与物理参数的映射关系如下：
    *   均值: $m_0 = \theta_0, \quad m_1 = \theta_1$
    *   标准差: $\sigma_x = s_0 = \theta_2^2, \quad \sigma_y = s_1 = \theta_3^2$ (注意这里使用了平方变换保证非负)
    *   相关系数: $\rho = r = \tanh(\theta_4)$ (使用 tanh 保证在 $(-1, 1)$ 之间)

2.  **Stereographic Projection (3D Projection)**:
    为了模拟更复杂的流形结构，我们将 2D 平面上的点 $(X, Y)$ 通过逆球极投影（Inverse Stereographic Projection）映射到 3D 单位球面上，得到观测数据 $(X', Y', Z')$：
    $$
    X' = \frac{2X}{1 + X^2 + Y^2}, \quad Y' = \frac{2Y}{1 + X^2 + Y^2}, \quad Z' = \frac{-1 + X^2 + Y^2}{1 + X^2 + Y^2}
    $$
    模型接收的观测数据即为这些 3D 坐标点集。

    **为什么要进行 3D 投影？**
    对于标准的 2D 高斯分布，其充分统计量（Sufficient Statistics）是显而易见的（即样本均值和样本协方差）。如果我们直接使用 2D 数据，神经网络很容易直接“猜”出这些统计量，从而使推断任务变得过于简单。通过引入非线性的立体投影，我们破坏了原始数据的简单统计特性，**防止模型显式地构造出对应的 summary statistics**，从而强制模型学习更复杂的特征表示。

## 2. Priors (Uniform & Conditional)

我们在代码中（`data_generation.py`）实现了两种类型的先验分布，可以通过 `GaussianTask(prior_type=...)` 进行选择：

### 1. Uniform Prior (默认)
这是最常用的设置，参数之间相互独立：
*   **分布**: $\theta_i \sim U(-3, 3)$ for $i=0, \dots, 4$
*   **范围**:
    *   均值 $m_0, m_1 \in [-3, 3]$
    *   标准差 $\sigma_x, \sigma_y \in [0, 9]$
    *   相关系数 $\rho \in [\tanh(-3), \tanh(3)] \approx [-0.995, 0.995]$

### 2. Conditional Prior
这是一个更具挑战性的设置，引入了参数间的依赖关系，测试模型处理相关先验的能力：
*   **依赖关系**: 第二个参数 $\theta_1$ (对应 $m_1$) 依赖于第一个参数 $\theta_0$ (对应 $m_0$)。
*   **构造方式**: 
    $$ \theta_1 = \theta_0^2 + 0.1 \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1) $$
*   **其他参数**: $\theta_0, \theta_2, \theta_3, \theta_4$ 依然保持 $U(-3, 3)$ 分布（但在计算 log_prior 时需注意边界截断）。
*   这种先验使得参数空间具有很强的非线性相关结构（抛物线形状），增加了推断难度。

## 3. 真实参数设置 (Ground Truth)

为了评估模型性能，我们设定了一组真实的参数值用于生成观测数据（Observation）：

*   **Parameter Values**: $\theta_{true} = [1.0, 1.0, -1.0, -0.9, 0.6]$
*   **Physical Interpretation**:
    *   $m_0 = 1.0$
    *   $m_1 = 1.0$
    *   $\sigma_x = (-1.0)^2 = 1.0$
    *   $\sigma_y = (-0.9)^2 = 0.81$
    *   $\rho = \tanh(0.6) \approx 0.537$

## 4. 实验模型与 Refinement 策略

### 支持模型
实验支持多种神经推断方法，通过 `config.json` 配置：
*   **SMMD**: Score-based Maximum Mean Discrepancy
*   **MMD**: Maximum Mean Discrepancy
*   **BayesFlow**: Invertible Neural Network (基于流的生成模型)
*   **DNNABC**: Deep Neural Network ABC (学习汇总统计量 + 拒绝采样)
*   **W2ABC**: Wasserstein-ABC (基于 SMC 的无似然推断)
*   **SNPE-A**: Sequential Neural Posterior Estimation (via `sbi`)

### Refinement 策略 (Local MCMC Refinement)
与 Lotka-Volterra 中的 "Refine+" (Sequential Training) 不同，Gaussian 实验采用的是 **测试时 (Test-time) MCMC Refinement**。

*   **目的**: 利用训练好的模型（主要是其学到的 Summary Statistics 或 Density）在特定观测值 $x_{obs}$ 附近进一步校准后验样本。
*   **方法**:
    1.  **Amortized Proposal**: 首先从训练好的模型中采样，作为 MCMC 的初始状态或 Proposal。
    2.  **Approximate Likelihood**: 构建一个基于核的近似似然函数：
        $$ L(\theta) \propto \exp\left(-\frac{\|T(x_{obs}) - T(x_{sim}(\theta))\|^2}{2\epsilon^2}\right) $$
        其中 $T(\cdot)$ 是模型学到的汇总统计量网络（Summary Network），$\epsilon$ 是自动计算的带宽。
    3.  **MCMC Sampling**: 使用 Metropolis-Hastings 算法从 $p(\theta) \cdot L(\theta)$ 中采样，得到 Refined Posterior。
*   **适用性**: 主要应用于 SMMD, MMD 和 BayesFlow。DNNABC 和 SNPE-A 通常跳过此步骤。

## 5. run_experiment 流程

`run_experiment.py` 是实验的主入口，执行流程如下：

1.  **独立多轮实验 (Independent Rounds)**:
    脚本中的 `NUM_ROUNDS` 循环代表**独立重复实验**（Independent Repetitions），用于计算统计方差。每一轮都会：
    *   重新生成全新的训练数据集（从先验采样）。
    *   重新生成 Ground Truth 观测数据。
    *   *注意：这里没有跨轮次的数据积累或序贯学习（Sequential Learning），除了 SNPE-A 内部自洽的序贯过程。*

2.  **单轮流程**:
    *   **Data Generation**: 生成参考表 $(\theta, x)$。
    *   **PyMC Reference**: 运行 MCMC (NUTS) 获取真实的参考后验样本（Ground Truth Posterior）。
    *   **Model Training**: 训练选定的神经推断模型。
    *   **Amortized Inference**: 使用训练好的模型快速生成后验样本。
    *   **Refinement (Optional)**: 对样本进行 MCMC Refinement。
    *   **Evaluation**: 计算 MMD 指标（对比 PyMC 参考样本），绘制 Pair Plot 和 Loss 曲线。

3.  **结果存储**:
    *   所有结果（Loss, Samples, Plots, Metrics）均按轮次存储在 `results/` 目录下。
    *   最终生成 `summary_mean.csv` 和 `summary_median.csv` 汇总各模型的平均性能。

## 6. SMMD Sequential Learning

在 Gaussian 实验的当前配置中，**未启用** SMMD 的序贯学习（Sequential Learning / Refine+ Training）。
*   SMMD 模型在每一轮中均在从先验生成的固定数据集上进行训练（Amortized Training）。
*   若需要序贯学习能力，目前仅有 **SNPE-A** 模型通过 `sbi` 库内置的 `simulate_for_sbi` 实现了多轮序贯推断。
