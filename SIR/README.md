# SIR 模型的 Gillespie 数据生成与观测说明

本说明介绍我们如何用 Gillespie 随机模拟方法生成 SIR（易感-感染-恢复）模型的数据，以及如何在离散时间点上进行观测并保存结果。

## 模型概述
- 状态变量：
  - S：易感人数（Susceptible）
  - I：感染人数（Infected）
  - R：恢复人数（Recovered）
  - 总人口 N = S + I + R（恒定）
- 参数：
  - β（beta）：接触传播率
  - γ（gamma）：恢复率
- 我们在代码中以批量方式处理参数向量 θ = (β, γ)，即一次模拟多组参数。

## Gillespie 随机模拟的核心思想
- SIR 是连续时间的跳跃过程，系统状态在事件发生时改变，两个事件：
  1) 感染事件：一名易感者被感染（S→S-1，I→I+1）
  2) 恢复事件：一名感染者恢复（I→I-1，R→R+1）
- 事件速率（倾向函数）：
  - 感染速率 a_inf = β * S * I / N
  - 恢复速率 a_rec = γ * I
  - 总速率 a0 = a_inf + a_rec
- 单步模拟：
  - 计算 a_inf、a_rec、a0；若 a0=0 或 I=0，过程停止（没有可发生事件）
  - 等待时间 τ ~ Exp(a0)，将当前时间 t ← t + τ
  - 以概率 a_inf/a0 选择“感染事件”，以概率 a_rec/a0 选择“恢复事件”
  - 根据事件类型更新 S、I、R
  - 重复直到达到最大时间 t_max 或所有观测时间点都已记录

## 批量与向量化
- 代码对多个参数批次同时模拟，维护每个批次的 S、I、R、t 以及目标观测索引。
- 这样可以在 NumPy 下高效地并行更新多个轨迹，显著提高速度。

## 观测方式（离散时间网格）
- 定义均匀的时间网格 OBS_TIMES = [0, dt, 2dt, …, t_max]
- 在模拟时，当某个批次的连续时间 t 超过“下一个观测点”的时间，就把当前状态写入该观测点。
- 输出观测为整数计数，形状为 (batch_size, n_obs, 3)，三列分别对应 S、I、R。

## 先验与示例
- 先验采样：β ~ Uniform[0, 3]，γ ~ Uniform[0, 1]
- 示例脚本会运行两类实验：
  - 固定参数（如 β=1, γ=0.5）的多条轨迹
  - 从先验中抽样 1000 组参数的批量轨迹
- 绘图文件保存到 `SIR/results/`：
  - `sir_fixed_traj.png`：S/I/R 轨迹，并在图例中标注每条曲线对应的 (β, γ)
  - `sir_prior_traj.png`：随机先验采样得到的部分轨迹（显示 I 曲线）
  - `sir_outbreak_dist.png`：各条轨迹在最终时间的 R 分布（总感染规模）

## 与代码对应
- 数据生成与观测逻辑集中在 [data_generation.py](file:///Users/liruismac/Documents/trae_projects/Summary/SIR/data_generation.py)：
  - `SIRTask.simulator(...)`：执行批量 Gillespie 模拟并在 `OBS_TIMES` 上记录观测
  - `SIRTask.sample_prior(...)`：按均匀先验采样参数
  - `plot_trajectories(...)`：绘制 S/I/R 轨迹（含 (β, γ) 图例）
  - `plot_outbreak_distribution(...)`：绘制最终 R 的直方图

## 如何运行
- 在项目根或 `Gaussian` 目录下执行：
```
python ../SIR/data_generation.py
```
- 结果会保存到 `SIR/results/` 目录。
