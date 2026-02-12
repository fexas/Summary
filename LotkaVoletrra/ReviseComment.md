# 修改建议

### Refine + 方法

（相较于原SliceMMD，去除掉SliceMMD计算中weights的部分）

记Summary Network为 $S$，conditional generative network为 $W$（以 Summary Network的输出为condtion，在我们的model中对应SMMD、MMD和Bayesflow的结构）。在获得SMMD、MMD和Bayesflow估计的approximate posterior $\mu_{\tilde{W}}(\cdot|\hat{S})$ 后，从中抽取 $N$ 个样本 $\{\tilde{\theta}_j\}_{j=1}^N$，并获取对应的模拟数据 $\{\tilde{X}_j^{(n)}\}_{j=1}^N$。将其与原始训练数据$\{\theta_j\}_{j=1}^N$和 $\{X_j^{(n)}\}_{j=1}^N$拼接为新的训练数据。

最小化损失函数
$$
\frac{1}{2N}\sum_{j=1}^{N}\ell\left(S, W,\widetilde{\theta}_{j},\widetilde{X}_{j}^{(n)}\right)+ \frac{1}{2N}\sum_{j=1}^{N}\ell\left(S, W,\theta_{j},X_{j}^{(n)}\right)(1)
$$
得到 $\mu_{\tilde{W}_{\text{new}}}(d\theta|\tilde{S}_{\text{new}})$ 。因为此时 target posterior变成 $\tilde{p}(\theta|X^{(n)}) \propto [0.5\mu_{\tilde{W}_{\text{old}}}(\theta|\tilde{S}_{\text{old}})+0.5\pi(\tilde{\theta})] \times p(X^{(n)}|\theta)$， 为了重新校准维原target posterior。我们从中$\mu_{\tilde{W}_{\text{new}}}(d\theta|\tilde{S}_{\text{new}})$中抽样，每个样本赋予 $\frac{\pi(\tilde{\theta})}{0.5\mu_{\tilde{W}_{\text{old}}}(\theta|\tilde{S}_{\text{old}})+0.5\pi(\tilde{\theta})}$ 的权重。依照权重新抽样，使用重抽样的样本点作为ABC-MCMC的起始点， 结合新的 $\tilde{S}_{\text{new}}$ 和 $\tilde{W}_{\text{new}}$ 进行 ABC-MCMC。


请将ABC-MCMC单独提取成一个函数，方便我比较有上述sequential training process和没有sequential training 的ABC-MCMC效果 (前者`sequential_training+ABCMCMC` 称作refine+ method，后者直接`ABCMCMC`称其为refine method）。在实验过程中，请分别记录两种方法的评价指标，并绘制原approximate posterior和refine及refine+的对比图。 

注意：原来的Refine+方法只可以应用于SMMD和MMD，不能应用于Bayesflow。现在的方法可以应用于Bayesflow了，所以请将Bayesflow的Refine+方法也包含在实验中。Refine的种类和SMMD及MMD对齐。