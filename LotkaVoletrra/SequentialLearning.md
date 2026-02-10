### 1. 为SMMD，MMD和Bayesflow添加Sequential training， 以提高表现（称为 refined+）

第一种方法：

记Summary Network为 $S$，conditional generative network为 $W$（以 Summary Network的输出为condtion，在我们的model中对应SMMD、MMD和Bayesflow的结构）。在获得SMMD、MMD和Bayesflow估计的approximate posterior
$\mu_{\tilde{W}}(\cdot|\hat{S})$ 后，从中抽取 $N$ 个样本 $\{\tilde{\theta}_j\}_{j=1}^N$，并获取对应的模拟数据 $\{\tilde{X}_j^{(n)}\}_{j=1}^N$。以新抽取的数据为训练数据，重新训练一遍，然后最小化以下损失函数：

$$
\frac{1}{N}\sum_{j=1}^{n}\ell\left(S, W,\widetilde{\theta}_{j},\widetilde{X}_{j}^{(n)}\right)\cdot\omega_{j},\qquad(1)
$$

其中 $\omega_j$ 是密度比 $\frac{\pi(\tilde{\theta}_j)}{\mu_{\tilde{W}}(\theta_j|\tilde{S})}$（对于 MMD 和 SMMD，$\mu_{\tilde{W}}(\tilde{\theta}_j|\tilde{S})$ 替换为由 $\{\tilde{\theta}_j\}_{j=1}^N$ 估计的核密度估计）。然后使用新的 $\tilde{S}_{\text{new}}$ 和 $\tilde{W}_{\text{new}}$ 进行 ABC-MCMC。

第二种方法：

将最初的训练数据集和新的数据集合并（要保证两者的数量相同），并使用合并后的数据集重新训练 $S$ 和 $W$。此时，损失函数（1）中的 $\omega_j$ 可以取为 $\frac{\pi(\tilde{\theta}_j)}{0.5\mu_{\tilde{W}}(\theta_j|\tilde{S})+0.5\pi(\tilde{\theta}_j)}$。然后使用新的 $\tilde{S}_{\text{new}}$ 和 $\tilde{W}_{\text{new}}$ 进行 ABC-MCMC。

ABC-MCMC的步骤请参考Gaussian文件夹中run_experiment时做refine_posterior的步骤。在每轮实验执行完"refine+"后，在得到的新的posterior的时候也要计算对应的那些指标，绘制对应的后验图（重新训练得到的posterior图和ABC-MCMC得到的posterior图都要画）。

实验结果添加记录每种方法每轮的训练时间（SMMD，MMD和Bayesflow，训练，重新训练和ABC-MCMC的时间分开记录，再记录一个refine+的整体时间），在所有轮次结束后计算对应的平均训练时间。DNNABC和SPNE记录每轮的训练时间和平均训练时间。



