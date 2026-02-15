import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from data_generation import LVTask
from models.smmd import SMMD_Model, mixture_sliced_mmd_loss
from models.mmd import MMD_Model, mmd_loss
import matplotlib.pyplot as plt
import seaborn as sns
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
REPEATS = int(os.environ.get("GRAD_REPEATS", "100"))
BATCH_SIZE = 128
M = 50
L = 20
N_TIME_STEPS = 151
def _flatten_grads(params):
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    if len(grads) == 0:
        return torch.zeros(1, device=DEVICE)
    return torch.cat(grads)
def _grad_stats_over_mc(model, task, loss_kind, repeats=REPEATS, batch_size=BATCH_SIZE):
    model.to(DEVICE)
    model.train(False)
    g_sum = None
    g_sq_sum = None
    norms = []
    for i in range(repeats):
        theta_np = task.sample_prior(batch_size, "vague")
        x_np = task.simulator(theta_np)
        theta = torch.from_numpy(theta_np).float().to(DEVICE)
        x = torch.from_numpy(x_np).float().to(DEVICE)
        z = torch.randn(x.size(0), M, model.d, device=DEVICE)
        if loss_kind == "smmd":
            theta_fake = model(x, z)
            loss = mixture_sliced_mmd_loss(theta, theta_fake, num_slices=L, n_time_steps=N_TIME_STEPS)
        else:
            theta_fake = model(x, z)
            loss = mmd_loss(theta, theta_fake, n_time_steps=N_TIME_STEPS)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        g_vec = _flatten_grads(model.G.parameters())
        norms.append(torch.norm(g_vec).item())
        if g_sum is None:
            g_sum = g_vec.clone()
            g_sq_sum = g_vec.clone() ** 2
        else:
            g_sum += g_vec
            g_sq_sum += g_vec ** 2
    mean_norm = float(np.mean(norms))
    std_norm = float(np.std(norms))
    max_norm = float(np.max(norms))
    min_norm = float(np.min(norms))
    g_mean = g_sum / repeats
    g_var = g_sq_sum / repeats - g_mean ** 2
    var_mean = float(torch.mean(g_var).item())
    var_p95 = float(torch.quantile(g_var, 0.95).item())
    var_p99 = float(torch.quantile(g_var, 0.99).item())
    return {
        "norm_mean": mean_norm,
        "norm_std": std_norm,
        "norm_min": min_norm,
        "norm_max": max_norm,
        "var_mean": var_mean,
        "var_p95": var_p95,
        "var_p99": var_p99,
        "norms": norms,
        "g_var": g_var.detach().cpu().numpy().tolist()
    }
def main():
    os.makedirs("results/grad_analysis", exist_ok=True)
    task = LVTask(t_max=(N_TIME_STEPS - 1) * 0.2, dt=0.2)
    smmd = SMMD_Model(summary_dim=10, d=4, d_x=2, n=N_TIME_STEPS)
    mmd = MMD_Model(summary_dim=10, d=4, d_x=2, n=N_TIME_STEPS)
    res_smmd = _grad_stats_over_mc(smmd, task, "smmd")
    res_mmd = _grad_stats_over_mc(mmd, task, "mmd")
    result = {"SMMD": res_smmd, "MMD": res_mmd}
    with open("results/grad_analysis/gradient_stats.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sns.set(style="whitegrid")
    plt.figure(figsize=(8,6))
    sns.kdeplot(res_smmd["norms"], label="SMMD", fill=True, alpha=0.3)
    sns.kdeplot(res_mmd["norms"], label="MMD", fill=True, alpha=0.3)
    plt.xlabel("Gradient Norm (Generator)")
    plt.ylabel("Density")
    plt.title("Gradient Norm Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/grad_analysis/grad_norm_kde.png")
    plt.close()
    plt.figure(figsize=(8,6))
    data = [res_smmd["norms"], res_mmd["norms"]]
    labels = ["SMMD", "MMD"]
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Boxplot")
    plt.tight_layout()
    plt.savefig("results/grad_analysis/grad_norm_box.png")
    plt.close()
    gv_smmd = np.array(res_smmd["g_var"])
    gv_mmd = np.array(res_mmd["g_var"])
    plt.figure(figsize=(8,6))
    sns.kdeplot(gv_smmd, label="SMMD g_var", fill=True, alpha=0.3)
    sns.kdeplot(gv_mmd, label="MMD g_var", fill=True, alpha=0.3)
    plt.xlabel("Per-parameter Gradient Variance")
    plt.ylabel("Density")
    plt.title("Per-parameter Gradient Variance Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/grad_analysis/grad_var_kde.png")
    plt.close()
    plt.figure(figsize=(8,6))
    for arr, label in [(gv_smmd, "SMMD"), (gv_mmd, "MMD")]:
        s = np.sort(arr)
        y = np.linspace(0,1,len(s))
        plt.plot(s, y, label=label)
    plt.xlabel("Per-parameter Gradient Variance")
    plt.ylabel("CDF")
    plt.title("Gradient Variance CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/grad_analysis/grad_var_cdf.png")
    plt.close()
if __name__ == "__main__":
    main()
