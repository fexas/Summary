import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_generation import GaussianTask
from utilities import run_gaussian_posterior_mcmc


CONFIG_PATH = "config.json"
try:
    with open(CONFIG_PATH, "r") as f:
        _CONFIG = json.load(f)
except FileNotFoundError:
    _CONFIG = {}

_TPC = _CONFIG.get("true_posterior_config", {})
BASE_PROPOSAL_SCALE = _TPC.get("proposal_scale", 0.12)


def inverse_stereo_projection(x_obs_3d):
    x_obs_3d = np.asarray(x_obs_3d, dtype=np.float32)
    if x_obs_3d.ndim != 2 or x_obs_3d.shape[1] != 3:
        raise ValueError(f"Expected x_obs_3d shape (n, 3), got {x_obs_3d.shape}")
    x_comp = x_obs_3d[:, 0]
    y_comp = x_obs_3d[:, 1]
    z_comp = x_obs_3d[:, 2]
    denom = 1.0 - z_comp
    denom[np.abs(denom) < 1e-6] = 1e-6
    x_2d = x_comp / denom
    y_2d = y_comp / denom
    return np.stack([x_2d, y_2d], axis=-1)


def plot_posterior(samples, theta_true, save_path):
    d = samples.shape[1]
    fig, axes = plt.subplots(d, 1, figsize=(6, 2.5 * d))
    if d == 1:
        axes = [axes]
    for i in range(d):
        ax = axes[i]
        sns.histplot(samples[:, i], kde=True, stat="density", ax=ax, color="C0")
        ax.axvline(theta_true[i], color="red")
        ax.set_xlabel(f"theta[{i}]")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def run_mcmc_for_n(n_obs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    task = GaussianTask(n=n_obs)
    theta_true, x_obs_3d = task.get_ground_truth()
    x_obs_2d = inverse_stereo_projection(x_obs_3d)
    scale_factor = float(25.0 / n_obs) ** 0.5
    proposal_scale = BASE_PROPOSAL_SCALE * scale_factor
    samples = run_gaussian_posterior_mcmc(
        x_obs_2d,
        task,
        n_draws=1,
        n_tune_chain=5000,
        chains=5000,
        proposal_scale=proposal_scale,
    )
    save_path = os.path.join(output_dir, f"mcmc_posterior_n{n_obs}.png")
    plot_posterior(samples, theta_true, save_path)


def main():
    base_dir = os.path.join("results", "test_mcmc")
    for n_obs in [25, 50, 100]:
        out_dir = os.path.join(base_dir, f"n_{n_obs}")
        run_mcmc_for_n(n_obs, out_dir)


if __name__ == "__main__":
    main()
