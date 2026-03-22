"""
Refinement Trajectory Experiment

This script visualizes the evolution of the posterior distribution and the MMD metric 
during the local refinement process of an SMMD model on the Gaussian Task.
"""

import os
import json
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Ensure backend
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

# Local imports
from data_generation import GaussianTask, generate_dataset, d, d_x
from models.smmd import SMMD_Model, sliced_mmd_loss
from utilities import (
    compute_bandwidth_smmd, 
    approximate_likelihood_core, 
    compute_mmd_metric, 
    run_gaussian_posterior_mcmc
)

# Configuration Constants (Defaults)
M = 50
L = 20
BATCH_SIZE = 256
DATASET_SIZE = 25600
LEARNING_RATE = 1e-3
L1_LAMBDA = 1e-4
NUM_ROUNDS = 20
n = 50
RESULTS_BASE = "results_trajectory"

# -----------------------------------------------------------------------------
# 1. Helper Functions
# -----------------------------------------------------------------------------

def load_config():
    """Loads configuration from config.json if available."""
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

def get_scheduler(optimizer, epochs):
    """Cosine Decay Scheduler"""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

def train_smmd(model, train_loader, epochs, device, lr=LEARNING_RATE):
    """Trains the SMMD model."""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, epochs)
    
    model.to(device)
    model.train()
    
    print(f"Starting SMMD training on {device}...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, theta_batch in train_loader:
            x_batch = x_batch.to(device)
            theta_batch = theta_batch.to(device)
            
            optimizer.zero_grad()
            
            with torch.enable_grad():
                z = torch.randn(x_batch.size(0), M, model.d, device=device)
                theta_fake = model(x_batch, z)
                
                loss = sliced_mmd_loss(theta_batch, theta_fake, num_slices=L, n_points=M)
                
                if hasattr(model, "G"):
                    l1_loss = 0.0
                    for param in model.G.parameters():
                        l1_loss = l1_loss + torch.sum(torch.abs(param))
                    loss = loss + L1_LAMBDA * l1_loss
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    print(f"Training finished in {time.time() - start_time:.2f}s")

def refine_with_trajectory_tracking(
    model, x_obs, task, true_posterior_samples,
    n_chains=1000, n_samples=500, burn_in=100, thin=1, 
    nsims=50, epsilon=None, proposal_std=0.5, 
    device="cpu", record_interval=10
):
    """
    Custom refinement loop that tracks posterior evolution and MMD metric.
    """
    print("Starting SMMD Refinement with Trajectory Tracking...")
    
    # 1. Setup Initial State (Amortized Samples)
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device)
    
    with torch.no_grad():
        # Compute summary stats for observation
        x_obs_stats = model.T(x_obs_tensor)
        
        # Sample initial chains from amortized posterior
        Z = torch.randn(n_chains, 1, task.d, device=device)
        # shape: (n_chains, d)
        current_theta = model.G(Z, x_obs_stats.expand(n_chains, -1)).squeeze(1).cpu().numpy()

    # Compute bandwidth if not provided
    if epsilon is None:
        epsilon = compute_bandwidth_smmd(model, x_obs, task, device=device)
        print(f"Computed bandwidth (epsilon): {epsilon}")

    # Define Likelihood Function Wrapper
    def likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(theta, target_stats, task, model.T, nsims, epsilon, device)

    # 2. MCMC Initialization
    # Clip to bounds
    current_theta = np.clip(current_theta, task.lower, task.upper)
    
    # Dimensional-wise Proposal Step Size
    std_per_dim = np.std(current_theta, axis=0)
    std_per_dim = np.maximum(std_per_dim, 1e-6)
    proposal_scale = std_per_dim * proposal_std
    print(f"Proposal Scale: {proposal_scale}")

    # Initial Probabilities
    current_log_prior = task.log_prior(current_theta)
    current_likelihood = likelihood_fn(current_theta, x_obs_stats)
    current_log_prob = current_log_prior + np.log(current_likelihood + 1e-5)

    # Tracking Variables
    trajectory_snapshots = [] # List of (iteration, samples)
    mmd_history = []          # List of (iteration, mmd_value)
    
    # Record initial state (Iteration 0)
    mmd_init = compute_mmd_metric(current_theta, true_posterior_samples)
    mmd_history.append((0, mmd_init))
    trajectory_snapshots.append((0, current_theta.copy()))
    
    total_steps = burn_in + n_samples # We define total iterations slightly differently here for tracking
    # Or follow run_experiment: burn_in + n_samples * thin. Let's simplify:
    # We just run N iterations total, treating early ones as burn-in implicitly in the plot or explicitly.
    # But usually refinement is short. Let's run `total_steps` iterations.
    
    # Let's align with config: usually we want `n_samples` valid samples after `burn_in`.
    # But for trajectory, we just want to see evolution.
    # Let's run for a fixed number of iterations, e.g. 500 or 1000.
    
    print(f"Running MCMC for {total_steps} steps...")
    
    accepted_count = 0
    
    for step in range(1, total_steps + 1):
        # Propose
        proposal_noise = np.random.randn(n_chains, task.d) * proposal_scale
        proposed_theta = current_theta + proposal_noise
        
        # Calculate Probabilities
        proposed_log_prior = task.log_prior(proposed_theta)
        proposed_likelihood = likelihood_fn(proposed_theta, x_obs_stats)
        proposed_log_prob = proposed_log_prior + np.log(proposed_likelihood + 1e-5)
        
        # Metropolis-Hastings Step
        log_alpha = proposed_log_prob - current_log_prob
        u = np.log(np.random.rand(n_chains))
        accept_mask = u < log_alpha
        
        # Update State
        current_theta[accept_mask] = proposed_theta[accept_mask]
        current_log_prob[accept_mask] = proposed_log_prob[accept_mask]
        
        accepted_count += np.sum(accept_mask)
        
        # Tracking
        if step % record_interval == 0:
            # Compute MMD
            mmd_val = compute_mmd_metric(current_theta, true_posterior_samples)
            mmd_history.append((step, mmd_val))
            
            # Save Snapshot (optional: can save less frequently to save memory)
            trajectory_snapshots.append((step, current_theta.copy()))
            
        if step % 50 == 0:
            acc_rate = accepted_count / (step * n_chains)
            print(f"Step {step}/{total_steps} | MMD: {mmd_history[-1][1]:.4f} | Acc Rate: {acc_rate:.2%}")

    return trajectory_snapshots, mmd_history

def plot_trajectory_and_mmd(snapshots, mmd_history, true_samples, output_dir, param_idx=0, param_name=r"$\theta_1$", x_lim=None):
    """
    Plots the MCMC trajectory density evolution and MMD metric.
    """
    # Setup Style
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'legend.fontsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })

    # Create Figure
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    plt.subplots_adjust(wspace=0.25, left=0.08, right=0.95, top=0.85, bottom=0.15)
    
    # --------------------------
    # Left Plot: Density Evolution
    # --------------------------
    ax_left = axs[0]
    
    # Custom Colormap (Light Blue -> Dark Blue)
    colors = [(0.7, 0.85, 1), (0.1, 0.3, 0.8)]
    cmap = LinearSegmentedColormap.from_list("iteration_cmap", colors, N=len(snapshots))
    
    # Plot snapshots
    # We want to show evolution. Plot a subset of snapshots to avoid clutter.
    # e.g., Start, 25%, 50%, 75%, End
    n_snaps = len(snapshots)
    indices_to_plot = np.linspace(0, n_snaps - 1, 6, dtype=int)
    
    for i, idx in enumerate(indices_to_plot):
        iteration, samples = snapshots[idx]
        progress = i / (len(indices_to_plot) - 1)
        
        label = f"Iter {iteration}" if i == 0 or i == len(indices_to_plot)-1 else None
        
        sns.kdeplot(
            samples[:, param_idx],
            ax=ax_left,
            fill=True,
            alpha=0.3 + 0.2 * progress, # Increasing opacity
            linewidth=2 + 2 * progress, # Increasing width
            color=cmap(progress),
            label=label
        )
        
    # Plot True Posterior
    sns.kdeplot(
        true_samples[:, param_idx],
        ax=ax_left,
        fill=False,
        label="True Posterior",
        color="#363336", # Dark Grey
        linestyle="-.",
        linewidth=4.0
    )
    
    ax_left.set_xlabel(param_name, fontweight='bold')
    ax_left.set_ylabel("Density", fontweight='bold')
    ax_left.set_title(f"Refinement Trajectory ({param_name})", pad=15)
    
    # Set X-limit for Posterior Plot
    if x_lim is not None:
        ax_left.set_xlim(x_lim)
    else:
        # Default fallback
        ax_left.set_xlim(0.5, 1.5)
    
    # Legend
    handles, labels = ax_left.get_legend_handles_labels()
    # Filter unique if needed, but here we controlled labels
    ax_left.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # Border
    for spine in ax_left.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # --------------------------
    # Right Plot: MMD Metric
    # --------------------------
    ax_right = axs[1]
    
    iterations = [x[0] for x in mmd_history]
    mmd_vals = [x[1] for x in mmd_history]
    
    ax_right.plot(
        iterations, mmd_vals,
        marker='o', markersize=6, color='#DB1218', linewidth=3.0, # SMMD Red
        label="SMMD Refinement"
    )
    
    ax_right.set_xlabel("Iteration", fontweight='bold')
    ax_right.set_ylabel("MMD (Median Heuristic)", fontweight='bold')
    ax_right.set_title("MMD over iteration", pad=15)
    
    # Grid
    ax_right.grid(True, linestyle='--', alpha=0.5)
    
    # Border
    for spine in ax_right.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        
    mmd_arr = np.asarray(mmd_vals, dtype=float)
    mmd_arr = mmd_arr[np.isfinite(mmd_arr)]
    if mmd_arr.size > 0:
        y_min = float(np.min(mmd_arr))
        y_max = float(np.max(mmd_arr))
        if y_min == y_max:
            pad = max(abs(y_min) * 0.1, 1e-3)
        else:
            pad = (y_max - y_min) * 0.08
        y0 = y_min - pad
        y1 = y_max + pad
        locator = mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], min_n_ticks=4)
        ticks = locator.tick_values(y0, y1)
        ax_right.set_ylim(float(ticks[0]), float(ticks[-1]))
        ax_right.yaxis.set_major_locator(locator)

    # Save
    safe_name = param_name.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    save_path = os.path.join(output_dir, f"refinement_trajectory_mmd_{safe_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

# -----------------------------------------------------------------------------
# 2. Main Execution
# -----------------------------------------------------------------------------

def main():
    # Setup
    os.makedirs(RESULTS_BASE, exist_ok=True)
    
    # Load Config (and override defaults)
    config = load_config()
    num_rounds = int(config.get("num_rounds", NUM_ROUNDS))
    n_obs = config.get("n_observation", n)
    # Refinement config
    refine_cfg = config.get("refine_config", {})
    n_chains = refine_cfg.get("n_chains", 1000)
    burn_in = refine_cfg.get("burn_in", 100)
    nsims = refine_cfg.get("nsims", 50)
    # We want a decent trajectory, so let's ensure total steps is sufficient
    # n_samples in config is usually small (e.g. 1 sample per chain after burn-in)
    # But for trajectory visualization, we want to see the burn-in phase + mixing.
    # Let's run for a fixed number of iterations for visualization purposes, 
    # overriding the standard "just get one sample" logic.
    trajectory_steps = 250 # 150 
    record_interval = 5
    
    print(f"=== Refinement Trajectory Experiment (n={n_obs}, rounds={num_rounds}) ===")
    
    # Check Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    for round_idx in range(1, num_rounds + 1):
        round_dir = os.path.join(RESULTS_BASE, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)
        print(f"\n--- Round {round_idx}/{num_rounds} ---")

        task = GaussianTask(n=n_obs)
        print("Generating training data...")
        theta_train, x_train = generate_dataset(task, n_sims=DATASET_SIZE, n_obs=n_obs)
        dataset = TensorDataset(
            torch.from_numpy(x_train).float(), torch.from_numpy(theta_train).float()
        )
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        theta_true, x_obs = task.get_ground_truth()
        print(f"True Params: {theta_true}")

        print("Sampling True Posterior (Reference)...")
        X_comp = x_obs[:, 0]
        Y_comp = x_obs[:, 1]
        Z_comp = x_obs[:, 2]
        denom = 1.0 - Z_comp
        denom[np.abs(denom) < 1e-6] = 1e-6
        x_2d = np.stack([X_comp / denom, Y_comp / denom], axis=-1)

        true_post_cfg = config.get("true_posterior_config", {})
        mcmc_draws = true_post_cfg.get("n_draws", 2000)
        mcmc_tune = true_post_cfg.get("n_tune_chain", 1000)
        mcmc_chains = true_post_cfg.get("chains", 30)
        base_scale = true_post_cfg.get("proposal_scale", 0.06)

        scale_factor = float(25.0 / n_obs) ** 0.5
        proposal_scale = base_scale * scale_factor

        print(
            f"MCMC Config: draws={mcmc_draws}, tune={mcmc_tune}, "
            f"chains={mcmc_chains}, scale={proposal_scale}"
        )

        true_posterior_samples = run_gaussian_posterior_mcmc(
            x_2d,
            task,
            n_draws=mcmc_draws,
            n_tune_chain=mcmc_tune,
            chains=mcmc_chains,
            proposal_scale=proposal_scale,
        )

        print("Initializing and Training SMMD...")
        smmd_model = SMMD_Model(summary_dim=10, d=d, d_x=d_x, n=n_obs)

        epochs_cfg = config.get("models_config", {}).get("smmd", 200)
        if isinstance(epochs_cfg, dict):
            epochs = epochs_cfg.get("epochs", 200)
        else:
            epochs = int(epochs_cfg)

        train_smmd(smmd_model, train_loader, epochs=epochs, device=device)

        snapshots, mmd_history = refine_with_trajectory_tracking(
            smmd_model,
            x_obs,
            task,
            true_posterior_samples,
            n_chains=n_chains,
            n_samples=trajectory_steps,
            burn_in=0,
            nsims=nsims,
            proposal_std=0.2,
            device=str(device),
            record_interval=record_interval,
        )

        print("Generating Visualizations...")
        params_config = [
            {"idx": 0, "name": r"$\theta_1$", "xlim": [0.5, 1.5]},
            {"idx": 1, "name": r"$\theta_2$", "xlim": [0.5, 1.5]},
            {"idx": 2, "name": r"$\theta_3$", "xlim": [-1.7, 1.7]},
            {"idx": 4, "name": r"$\theta_5$", "xlim": [0.0, 1.2]},
        ]

        for p_cfg in params_config:
            plot_trajectory_and_mmd(
                snapshots,
                mmd_history,
                true_posterior_samples,
                round_dir,
                param_idx=p_cfg["idx"],
                param_name=p_cfg["name"],
                x_lim=p_cfg["xlim"],
            )

        np.save(os.path.join(round_dir, "mmd_history.npy"), np.array(mmd_history))

    print("Experiment Complete.")

if __name__ == "__main__":
    main()
