"""
Run Refine+ Validation Experiment for Lotka-Volterra Task.
Focus: Validate improvement across Amortized -> Sequential Training -> Refine+ (MCMC).
"""

import os
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde

# Fix for MPS 'aten::linalg_qr.out' not implemented error
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ensure the backend is set
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

# Import from local modules
from data_generation import LVTask
from models.smmd import SMMD_Model, sliced_mmd_loss
from utilities import compute_metrics, refine_posterior

# ============================================================================
# 1. Configuration & Global Parameters
# ============================================================================

# EXPERIMENT_ROUNDS: Number of experiment repetitions to average over
EXPERIMENT_ROUNDS = 10

# Load Config
try:
    with open("config_rss.json", "r") as f:
        CONFIG = json.load(f)
    print("✅ Loaded configuration from config_rss.json")
except FileNotFoundError:
    print("⚠️ config_rss.json not found. Using default configuration.")
    # Fallback default
    CONFIG = {
        "dataset_size": 25600,
        "batch_size": 128,
        "models_config": {
            "smmd": {
                "epochs": 200,
                "summary_dim": 8,
                "refined_mode": 1,
                "num_refine_rounds": 1
            }
        },
        "smmd_mmd_config": {
            "M": 50, 
            "L": 20, 
            "initial_train_samples": 12800,
            "refine_train_samples": 12800,
            "bandwidth_n_samples": 2000
        },
        "n_time_steps": 151,
        "dt": 0.2,
        "learning_rate": 1e-3,
        "n_samples_posterior": 1000
    }

# Hyperparameters
DATASET_SIZE = CONFIG.get("dataset_size", 25600)
BATCH_SIZE = CONFIG.get("batch_size", 128)
MODELS_CONFIG = CONFIG.get("models_config", {})
SMMD_MMD_CONFIG = CONFIG.get("smmd_mmd_config", {"M": 50, "L": 20})

# SMMD specific params
M = SMMD_MMD_CONFIG.get("M", 50)
L = SMMD_MMD_CONFIG.get("L", 20)
SMMD_INITIAL_TRAIN_SAMPLES = SMMD_MMD_CONFIG.get("initial_train_samples", DATASET_SIZE)
SMMD_REFINE_TRAIN_SAMPLES = SMMD_MMD_CONFIG.get("refine_train_samples", DATASET_SIZE)
BANDWIDTH_N_SAMPLES = SMMD_MMD_CONFIG.get("bandwidth_n_samples", 4400)

# Task params
N_TIME_STEPS = CONFIG.get("n_time_steps", 151)
DT = CONFIG.get("dt", 0.2)
LEARNING_RATE = CONFIG.get("learning_rate", 1e-3)
N_SAMPLES_POSTERIOR = CONFIG.get("n_samples_posterior", 1000)

if os.environ.get("LV_RSS_QUICK_TEST") == "1":
    EXPERIMENT_ROUNDS = 1
    DATASET_SIZE = min(int(DATASET_SIZE), 512)
    N_SAMPLES_POSTERIOR = min(int(N_SAMPLES_POSTERIOR), 200)
    model_conf_smmd = dict(MODELS_CONFIG.get("smmd", {}))
    model_conf_smmd["epochs"] = min(int(model_conf_smmd.get("epochs", 200)), 20)
    MODELS_CONFIG = dict(MODELS_CONFIG)
    MODELS_CONFIG["smmd"] = model_conf_smmd
    BATCH_SIZE = min(int(BATCH_SIZE), 128)
    SMMD_INITIAL_TRAIN_SAMPLES = min(int(SMMD_INITIAL_TRAIN_SAMPLES), DATASET_SIZE)
    SMMD_REFINE_TRAIN_SAMPLES = min(int(SMMD_REFINE_TRAIN_SAMPLES), DATASET_SIZE)
    BANDWIDTH_N_SAMPLES = min(int(BANDWIDTH_N_SAMPLES), 1000)

# Task Dimensions
d = 4 # alpha, beta, gamma, delta
d_x = 2 # Prey, Predator
n_obs = N_TIME_STEPS

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ============================================================================
# 2. Helper Functions
# ============================================================================

def get_scheduler(optimizer, epochs):
    """Cosine Decay Scheduler"""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

def train_smmd(model, train_loader, epochs, device, n_time_steps=151):
    """Training loop for SMMD."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(optimizer, epochs)
    
    model.to(device)
    model.train()
    
    # L1 Penalty Factor
    L1_LAMBDA = 1e-4
    
    print(f"Starting SMMD training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            theta_batch = batch[1].to(device)
            
            optimizer.zero_grad()
            
            with torch.enable_grad():
                # Sample Z: (batch, M, d)
                z = torch.randn(x_batch.size(0), M, model.d, device=device)
                
                # Forward pass
                theta_fake = model(x_batch, z)
                
                loss = sliced_mmd_loss(theta_batch, theta_fake, num_slices=L, n_time_steps=n_time_steps)
                
                # Add L1 Penalty to Generator weights
                if hasattr(model, 'G'):
                    l1_loss = 0.0
                    for param in model.G.parameters():
                        l1_loss += torch.sum(torch.abs(param))
                    loss = loss + L1_LAMBDA * l1_loss
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f}s")
    return training_time

def plot_three_stage_comparison(samples_dict, theta_true, output_dir, round_idx):
    """
    Plots Amortized vs Sequential vs Refined+ (MCMC) posterior comparison.
    samples_dict: { 'Amortized': samples, 'Sequential': samples, 'Refined+ (MCMC)': samples }
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"smmd_refine_plus_comparison_round_{round_idx}.png"
    
    # Style settings
    sns.set_theme(style="whitegrid", font_scale=1.5)
    
    num_params = next(iter(samples_dict.values())).shape[1]
    cols = num_params
    
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    
    if cols == 1: axes = [axes]
    
    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24
    
    # Define colors/styles for stages
    # Distinct colors: Amortized (Blue), Sequential (Orange), Refined+ (Green)
    # True Value: Red solid
    styles = {
        'Amortized': {'color': '#1f77b4', 'linestyle': '--', 'label': 'Amortized', 'alpha': 1.0, 'linewidth': 3.5, 'fill': False},
        'Sequential': {'color': '#ff7f0e', 'linestyle': '-.', 'label': 'Sequential', 'alpha': 1.0, 'linewidth': 3.5, 'fill': False},
        'Refined+ (MCMC)': {'color': '#2ca02c', 'linestyle': '-', 'label': 'Refined+ (MCMC)', 'alpha': 0.2, 'linewidth': 4.5, 'fill': True},
    }
    
    for i in range(cols):
        ax = axes[i]
        param_symbol = rf'$\log(\theta_{{{i+1}}})$'
        
        for label, samples in samples_dict.items():
            if samples is not None:
                style = styles.get(label, {})
                sns.kdeplot(
                    x=np.asarray(samples[:, i], dtype=float), 
                    ax=ax, 
                    fill=style.get('fill', False), 
                    alpha=style.get('alpha', 1.0),
                    linewidth=style.get('linewidth', 2.5),
                    label=style.get('label', label),
                    color=style.get('color'),
                    linestyle=style.get('linestyle', '-')
                )
            
        if i < len(theta_true):
            ax.axvline(x=float(theta_true[i]), color='#FA0101', linestyle='-', linewidth=4.5, label='True Value')
            
        ax.set_title(f'{param_symbol}', fontsize=title_fontsize, fontweight='bold', pad=15)
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize, width=2, length=6)

        x_limits = [
            (-7.0, -3.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-7.0, -3.0),
        ]
        if i < len(x_limits):
            ax.set_xlim(x_limits[i])
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
            
        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight='bold')
        else:
            ax.set_ylabel("")
            
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.0)
        
    # Legend handling
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Desired order
    order = ["Amortized", "Sequential", "Refined+ (MCMC)", "True Value"]
    sorted_handles = []
    sorted_labels = []
    
    for l in order:
        if l in by_label:
            sorted_handles.append(by_label[l])
            sorted_labels.append(l)
            
    fig.legend(
        sorted_handles,
        sorted_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.08),
        ncol=len(sorted_handles),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5
    )
        
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {os.path.join(output_dir, filename)}")

# ============================================================================
# 3. Experiment Logic
# ============================================================================

def run_experiment_round(round_idx, task, x_obs, theta_true, train_loader, x_train, theta_train):
    """
    Runs a single round of the experiment:
    1. Amortized Training & Sampling
    2. Sequential Training & Sampling
    3. MCMC Refinement & Sampling
    """
    print(f"\n{'='*10} Round {round_idx} {'='*10}")
    
    results = {
        "round": round_idx,
        "metrics": {},
        "samples": {}
    }
    
    model_conf = MODELS_CONFIG.get("smmd", {})
    epochs = model_conf.get("epochs", 200)
    summary_dim = model_conf.get("summary_dim", 8)
    
    # --- Stage 1: Amortized ---
    print("\n--- Stage 1: Amortized Training ---")
    
    # Initialize Model
    n_obs_curr = x_obs.shape[0]
    model = SMMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs_curr)
    
    # Train Amortized
    train_smmd(model, train_loader, epochs, DEVICE, n_time_steps=n_obs_curr)
    
    # Sample Amortized
    print("Sampling Amortized Posterior...")
    theta_amortized = model.sample_posterior(x_obs, N_SAMPLES_POSTERIOR)
    if isinstance(theta_amortized, torch.Tensor):
        theta_amortized = theta_amortized.cpu().numpy()
        
    # Metrics
    metrics_amortized = compute_metrics(theta_amortized, theta_true)
    results["metrics"]["amortized"] = metrics_amortized
    results["samples"]["Amortized"] = theta_amortized
    print(f"Amortized Bias L2: {metrics_amortized['bias_l2']:.4f}")
    
    # --- Stage 2: Sequential Training ---
    print("\n--- Stage 2: Sequential Training (Refined+) ---")
    
    # Prepare Sequential Data
    N_new = SMMD_REFINE_TRAIN_SAMPLES
    
    # 1. Sample from current posterior (Amortized)
    theta_new = model.sample_posterior(x_obs, N_new)
    if isinstance(theta_new, torch.Tensor):
        theta_new = theta_new.cpu().numpy()
        
    # 2. Simulate new data
    x_new = task.simulator(theta_new)
    
    # 3. Reuse prior data (from train_loader/x_train)
    # We use x_train, theta_train directly which contains all data
    # Subsample to match N_new if needed or use defined initial samples
    n_initial = SMMD_INITIAL_TRAIN_SAMPLES
    if len(theta_train) > n_initial:
        indices = np.random.choice(len(theta_train), size=n_initial, replace=False)
        theta_old = theta_train[indices]
        x_old = x_train[indices]
    else:
        theta_old = theta_train
        x_old = x_train
        
    # If N_new differs from len(theta_old), we might want to balance. 
    # Current logic: concatenate.
    
    # Fit KDE on theta_new for later resampling weights
    kde_q_train_model = gaussian_kde(theta_new.T)
    
    # Combine Data
    theta_combined = np.concatenate([theta_old, theta_new], axis=0)
    x_combined = np.concatenate([x_old, x_new], axis=0)
    
    # Retrain
    print(f"Retraining on combined dataset (N={len(theta_combined)})...")
    ds_retrain = TensorDataset(torch.from_numpy(x_combined).float(), torch.from_numpy(theta_combined).float())
    loader_retrain = DataLoader(ds_retrain, batch_size=BATCH_SIZE, shuffle=True)
    
    train_smmd(model, loader_retrain, epochs, DEVICE, n_time_steps=n_obs_curr)
    
    # Sample Sequential (Mixture)
    # Note: Sequential posterior is defined as the resampled posterior from the retrained model
    print("Sampling Sequential Posterior (Resampled)...")
    theta_retrained = model.sample_posterior(x_obs, N_SAMPLES_POSTERIOR)
    if isinstance(theta_retrained, torch.Tensor):
        theta_retrained = theta_retrained.cpu().numpy()
        
    # Calculate Weights for Resampling
    kde_prob_q_train = kde_q_train_model(theta_retrained.T)
    log_prior_retrained = task.log_prior(theta_retrained)
    prior_prob_retrained = np.exp(log_prior_retrained)
    
    denom = 0.5 * kde_prob_q_train + 0.5 * prior_prob_retrained
    weights_resample = prior_prob_retrained / (denom + 1e-10)
    weights_resample = np.nan_to_num(weights_resample, nan=0.0, posinf=0.0)
    if np.sum(weights_resample) < 1e-9:
        weights_resample = np.ones_like(weights_resample) / len(weights_resample)
    else:
        weights_resample /= np.sum(weights_resample)
        
    # Resample
    indices = np.random.choice(len(theta_retrained), size=N_SAMPLES_POSTERIOR, replace=True, p=weights_resample)
    theta_sequential = theta_retrained[indices]
    
    # Metrics
    metrics_sequential = compute_metrics(theta_sequential, theta_true)
    results["metrics"]["sequential"] = metrics_sequential
    results["samples"]["Sequential"] = theta_sequential
    print(f"Sequential Bias L2: {metrics_sequential['bias_l2']:.4f}")
    
    # --- Stage 3: MCMC Refinement ---
    print("\n--- Stage 3: MCMC Refinement ---")
    
    mcmc_burn_in = model_conf.get("mcmc_burn_in", 29) # From config or default
    
    # Use Sequential samples as initialization
    theta_init_mcmc = theta_sequential
    
    # Refine
    theta_refined = refine_posterior(
        model, x_obs, task=task, 
        n_chains=N_SAMPLES_POSTERIOR, 
        n_samples=1, 
        burn_in=mcmc_burn_in, 
        device=DEVICE, 
        theta_init=theta_init_mcmc,
        bandwidth_n_samples=BANDWIDTH_N_SAMPLES
    )
    
    # Metrics
    metrics_refined = compute_metrics(theta_refined, theta_true)
    results["metrics"]["refined"] = metrics_refined
    results["samples"]["Refined+ (MCMC)"] = theta_refined
    print(f"Refined+ Bias L2: {metrics_refined['bias_l2']:.4f}")
    
    return results

def main():
    os.makedirs("results_refine_plus", exist_ok=True)
    os.makedirs("results_refine_plus/plots", exist_ok=True)
    os.makedirs("results_refine_plus/tables", exist_ok=True)
    
    print(f"Starting Refine+ Validation Experiment (Rounds={EXPERIMENT_ROUNDS})")
    
    aggregated_metrics = []
    
    for round_idx in range(1, EXPERIMENT_ROUNDS + 1):
        # Data Generation (Same per round)
        dt = DT
        t_max = (N_TIME_STEPS - 1) * dt
        task = LVTask(t_max=t_max, dt=dt)
        
        print(f"\nGenerating data for Round {round_idx}...")
        theta_train = task.sample_prior(DATASET_SIZE, "vague")
        x_train = task.simulator(theta_train)
        
        x_tensor = torch.from_numpy(x_train).float()
        theta_tensor = torch.from_numpy(theta_train).float()
        
        smmd_n = min(SMMD_INITIAL_TRAIN_SAMPLES, DATASET_SIZE)
        dataset_smmd = TensorDataset(x_tensor[:smmd_n], theta_tensor[:smmd_n])
        train_loader = DataLoader(dataset_smmd, batch_size=BATCH_SIZE, shuffle=True)
        
        # Ground Truth
        theta_true, x_obs = task.get_ground_truth()
        
        # Run Experiment
        res = run_experiment_round(round_idx, task, x_obs, theta_true, train_loader, x_train[:smmd_n], theta_train[:smmd_n])
        
        # Plotting
        plot_three_stage_comparison(
            res["samples"], 
            theta_true, 
            "results_refine_plus/plots", 
            round_idx
        )
        
        # Collect Metrics
        metrics_row = {"round": round_idx}
        for stage in ["amortized", "sequential", "refined"]:
            m = res["metrics"].get(stage)
            if m:
                metrics_row[f"bias_l2_{stage}"] = m["bias_l2"]
                metrics_row[f"hdi_length_{stage}"] = np.mean(m["hdi_length"])
                metrics_row[f"coverage_{stage}"] = np.mean(m["coverage"])
            
        aggregated_metrics.append(metrics_row)
        
    # Final Summary
    df = pd.DataFrame(aggregated_metrics)
    
    # Calculate Mean/Std
    summary_row_mean = df.mean(numeric_only=True).to_dict()
    summary_row_mean["round"] = "MEAN"
    summary_row_std = df.std(numeric_only=True).to_dict()
    summary_row_std["round"] = "STD"
    
    df_final = pd.concat([df, pd.DataFrame([summary_row_mean]), pd.DataFrame([summary_row_std])], ignore_index=True)
    
    csv_path = "results_refine_plus/tables/final_summary.csv"
    df_final.to_csv(csv_path, index=False)
    print(f"\nExperiment Complete. Summary saved to {csv_path}")
    print(df_final)

if __name__ == "__main__":
    main()
