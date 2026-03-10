"""
Main experiment script for the SIR benchmark.

This script:
- loads configuration from config.json (dataset size, models to run, etc.),
- generates a synthetic training dataset from the SIRTask simulator,
- trains each selected model (SMMD, MMD, DNNABC, NPE, BayesFlow),
- performs a simple refinement step via ABC-MCMC (for SMMD/MMD/BayesFlow),
- computes metrics and writes all results (tables + plots) into results/.
"""

import os
import time
from datetime import timedelta
import json

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

try:
    import bayesflow as bf

    BAYESFLOW_AVAILABLE = True
except ImportError:
    BAYESFLOW_AVAILABLE = False

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde
import keras

from data_generation import SIRTask, d, d_x
from models.smmd import SMMD_Model, sliced_mmd_loss, mixture_sliced_mmd_loss
from models.mmd import MMD_Model, mmd_loss
from models.dnnabc import DNNABC_Model, train_dnnabc, abc_rejection_sampling
from models.sbi_wrappers import run_sbi_model
from models.bayesflow_net import build_bayesflow_model
from utilities import compute_metrics, fit_kde_and_evaluate, refine_posterior

try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {
        "num_rounds": 1,
        "N": 1000000.0,
        "T_MAX": 160.0,
        "NUM_OBS": 10,
        "dataset_size": 12800,
        "val_size": 100,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "n_samples_posterior": 1000,
        "models_to_run": ["smmd", "mmd", "bayesflow", "dnnabc", "npe"],
        "models_config": {
            "smmd": {"epochs": 300, "summary_dim": 4},
            "mmd": {"epochs": 300, "summary_dim": 4},
            "bayesflow": {"epochs": 300, "summary_dim": 4},
            "dnnabc": {"epochs": 300, "n_pool": 34400},
            "npe": {"sbi_rounds": 10, "sims_per_round": 6000, "epochs": 200},
        },
        "smmd_mmd_config": {
            "M": 50,
            "L": 20,
            "bandwidth_n_samples": 4400,
        },
        "refinement_config": {
            "n_chains": 1000,
            "burn_in": 29,
            "n_samples_per_chain": 1,
        },
    }

DATASET_SIZE = CONFIG.get("dataset_size", 12800)
VAL_SIZE = CONFIG.get("val_size", 100)
BATCH_SIZE = CONFIG.get("batch_size", 128)
NUM_ROUNDS = CONFIG.get("num_rounds", 1)
MODELS_TO_RUN = CONFIG.get("models_to_run", ["smmd", "mmd", "bayesflow", "dnnabc", "npe"])
MODELS_CONFIG = CONFIG.get("models_config", {})

SMMD_MMD_CONFIG = CONFIG.get(
    "smmd_mmd_config", {"M": 50, "L": 20, "bandwidth_n_samples": 4400}
)
M = SMMD_MMD_CONFIG.get("M", 50)
L = SMMD_MMD_CONFIG.get("L", 20)
BANDWIDTH_N_SAMPLES = SMMD_MMD_CONFIG.get("bandwidth_n_samples", 4400)

REFINEMENT_CONFIG = CONFIG.get(
    "refinement_config",
    {"n_chains": 1000, "burn_in": 29, "n_samples_per_chain": 1},
)
REFINE_N_CHAINS = REFINEMENT_CONFIG.get("n_chains", 1000)
REFINE_BURN_IN = REFINEMENT_CONFIG.get("burn_in", 29)
REFINE_NS_PER_CHAIN = REFINEMENT_CONFIG.get("n_samples_per_chain", 1)

LEARNING_RATE = CONFIG.get("learning_rate", 1e-3)
N_SAMPLES_POSTERIOR = CONFIG.get("n_samples_posterior", 1000)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def get_scheduler(optimizer, epochs):
    """Return a cosine annealing learning rate scheduler used for SMMD/MMD."""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)


def train_smmd_mmd(model, train_loader, epochs, device, model_type="smmd", n_time_steps=50):
    """
    Train SMMD or MMD model for a given number of epochs.

    Args:
        model: SMMD_Model or MMD_Model instance.
        train_loader: DataLoader yielding (x, theta) batches.
        epochs: number of training epochs.
        device: torch.device on which to run training.
        model_type: "smmd" or "mmd" to pick the correct loss.
        n_time_steps: used only to scale the kernel bandwidth in the loss.

    Returns:
        loss_history: list of average loss values per epoch.
        training_time: wall-clock training time in seconds.
    """
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(optimizer, epochs)
    model.to(device)
    model.train()
    loss_history = []
    print(f"Starting training ({model_type}) for {epochs} epochs...")
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            theta_batch = batch[1].to(device)
            optimizer.zero_grad()
            with torch.enable_grad():
                z = torch.randn(x_batch.size(0), M, model.d, device=device)
                theta_fake = model(x_batch, z)
                if model_type == "smmd":
                    loss = sliced_mmd_loss(
                        theta_batch, theta_fake, num_slices=L, n_time_steps=n_time_steps
                    )
                elif model_type == "mmd":
                    loss = mmd_loss(theta_batch, theta_fake, n_time_steps=n_time_steps)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"[{model_type.upper()}] Epoch {epoch+1}/{epochs}, loss = {avg_loss:.4f}")
    training_time = time.time() - start_time
    print(f"Finished training ({model_type}) in {training_time:.2f} seconds")
    return loss_history, training_time


def train_bayesflow(train_loader, epochs, device, summary_dim=10):
    """
    Build and train a BayesFlow amortized posterior for the SIR task.

    The BayesFlow model uses Keras with a PyTorch backend. We wrap the training
    loop here so the rest of the script can treat it like a standard model.
    """
    if not BAYESFLOW_AVAILABLE:
        raise ImportError("BayesFlow is not available.")
    print(f"Starting training (bayesflow) for {epochs} epochs...")
    first_batch = next(iter(train_loader))
    first_x, first_theta = first_batch[0], first_batch[1]
    amortized_posterior = build_bayesflow_model(d, d_x, summary_dim=summary_dim)
    try:
        dummy_dict = {
            "inference_variables": first_theta.float(),
            "summary_variables": first_x.float(),
        }
        if hasattr(amortized_posterior, "adapter"):
            _ = amortized_posterior.adapter(dummy_dict)
        _ = amortized_posterior.log_prob(dummy_dict)
    except Exception:
        pass
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    amortized_posterior.compile(optimizer=optimizer)
    loss_history = []
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for batch in train_loader:
            x_batch = batch[0].float()
            theta_batch = batch[1].float()
            batch_dict = {
                "inference_variables": theta_batch,
                "summary_variables": x_batch,
            }
            metrics = amortized_posterior.train_step(batch_dict)
            loss_val = metrics.get("loss", list(metrics.values())[0])
            if hasattr(loss_val, "item"):
                loss_val = loss_val.item()
            else:
                loss_val = float(loss_val)
            epoch_loss += loss_val
            num_batches += 1
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"[BAYESFLOW] Epoch {epoch+1}/{epochs}, loss = {avg_loss:.4f}")
    training_time = time.time() - start_time
    print(f"Finished training (bayesflow) in {training_time:.2f} seconds")
    return amortized_posterior, loss_history, training_time

def plot_posterior(theta_samples, theta_true, output_dir, filename="posterior.png"):
    """
    Plot 1D marginal posteriors for each parameter against the ground truth.
    """
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(theta_samples, torch.Tensor):
        theta_samples = theta_samples.detach().cpu().numpy()
    theta_true = np.asarray(theta_true, dtype=np.float32)
    num_params = theta_samples.shape[1]
    fig, axes = plt.subplots(1, num_params, figsize=(4 * num_params, 4))
    if num_params == 1:
        axes = [axes]
    for i in range(num_params):
        ax = axes[i]
        sns.kdeplot(theta_samples[:, i], ax=ax, fill=True)
        ax.axvline(theta_true[i], color="red", linestyle="--")
        ax.set_title(f"Param {i+1}")
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()


def plot_combined_posteriors(all_model_samples, theta_true, output_dir, filename="combined_posteriors.png"):
    """
    Plots posteriors from multiple models on one figure.
    all_model_samples: dict {model_name: samples}
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_model_samples:
        return

    # Style settings
    sns.set_theme(style="whitegrid", font_scale=1.5)  # Switch to whitegrid for cleaner look
    
    # Dimensions to plot
    dims_to_plot = [0, 1]
    param_names = [r"$\beta$", r"$\gamma$"]
    
    # X limits (optional, can be adjusted based on data range)
    # Beta ~ LogNormal, Gamma ~ LogNormal. True values are often small positive.
    # We can let it auto-scale or set reasonable priors.
    # For SIR, usually beta in [0, 2], gamma in [0, 1] roughly.
    x_limits = [
        [0.0, 1.5],   # beta
        [0.0, 1.0],   # gamma
    ]
    
    # Colors
    colors = {
        "SMMD": "#DB1218",       # Red
        "MMD": "#1083CA",        # Blue
        "BAYESFLOW": "#49C926",  # Green
        "DNNABC": "#8315dd",     # Orange/Purple
        "W2ABC": "#0f12ae",      # Purple/Orange
        "NPE": "#f19327",        # Use W2 color for NPE
        "TRUE": "#FA0101",       # Red for True Value line
    }
    
    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(14, 6)) # Adjusted width
    plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    
    if cols == 1: axes = [axes]
    
    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24
    
    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]
        
        # Plot models
        for model_name, samples in all_model_samples.items():
            if samples is None: continue
            
            label_name = model_name.upper()
            
            # Determine color
            c = "#333333"
            if "SMMD" in label_name: c = colors["SMMD"]
            elif "MMD" in label_name: c = colors["MMD"]
            elif "BAYESFLOW" in label_name: c = colors["BAYESFLOW"]
            elif "DNN" in label_name: c = colors["DNNABC"]
            elif "W2" in label_name: c = colors["W2ABC"]
            elif "NPE" in label_name or "SNPE" in label_name: c = colors["NPE"]
            
            # Plot
            sns.kdeplot(x=samples[:, dim_idx], label=label_name, ax=ax, color=c, fill=True, alpha=0.15, linewidth=4.5)
            
        # True Value line
        if dim_idx < len(theta_true):
            ax.axvline(x=theta_true[dim_idx], color=colors["TRUE"], linestyle='-', linewidth=3.5, label='True Value', alpha=0.9)
            
        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight='bold', pad=15)
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize, width=2, length=6)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight='bold')
        else:
            ax.set_ylabel("")
            
        # Optional x-limits
        # ax.set_xlim(x_limits[i])
        
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.0)
        
    # Legend handling
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Define desired order
    order = ["SMMD", "MMD", "BAYESFLOW", "DNNABC", "NPE", "W2ABC", "True Value"]
    
    # Filter and sort
    handles_sorted = []
    labels_sorted = []
    for l in order:
        for k in by_label.keys():
            if l == "True Value" and k == "True Value":
                handles_sorted.append(by_label[k])
                labels_sorted.append(k)
            elif l in k and "True Value" not in k: # Avoid double matching
                if k not in labels_sorted:
                    handles_sorted.append(by_label[k])
                    labels_sorted.append(k)
                
    # If the specific order logic misses something, just use what's available
    if not handles_sorted:
        handles_sorted = handles
        labels_sorted = labels
        
    fig.legend(
        handles_sorted,
        labels_sorted,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.08), # Adjusted for bottom margin
        ncol=min(len(handles_sorted), 4), # Max 4 columns
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5
    )
        
    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches='tight')
    plt.close()

def plot_all_refinement_comparisons(all_results_dict, theta_true, output_dir):
    """
    Plots Amortized vs Refined for ALL models (SMMD, MMD) on one figure.
    all_results_dict: { model_name: {'Amortized': samples, 'Refined': samples} }
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = "all_models_refinement_comparison.png"
    
    # Style settings
    sns.set_theme(style="whitegrid", font_scale=1.5)
    
    # Dimensions to plot
    dims_to_plot = [0, 1]
    param_names = [r"$\beta$", r"$\gamma$"]
    
    # Colors
    colors = {
        "SMMD (Refined)": "#FF3A20",   # Bright Red
        "SMMD (Baseline)": "#DB1218",  # Dark Red
        "MMD (Refined)": "#3FA7D6",    # Bright Blue
        "MMD (Baseline)": "#1083CA",   # Dark Blue
        "BAYESFLOW (Refined)": "#49C926", # Bright Green
        "BAYESFLOW (Baseline)": "#2E7D32", # Dark Green
        "True Value": "#FA0101",
    }
    
    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(14, 6))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    
    if cols == 1: axes = [axes]
    
    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24
    
    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]
        
        # 1. Plot each model's Amortized and Refined
        for model_name, result_dict in all_results_dict.items():
            if result_dict is None: continue
            
            # Check for Amortized
            amortized = result_dict.get('Amortized')
            refined = result_dict.get('Refined')
            
            # Key lookup
            if model_name.lower() == "smmd":
                 c_key_base = f"SMMD (Baseline)"
                 c_key_ref = f"SMMD (Refined)"
                 m_label = "SMMD"
            elif model_name.lower() == "mmd":
                 c_key_base = f"MMD (Baseline)"
                 c_key_ref = f"MMD (Refined)"
                 m_label = "MMD"
            elif model_name.lower() == "bayesflow":
                 c_key_base = f"BAYESFLOW (Baseline)"
                 c_key_ref = f"BAYESFLOW (Refined)"
                 m_label = "BAYESFLOW"
            else:
                 continue 
            
            # Plot Amortized
            if amortized is not None:
                c = colors.get(c_key_base, "#333333")
                sns.kdeplot(x=amortized[:, dim_idx], label=f"{m_label} (Amortized)", ax=ax, color=c, fill=False, linestyle='--', linewidth=3.5)
                
            # Plot Refined
            if refined is not None:
                c = colors.get(c_key_ref, "#333333")
                sns.kdeplot(x=refined[:, dim_idx], label=f"{m_label} (Refined)", ax=ax, color=c, fill=True, alpha=0.15, linewidth=4.5)
        
        # True Value line
        if dim_idx < len(theta_true):
            ax.axvline(x=theta_true[dim_idx], color=colors["True Value"], linestyle='-', linewidth=3.5, label='True Value', alpha=0.9)
            
        ax.set_title(param_names[i], fontsize=title_fontsize, fontweight='bold', pad=15)
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize, width=2, length=6)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

        if i == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, fontweight='bold')
        else:
            ax.set_ylabel("")
            
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.0)
        
    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Desired order
    order = [
        "SMMD (Refined)", "SMMD (Amortized)",
        "MMD (Refined)", "MMD (Amortized)",
        "BAYESFLOW (Refined)", "BAYESFLOW (Amortized)",
        "True Value"
    ]
    
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
        ncol=min(len(sorted_handles), 3),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5
    )
    
    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches='tight')
    plt.close()


def format_time(seconds):
    """Format a duration in seconds as H:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def run_single_experiment(
    model_type,
    task,
    train_loader,
    theta_true,
    x_obs,
    round_id,
    model_config=None,
    initial_training_data=None,
):
    """
    Run the full pipeline for a single model in a single round.

    Steps:
    1. Train the amortized model (if applicable).
    2. Draw an initial amortized posterior given x_obs.
    3. Optionally run ABC-MCMC refinement (SMMD/MMD/BayesFlow).
    4. Save samples, plots and metrics for this round.
    """
    model_config = model_config or {}
    epochs = model_config.get("epochs", 50)
    summary_dim = model_config.get("summary_dim", 8)
    loss_history = []
    training_time = 0.0
    timings = {"initial": 0.0, "refine": 0.0, "mcmc": 0.0}
    samples_dict = {}
    n_obs_curr = x_obs.shape[0]
    print(f"\n=== Model: {model_type} | Round: {round_id} | Epochs: {epochs} ===")
    if model_type == "smmd":
        model = SMMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs_curr)
        loss_history, training_time = train_smmd_mmd(
            model, train_loader, epochs, DEVICE, "smmd", n_time_steps=n_obs_curr
        )
    elif model_type == "mmd":
        model = MMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs_curr)
        loss_history, training_time = train_smmd_mmd(
            model, train_loader, epochs, DEVICE, "mmd", n_time_steps=n_obs_curr
        )
    elif model_type == "bayesflow":
        if BAYESFLOW_AVAILABLE:
            bf_device = torch.device("cpu")
            model, loss_history, training_time = train_bayesflow(
                train_loader, epochs, bf_device, summary_dim=summary_dim
            )
        else:
            return None
    elif model_type == "dnnabc":
        model = DNNABC_Model(d=d, d_x=d_x, n_points=n_obs_curr)
        loss_history = train_dnnabc(model, train_loader, epochs, DEVICE)
        training_time = 0.0
    elif model_type in ["snpe_a", "npe", "snpe", "snpe_c"]:
        model = None
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    if loss_history:
        plot_posterior(
            np.array(loss_history).reshape(-1, 1),
            np.array([0.0]),
            f"results/models/{model_type}/round_{round_id}",
            "loss_history.png",
        )
    print(f"--- Amortized Inference ({model_type}) ---")
    n_samples = N_SAMPLES_POSTERIOR
    t_start_sample = time.time()
    if model_type in ["smmd", "mmd"]:
        if not isinstance(x_obs, torch.Tensor):
            x_obs_tensor = torch.from_numpy(x_obs).float().to(DEVICE)
        else:
            x_obs_tensor = x_obs.to(DEVICE)
        if x_obs_tensor.ndim == 2:
            x_obs_tensor = x_obs_tensor.unsqueeze(0)
        with torch.no_grad():
            stats = model.T(x_obs_tensor)
            z = torch.randn(1, n_samples, d, device=DEVICE)
            theta_samples = model.G(z, stats).squeeze(0).cpu().numpy()
        posterior_samples = theta_samples
    elif model_type == "dnnabc":
        posterior_samples = abc_rejection_sampling(
            model,
            x_obs,
            task,
            n_samples=n_samples,
            n_pool=model_config.get("n_pool", 6000),
            device=DEVICE,
        )
    elif model_type in ["snpe_a", "npe", "snpe", "snpe_c"]:
        sbi_rounds = model_config.get("sbi_rounds", 2)
        sims_per_round = model_config.get("sims_per_round", 1000)
        posterior_samples = run_sbi_model(
            model_type=model_type,
            train_loader=None,
            x_obs=x_obs,
            theta_true=theta_true,
            task=task,
            device=str(DEVICE),
            num_rounds=sbi_rounds,
            sims_per_round=sims_per_round,
            max_epochs=epochs,
            initial_training_data=initial_training_data,
        )
    elif model_type == "bayesflow":
        x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
        if x_obs_cpu.ndim == 2:
            x_obs_cpu = x_obs_cpu[np.newaxis, ...]
        post = model.sample(conditions={"summary_variables": x_obs_cpu}, num_samples=n_samples)
        if isinstance(post, dict):
            post = post["inference_variables"]
        if hasattr(post, "numpy"):
            post = post.numpy()
        posterior_samples = np.asarray(post).reshape(-1, d)
        std_per_dim = np.std(posterior_samples, axis=0)
        max_abs = np.max(np.abs(posterior_samples), axis=0)
        has_nan = not np.isfinite(posterior_samples).all()
        zero_var = np.any(std_per_dim < 1e-6)
        extreme = np.any(max_abs > 1e3)
        if has_nan or zero_var or extreme:
            metrics = {
                "stage": "initial",
                "bias_l2": np.nan,
                "hdi_length": np.full(d, np.nan),
                "coverage": np.full(d, np.nan),
                "training_time": training_time,
            }
            timings["initial"] = training_time
            return {
                "metrics": metrics,
                "metrics_initial": metrics,
                "timings": timings,
                "samples": {"initial": posterior_samples},
            }
    else:
        return None
    sampling_time = time.time() - t_start_sample
    metrics = compute_metrics(posterior_samples, theta_true)
    metrics["training_time"] = training_time
    metrics["sampling_time"] = sampling_time
    metrics["stage"] = "initial"
    metrics_initial = metrics.copy()
    timings["initial"] = training_time + sampling_time
    samples_dict["initial"] = posterior_samples
    base_dir = f"results/models/{model_type}/round_{round_id}/initial"
    os.makedirs(base_dir, exist_ok=True)
    np.save(f"{base_dir}/posterior_samples.npy", posterior_samples)
    plot_posterior(posterior_samples, theta_true, base_dir, "posterior_plot.png")
    save_model_dir = f"results/models/{model_type}/round_{round_id}"
    os.makedirs(save_model_dir, exist_ok=True)
    if model_type == "bayesflow":
        try:
            model.save_weights(f"{save_model_dir}/initial_weights.weights.h5")
        except Exception:
            pass
    elif model_type in ["smmd", "mmd"]:
        torch.save(model.state_dict(), f"{save_model_dir}/initial_state_dict.pt")
    
    # Refinement step (ABC-MCMC) only for SMMD and MMD and BayesFlow
    run_mcmc = model_type in ["smmd", "mmd", "bayesflow"]
    
    if run_mcmc:
        t_mcmc_start = time.time()
        if hasattr(model, "eval"):
            model.eval()
        theta_mcmc = refine_posterior(
            model,
            x_obs,
            task=task,
            n_chains=REFINE_N_CHAINS,
            n_samples=REFINE_NS_PER_CHAIN,
            burn_in=REFINE_BURN_IN,
            device=DEVICE,
            bandwidth_n_samples=BANDWIDTH_N_SAMPLES,
        )
        time_mcmc = time.time() - t_mcmc_start
        timings["mcmc"] = time_mcmc
        timings["refine"] = timings["refine"] + timings["mcmc"]
        samples_dict["mcmc"] = theta_mcmc
        samples_dict["refine"] = theta_mcmc
        metrics_mcmc = compute_metrics(theta_mcmc, theta_true)
        total_train_time = training_time
        metrics_mcmc["training_time"] = total_train_time + time_mcmc
        metrics_mcmc["stage"] = "refined_mcmc"
        refine_dir = f"results/models/{model_type}/round_{round_id}/refine"
        os.makedirs(refine_dir, exist_ok=True)
        np.save(f"{refine_dir}/posterior_samples.npy", theta_mcmc)
        plot_posterior(theta_mcmc, theta_true, refine_dir, "posterior_plot.png")
        return {
            "metrics": metrics_mcmc,
            "timings": timings,
            "samples": samples_dict,
            "metrics_initial": metrics_initial,
        }
    return {
        "metrics": metrics,
        "timings": timings,
        "samples": samples_dict,
        "metrics_initial": metrics_initial,
    }


def aggregate_and_save_results(model_results_table):
    """
    Aggregate per-round metrics for each model and write summary CSV files.

    This function produces:
    - results/tables/<model>_results.csv with all raw round-wise metrics.
    - results/models/<model>/<model>_per_round_metrics.csv with a compact view.
    - results/final_summary.csv with averages across rounds.
    """
    os.makedirs("results/tables", exist_ok=True)
    summary_data = []
    for model_name, rows in model_results_table.items():
        if not rows:
            continue
        df_model = pd.DataFrame(rows)
        cols = [
            "round",
            "status",
            "stage",
            "bias_l2",
            "hdi_length",
            "coverage",
            "time_initial",
            "time_refine",
            "time_mcmc",
        ]
        other_cols = [
            c for c in df_model.columns if c not in cols and not c.startswith("seconds_")
        ]
        final_cols = cols + other_cols
        final_cols = [c for c in final_cols if c in df_model.columns]
        df_model = df_model[final_cols]
        df_model.to_csv(f"results/tables/{model_name}_results.csv", index=False)
        success_rows = [r for r in rows if r.get("status", 0) == 1]
        if not success_rows:
            continue
        example = None
        for r in success_rows:
            if "hdi_length_initial" in r or "hdi_length" in r:
                example = r
                break
        if example is None:
            continue
        if "hdi_length_initial" in example:
            example_hdi = np.asarray(example["hdi_length_initial"], dtype=float)
        else:
            example_hdi = np.asarray(example["hdi_length"], dtype=float)
        num_params = example_hdi.shape[0]
        per_round_rows = []
        refine_models = ["smmd", "mmd", "bayesflow"]
        for r in rows:
            round_id = r.get("round", np.nan)
            status = r.get("status", 0)
            if status != 1:
                row_metrics = {
                    "round": round_id,
                    "Bias": np.nan,
                    "Refine_Bias": np.nan,
                }
                for j in range(num_params):
                    row_metrics[f"HDI_Len_Param{j+1}"] = np.nan
                    row_metrics[f"Coverage_Param{j+1}"] = np.nan
                    row_metrics[f"Refine_HDI_Len_Param{j+1}"] = np.nan
                    row_metrics[f"Refine_Coverage_Param{j+1}"] = np.nan
                per_round_rows.append(row_metrics)
                continue
            if "bias_l2_initial" in r:
                bias_init = r["bias_l2_initial"]
            elif "bias_l2" in r:
                bias_init = r["bias_l2"]
            else:
                bias_init = np.nan
            if model_name in refine_models and "bias_l2_refine" in r:
                bias_ref = r["bias_l2_refine"]
            else:
                bias_ref = np.nan
            if "hdi_length_initial" in r:
                hdi_init = np.asarray(r["hdi_length_initial"], dtype=float)
            elif "hdi_length" in r:
                hdi_init = np.asarray(r["hdi_length"], dtype=float)
            else:
                hdi_init = np.full(num_params, np.nan)
            if model_name in refine_models and "hdi_length_refine" in r:
                hdi_ref = np.asarray(r["hdi_length_refine"], dtype=float)
            else:
                hdi_ref = np.full(num_params, np.nan)
            if "coverage_initial" in r:
                cov_init = np.asarray(r["coverage_initial"], dtype=float)
            elif "coverage" in r:
                cov_init = np.asarray(r["coverage"], dtype=float)
            else:
                cov_init = np.full(num_params, np.nan)
            if model_name in refine_models and "coverage_refine" in r:
                cov_ref = np.asarray(r["coverage_refine"], dtype=float)
            else:
                cov_ref = np.full(num_params, np.nan)
            row_metrics = {
                "round": round_id,
                "Bias": bias_init,
                "Refine_Bias": bias_ref,
            }
            for j in range(num_params):
                row_metrics[f"HDI_Len_Param{j+1}"] = hdi_init[j]
                row_metrics[f"Coverage_Param{j+1}"] = cov_init[j]
                row_metrics[f"Refine_HDI_Len_Param{j+1}"] = hdi_ref[j]
                row_metrics[f"Refine_Coverage_Param{j+1}"] = cov_ref[j]
            per_round_rows.append(row_metrics)
        per_round_df = pd.DataFrame(per_round_rows)
        model_dir = f"results/models/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        per_round_path = f"{model_dir}/{model_name}_per_round_metrics.csv"
        per_round_df.to_csv(per_round_path, index=False)
        bias_initial = []
        bias_refine = []
        hdi_initial_list = []
        hdi_refine_list = []
        coverage_initial_list = []
        coverage_refine_list = []
        for r in success_rows:
            if "bias_l2_initial" in r:
                bias_initial.append(r["bias_l2_initial"])
            elif "bias_l2" in r:
                bias_initial.append(r["bias_l2"])
            if model_name in refine_models and "bias_l2_refine" in r:
                bias_refine.append(r["bias_l2_refine"])
            if "hdi_length_initial" in r:
                hdi_initial_list.append(np.asarray(r["hdi_length_initial"], dtype=float))
            elif "hdi_length" in r:
                hdi_initial_list.append(np.asarray(r["hdi_length"], dtype=float))
            if model_name in refine_models and "hdi_length_refine" in r:
                hdi_refine_list.append(
                    np.asarray(r["hdi_length_refine"], dtype=float)
                )
            if "coverage_initial" in r:
                coverage_initial_list.append(
                    np.asarray(r["coverage_initial"], dtype=float)
                )
            elif "coverage" in r:
                coverage_initial_list.append(np.asarray(r["coverage"], dtype=float))
            if model_name in refine_models and "coverage_refine" in r:
                coverage_refine_list.append(
                    np.asarray(r["coverage_refine"], dtype=float)
                )
        hdi_initial_arr = np.vstack(hdi_initial_list)
        avg_hdi_len_initial = hdi_initial_arr.mean(axis=0)
        coverage_initial_arr = np.vstack(coverage_initial_list)
        avg_coverage_initial = coverage_initial_arr.mean(axis=0)
        if hdi_refine_list:
            hdi_refine_arr = np.vstack(hdi_refine_list)
            avg_hdi_len_refine = hdi_refine_arr.mean(axis=0)
        else:
            avg_hdi_len_refine = np.full_like(avg_hdi_len_initial, np.nan)
        if coverage_refine_list:
            coverage_refine_arr = np.vstack(coverage_refine_list)
            avg_coverage_refine = coverage_refine_arr.mean(axis=0)
        else:
            avg_coverage_refine = np.full_like(avg_coverage_initial, np.nan)
        avg_sec_initial = np.mean([r["seconds_initial"] for r in success_rows])
        avg_sec_refine = np.mean([r["seconds_refine"] for r in success_rows])
        avg_sec_mcmc = np.mean([r.get("seconds_mcmc", 0.0) for r in success_rows])
        mean_bias_initial = np.mean(bias_initial)
        if bias_refine:
            mean_bias_refine = np.mean(bias_refine)
        else:
            mean_bias_refine = np.nan
        summary_row = {
            "Model": model_name,
            "Bias_Mean": mean_bias_initial,
            "Refine_Bias_Mean": mean_bias_refine,
            "Avg_Time_Initial": format_time(avg_sec_initial),
            "Avg_Time_Refine": format_time(avg_sec_refine),
            "Avg_Time_MCMC": format_time(avg_sec_mcmc),
        }
        for j in range(num_params):
            summary_row[f"HDI_Len_Param{j+1}_Mean"] = avg_hdi_len_initial[j]
            summary_row[f"Coverage_Param{j+1}_Mean"] = avg_coverage_initial[j]
            summary_row[f"Refine_HDI_Len_Param{j+1}_Mean"] = avg_hdi_len_refine[j]
            summary_row[f"Refine_Coverage_Param{j+1}_Mean"] = avg_coverage_refine[j]
        summary_data.append(summary_row)
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv("results/final_summary.csv", index=False)

def main():
    """
    Entry point for the SIR benchmark experiment.

    It:
    - builds a global training dataset from the SIRTask prior and simulator,
    - prepares a shared DataLoader used by all amortized models,
    - runs multiple rounds of inference for each selected model,
    - performs refinement (ABC-MCMC) where applicable,
    - and calls aggregate_and_save_results to export all metrics.
    """
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/comparisons", exist_ok=True)
    model_results_table = {m: [] for m in MODELS_TO_RUN}
    
    # Initialize Task
    task = SIRTask()
    
    for round_idx in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*20} ROUND {round_idx}/{NUM_ROUNDS} {'='*20}")
        
        # 1. Generate Training Data (Fresh for each round)
        print("Generating new training dataset...")
        # SIRTask uses numpy random, so calling sample_prior produces new samples
        theta_all = task.sample_prior(DATASET_SIZE)
        x_all = task.simulator(theta_all)
        x_train = x_all[:DATASET_SIZE]
        theta_train = theta_all[:DATASET_SIZE]
        
        # Prepare DataLoader
        x_tensor = torch.from_numpy(x_train).float()
        theta_tensor = torch.from_numpy(theta_train).float()
        dataset = TensorDataset(x_tensor, theta_tensor)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 2. Get Ground Truth (Fresh for each round)
        theta_true, x_obs = task.get_ground_truth()
        
        round_initial_samples = {}
        round_refinement_data = {} # {model_name: {'Amortized': ..., 'Refined': ...}}

        for model_name in MODELS_TO_RUN:
            try:
                model_conf = MODELS_CONFIG.get(model_name, {})
                if isinstance(model_conf, int):
                    model_conf = {"epochs": model_conf}
                
                # Pass the fresh train_loader
                res = run_single_experiment(
                    model_name,
                    task,
                    train_loader,
                    theta_true,
                    x_obs,
                    round_idx,
                    model_config=model_conf,
                    initial_training_data=(x_train, theta_train),
                )
                
                if res:
                    metrics = res["metrics"]
                    timings = res["timings"]
                    samples = res["samples"]
                    
                    # Store initial samples for combined plot
                    if "initial" in samples and samples["initial"] is not None:
                        round_initial_samples[model_name] = samples["initial"]
                    
                    # Store refinement data if applicable
                    if model_name in ["smmd", "mmd", "bayesflow"]:
                        round_refinement_data[model_name] = {
                            "Amortized": samples.get("initial"),
                            "Refined": samples.get("refine")
                        }
                    
                    metrics_initial = res.get("metrics_initial")
                    row = metrics.copy()
                    row["round"] = round_idx
                    row["status"] = 1
                    
                    # Record Initial Metrics
                    if metrics_initial is not None:
                        if "bias_l2" in metrics_initial:
                            row["bias_l2_initial"] = metrics_initial["bias_l2"]
                        if "hdi_length" in metrics_initial:
                            row["hdi_length_initial"] = metrics_initial["hdi_length"]
                        if "coverage" in metrics_initial:
                            row["coverage_initial"] = metrics_initial["coverage"]
                    else:
                        # Fallback if no separate initial metrics
                        if "bias_l2" in metrics:
                            row["bias_l2_initial"] = metrics["bias_l2"]
                        if "hdi_length" in metrics:
                            row["hdi_length_initial"] = metrics["hdi_length"]
                        if "coverage" in metrics:
                            row["coverage_initial"] = metrics["coverage"]
                            
                    # Record Refined Metrics (only for SMMD/MMD/BayesFlow)
                    if model_name in ["smmd", "mmd", "bayesflow"]:
                        if "bias_l2" in metrics: # metrics contains refined metrics here
                            row["bias_l2_refine"] = metrics["bias_l2"]
                        if "hdi_length" in metrics:
                            row["hdi_length_refine"] = metrics["hdi_length"]
                        if "coverage" in metrics:
                            row["coverage_refine"] = metrics["coverage"]
                            
                    row["time_initial"] = format_time(timings.get("initial", 0))
                    row["time_refine"] = format_time(timings.get("refine", 0))
                    row["seconds_initial"] = timings.get("initial", 0)
                    row["seconds_refine"] = timings.get("refine", 0)
                    row["seconds_mcmc"] = timings.get("mcmc", 0)
                    
                    model_results_table[model_name].append(row)
            except Exception as e:
                print(f"Error running {model_name}: {e}")
                row = {"round": round_idx, "status": 0, "error_msg": str(e)}
                model_results_table[model_name].append(row)
        
        # Plot 1: All Amortized Posteriors
        if round_initial_samples:
            plot_combined_posteriors(
                round_initial_samples,
                theta_true,
                f"results/comparisons/round_{round_idx}",
                "all_methods_amortized_posterior.png",
            )
            
        # Plot 2: Refinement Comparison (Amortized vs Refined)
        if round_refinement_data:
            plot_all_refinement_comparisons(
                round_refinement_data,
                theta_true,
                f"results/comparisons/round_{round_idx}"
            )

        aggregate_and_save_results(model_results_table)
    aggregate_and_save_results(model_results_table)

if __name__ == "__main__":
    main()
