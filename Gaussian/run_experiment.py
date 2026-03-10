
"""
Run Experiment for Gaussian Task with SMMD/MMD/BayesFlow and Refinement.
"""

import os
import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# ensure the backend is set
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import keras


# Import from local modules
from data_generation import (
    GaussianTask,
    generate_dataset,
    TRUE_PARAMS,
    d, d_x, p
)
from models.smmd import SMMD_Model, sliced_mmd_loss
from models.mmd import MMD_Model, mmd_loss
from models.bayesflow_net import build_bayesflow_model
from models.dnnabc import DNNABC_Model, train_dnnabc, abc_rejection_sampling
from models.w2abc import run_w2abc
from models.sbi_wrappers import run_sbi_model
from utilities import refine_posterior, compute_bandwidth_torch, compute_mmd_metric, run_gaussian_posterior_mcmc
from pymc_sampler import run_pymc

# Try importing BayesFlow (optional)
try:
    import bayesflow as bf
    BAYESFLOW_AVAILABLE = True
except ImportError:
    BAYESFLOW_AVAILABLE = False
    print("BayesFlow not installed or import failed. BayesFlow model will be unavailable.")

# ============================================================================
# 1. Hyperparameters & Device
# ============================================================================
# Default values (will be overridden by config.json if available)
M = 50
L = 20
BATCH_SIZE = 256
DATASET_SIZE = 25600
LEARNING_RATE = 1e-3
L1_LAMBDA = 1e-4
NUM_ROUNDS = 5
n = 50
RESULTS_BASE = "results"
N_OBS_LIST = None
N_SAMPLES_POSTERIOR = 1000

# Load Configuration from JSON
CONFIG_PATH = "config.json"
try:
    print(f"Loading configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print(f"Warning: {CONFIG_PATH} not found. Using default parameters.")
    CONFIG = {}

# Generic experiment-level settings
n = CONFIG.get("n_observation", n)
N_OBS_LIST = CONFIG.get("n_observation_list", None)
NUM_ROUNDS = CONFIG.get("num_rounds", NUM_ROUNDS)
DATASET_SIZE = CONFIG.get("dataset_size", DATASET_SIZE)
BATCH_SIZE = CONFIG.get("batch_size", BATCH_SIZE)
LEARNING_RATE = CONFIG.get("learning_rate", LEARNING_RATE)
N_SAMPLES_POSTERIOR = CONFIG.get("n_samples_posterior", N_SAMPLES_POSTERIOR)

# True posterior sampling config
TRUE_POSTERIOR_CONFIG = CONFIG.get(
    "true_posterior_config",
    {
        "method": "mcmc",
        "n_draws": 2000,
        "n_tune_chain": 1000,
        "chains": 30,
        "proposal_scale": 0.06,
    },
)

# ABC-MCMC refinement config (shared by SMMD/MMD/BayesFlow)
REFINE_CONFIG = CONFIG.get(
    "refine_config",
    {
        "n_chains": 1000,
        "burn_in": 99,
        "thin": 1,
        "nsims": 50,
        "epsilon": None,
    },
)

RAW_MODELS_CONFIG = CONFIG.get(
    "models_config",
    {
        "smmd": 200,
        "mmd": 200,
        "bayesflow": 200,
        "dnnabc": 200,
        "npe": 1000,
        "w2abc": 0,
    },
)

MODELS_CONFIG = {}
for name, cfg in RAW_MODELS_CONFIG.items():
    if isinstance(cfg, dict):
        MODELS_CONFIG[name] = cfg.copy()
    else:
        MODELS_CONFIG[name] = {"epochs": cfg}

# Fill per-model defaults
for name in ["smmd", "mmd", "bayesflow"]:
    mc = MODELS_CONFIG.setdefault(name, {})
    mc.setdefault("epochs", 200)
    mc.setdefault("summary_dim", 10)
    mc.setdefault("learning_rate", LEARNING_RATE)

mc = MODELS_CONFIG.setdefault("dnnabc", {})
mc.setdefault("epochs", 200)

mc = MODELS_CONFIG.setdefault("npe", {})
mc.setdefault("epochs", 1000)
mc.setdefault("sbi_rounds", 2)
mc.setdefault("sims_per_round", 1000)

mc = MODELS_CONFIG.setdefault("w2abc", {})
mc.setdefault("max_populations", 2)

MODELS_TO_RUN = CONFIG.get("models_to_run", list(MODELS_CONFIG.keys()))
print(f"Configuration loaded: n={n}, rounds={NUM_ROUNDS}, models={MODELS_TO_RUN}")

if N_OBS_LIST is None:
    N_OBS_LIST = [n]

# Check for MPS (Apple Silicon) or CUDA
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"✅ Using MPS (Apple Silicon) acceleration. Device: {DEVICE}")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    # Set default tensor type to cuda if desired, but explicit placement is better
    print(f"✅ Using CUDA acceleration. Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
else:
    DEVICE = torch.device("cpu")
    print(f"ℹ️ Using CPU. Device: {DEVICE}")

def get_scheduler(optimizer, epochs):
    """Cosine Decay Scheduler"""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

def train_smmd_mmd(model, train_loader, epochs, device, model_type="smmd", lr=None):
    if lr is None:
        lr = LEARNING_RATE
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, epochs)
    
    model.to(device)
    model.train()
    
    # Verify device placement
    param_device = next(model.parameters()).device
    print(f"Model {model_type.upper()} initialized on: {param_device}")

    loss_history = []
    
    print(f"Starting training ({model_type})...")
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
                
                if model_type == "smmd":
                    loss = sliced_mmd_loss(theta_batch, theta_fake, num_slices=L, n_points=M)
                elif model_type == "mmd":
                    loss = mmd_loss(theta_batch, theta_fake, n_points=M)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
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
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f}s")
    return loss_history, training_time

def train_bayesflow(train_loader, epochs, device, summary_dim=10, learning_rate=None):
    """
    Train BayesFlow model using Keras training loop.
    """
    if not BAYESFLOW_AVAILABLE:
        raise ImportError("BayesFlow is not available.")
        
    print("Starting training (BayesFlow with Keras Loop)...")
    
    # Build model
    amortized_posterior = build_bayesflow_model(d, d_x, summary_dim=summary_dim)
    
    # Manual build (one forward pass) to initialize weights
    try:
        first_x, first_theta = next(iter(train_loader))
        # Keep on CPU for adapter
        dummy_dict = {
            "inference_variables": first_theta.float(),
            "summary_variables": first_x.float()
        }
        
        # Explicitly initialize adapter
        if hasattr(amortized_posterior, "adapter"):
             _ = amortized_posterior.adapter(dummy_dict)
        
        # Run one forward pass via log_prob to build networks
        _ = amortized_posterior.log_prob(dummy_dict)
        print("BayesFlow model built successfully.")
    except Exception as e:
        print(f"Manual build warning: {e}")
        
    # Compile model
    if learning_rate is None:
        learning_rate = LEARNING_RATE
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    amortized_posterior.compile(optimizer=optimizer)
    
    loss_history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        for x_batch, theta_batch in train_loader:
            # Keep data on CPU for BayesFlow adapter
            x_batch_cpu = x_batch.float()
            theta_batch_cpu = theta_batch.float()
            
            batch_dict = {
                "inference_variables": theta_batch_cpu,
                "summary_variables": x_batch_cpu
            }
            
            try:
                # Use Keras train_step
                metrics = amortized_posterior.train_step(batch_dict)
                
                # metrics is a dict, extract loss
                # Usually key is "loss"
                loss_val = metrics.get("loss", list(metrics.values())[0])
                
                # loss_val might be a tensor or scalar
                if hasattr(loss_val, "item"):
                    loss_val = loss_val.item()
                else:
                    loss_val = float(loss_val)
                    
                epoch_loss += loss_val
                num_batches += 1
            except Exception as e:
                print(f"Error in train_step: {e}")
                break
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            
            # Progress reporting
            if (epoch + 1) % 1 == 0: 
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f}s")
    
    return amortized_posterior, loss_history, training_time

def plot_loss(loss_history, title="Training Loss", round_id=0):
    output_dir = f"{RESULTS_BASE}/plots/round_{round_id}/loss"
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title(f"{title} (Round {round_id})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{output_dir}/{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_posterior(theta_samples, theta_true, output_dir, filename="posterior.png"):
    """
    Plots the marginal posterior for each of the 5 dimensions in a single row.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_params = theta_samples.shape[1]
    # Ensure we have at least 5 dimensions or plot all available
    cols = num_params
    
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]
    
    # Create DataFrame for easier plotting with seaborn
    df = pd.DataFrame(theta_samples, columns=[f'theta{i+1}' for i in range(num_params)])
    
    for i in range(cols):
        ax = axes[i]
        param_name = f'theta{i+1}'
        
        # Plot KDE
        sns.kdeplot(data=df, x=param_name, fill=True, alpha=0.5, ax=ax, label='Posterior')
        
        # Plot True Parameter
        if i < len(theta_true):
            ax.axvline(x=theta_true[i], color='red', linestyle='--', linewidth=2, label='True Parameter')
            
        ax.set_title(f'Marginal {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
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
    
    # Dimensions to plot: theta_1, theta_2, theta_3, theta_5 (Indices: 0, 1, 2, 4)
    dims_to_plot = [0, 1, 2, 4]
    param_names = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_5$"]
    
    # X limits
    x_limits = [
        [0.5, 1.5],   # theta_1
        [0.5, 1.5],   # theta_2
        [-1.7, 1.7],  # theta_3
        [0.0, 1.2],  # theta_5
    ]
    
    # Colors
    colors = {
        "SMMD": "#DB1218",       # Red
        "MMD": "#1083CA",        # Blue
        "BAYESFLOW": "#49C926",  # Green
        "DNNABC": "#8315dd",     # Orange/Purple
        "W2ABC": "#0f12ae",      # Purple/Orange
        "NPE": "#f19327",        # Use W2 color for NPE
        "PYMC": "#363336",       # Dark Grey for True Posterior
        "TRUE": "#FA0101",       # Red for True Value line
    }
    
    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6)) # Slightly reduced height
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    
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
            elif "PYMC" in label_name: c = colors["PYMC"]
            
            # Plot
            if "PYMC" in label_name:
                sns.kdeplot(x=samples[:, dim_idx], label="True Posterior", ax=ax, color=c, fill=False, linestyle='-.', linewidth=4.0)
            else:
                sns.kdeplot(x=samples[:, dim_idx], label=label_name, ax=ax, color=c, fill=True, alpha=0.15, linewidth=4.5)
            
        # True Value line removed as requested
        # if dim_idx < len(theta_true):
        #     ax.axvline(x=theta_true[dim_idx], color=colors["TRUE"], linestyle='-', linewidth=3.5, label='True Value', alpha=0.9)
            
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
            # ax.tick_params(axis='y', labelleft=False) # Keep y-ticks visible for better readability
            
        ax.set_xlim(x_limits[i])
        
        # sns.despine(ax=ax, left=True if i > 0 else False) # Remove despine for cleaner box look
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.0)
        
    # Legend handling
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Define desired order
    order = ["SMMD", "MMD", "BAYESFLOW", "DNNABC", "NPE", "W2ABC", "True Posterior"]
    
    # Filter and sort
    handles_sorted = []
    labels_sorted = []
    for l in order:
        # Check if any label contains this key (exact match or substring)
        # Actually existing labels are likely "SMMD", "MMD", "BAYESFLOW", "DNNABC", "NPE", "PYMC" (mapped to True Posterior)
        # Let's try to match existing keys
        for k in by_label.keys():
            if l == "True Posterior" and k == "True Posterior":
                handles_sorted.append(by_label[k])
                labels_sorted.append(k)
            elif l == "True Value" and k == "True Value":
                handles_sorted.append(by_label[k])
                labels_sorted.append(k)
            elif l in k and "True Posterior" not in k and "True Value" not in k: # Avoid double matching
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

def plot_refinement_comparison(samples_dict, theta_true, output_dir, model_name, true_posterior_samples=None):
    """
    Plots Amortized vs Refined for a single model.
    samples_dict: { 'Amortized': samples, 'Refined': samples }
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_refinement_comparison.png"
    
    # Check if we have samples
    valid_samples = [s for s in samples_dict.values() if s is not None]
    if not valid_samples:
        return

    # Style settings
    sns.set_theme(style="whitegrid", font_scale=1.5)

    # Dimensions to plot: theta_1, theta_2, theta_3, theta_5 (Indices: 0, 1, 2, 4)
    dims_to_plot = [0, 1, 2, 4]
    param_names = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_5$"]
    
    # X limits
    x_limits = [
        [0.5, 1.5],   # theta_1
        [0.5, 1.5],   # theta_2
        [-1.7, 1.7],  # theta_3
        [0.0, 1.2],  # theta_5
    ]
    
    colors = {
        "SMMD (Refined)": "#FF3A20",   # Bright Red
        "SMMD (Baseline)": "#DB1218",  # Dark Red
        "MMD (Refined)": "#3FA7D6",    # Bright Blue
        "MMD (Baseline)": "#1083CA",   # Dark Blue
        "BayesFlow (Refined)": "#59CD90", # Bright Green
        "BayesFlow (Baseline)": "#49C926", # Dark Green
        "True Posterior": "#363336",
        "True Value": "#FA0101"
    }
    
    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    
    if cols == 1: axes = [axes]
    
    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24
    
    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]
        
        # 1. Plot True Posterior first if available
        if true_posterior_samples is not None:
             sns.kdeplot(
                x=true_posterior_samples[:, dim_idx], 
                ax=ax, 
                color=colors["True Posterior"], 
                fill=False, 
                linestyle='-.', 
                linewidth=4.0, 
                label="True Posterior"
            )
        
        # 2. Plot Amortized/Refined
        for label, samples in samples_dict.items():
            if samples is None: continue
            
            # Determine color
            # label is usually "Amortized" or "Refined"
            # model_name is "smmd", "mmd", "bayesflow"
            
            full_label = f"{model_name} ({'Baseline' if label == 'Amortized' else label})"
            # Fix case sensitivity
            if model_name.lower() == "smmd":
                 c_key = f"SMMD ({'Baseline' if label == 'Amortized' else 'Refined'})"
            elif model_name.lower() == "mmd":
                 c_key = f"MMD ({'Baseline' if label == 'Amortized' else 'Refined'})"
            elif model_name.lower() == "bayesflow":
                 c_key = f"BayesFlow ({'Baseline' if label == 'Amortized' else 'Refined'})"
            else:
                 c_key = None
            
            c = colors.get(c_key, "#333333")
            
            # Style
            if label == "Amortized":
                sns.kdeplot(x=samples[:, dim_idx], label=f"{model_name.upper()} (Amortized)", ax=ax, color=c, fill=False, linestyle='--', linewidth=3.5)
            else:
                sns.kdeplot(x=samples[:, dim_idx], label=f"{model_name.upper()} (Refined)", ax=ax, color=c, fill=True, alpha=0.15, linewidth=4.5)
            
        # True Value line removed as requested
        # if dim_idx < len(theta_true):
        #     ax.axvline(x=theta_true[dim_idx], color=colors["True Value"], linestyle='-', linewidth=3.5, label='True Value', alpha=0.9)
            
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
            # ax.tick_params(axis='y', labelleft=False)
            
        ax.set_xlim(x_limits[i])
        
        # sns.despine(ax=ax, left=True if i > 0 else False)
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.0)
        
    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    by_label = dict(zip(labels, handles))
    
    # Sort order
    # Refined, Amortized, True Posterior, True Value
    sorted_handles = []
    sorted_labels = []
    
    possible_keys = [
        f"{model_name.upper()} (Refined)",
        f"{model_name.upper()} (Amortized)",
        "True Posterior"
    ]
    
    for k in possible_keys:
        if k in by_label:
            sorted_handles.append(by_label[k])
            sorted_labels.append(k)
            
    fig.legend(
        sorted_handles,
        sorted_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.08),
        ncol=min(len(sorted_handles), 4),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5
    )
        
    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches='tight')
    plt.close()

def plot_all_refinement_comparisons(all_results_dict, theta_true, output_dir, true_posterior_samples=None):
    """
    Plots Amortized vs Refined for ALL models (SMMD, MMD, BayesFlow) on one figure.
    all_results_dict: { model_name: {'Amortized': samples, 'Refined': samples} }
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = "all_models_refinement_comparison.png"
    
    # Style settings
    sns.set_theme(style="whitegrid", font_scale=1.5)
    
    # Dimensions to plot: theta_1, theta_2, theta_3, theta_5 (Indices: 0, 1, 2, 4)
    dims_to_plot = [0, 1, 2, 4]
    param_names = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_5$"]
    
    # X limits
    x_limits = [
        [0.5, 1.5],   # theta_1
        [0.5, 1.5],   # theta_2
        [-1.7, 1.7],  # theta_3
        [0.0, 1.2],  # theta_5
    ]
    
    colors = {
        "SMMD (Refined)": "#FF3A20",   # Bright Red
        "SMMD (Baseline)": "#DB1218",  # Dark Red
        "MMD (Refined)": "#3FA7D6",    # Bright Blue
        "MMD (Baseline)": "#1083CA",   # Dark Blue
        "BayesFlow (Refined)": "#59CD90", # Bright Green
        "BayesFlow (Baseline)": "#49C926", # Dark Green
        "True Posterior": "#363336",
    }
    
    cols = len(dims_to_plot)
    fig, axes = plt.subplots(1, cols, figsize=(24, 6))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.20, wspace=0.25)
    
    if cols == 1: axes = [axes]
    
    tick_labelsize = 22
    title_fontsize = 28
    legend_fontsize = 22
    label_fontsize = 24
    
    for i, dim_idx in enumerate(dims_to_plot):
        ax = axes[i]
        
        # 1. Plot True Posterior first
        if true_posterior_samples is not None:
             sns.kdeplot(
                x=true_posterior_samples[:, dim_idx], 
                ax=ax, 
                color=colors["True Posterior"], 
                fill=False, 
                linestyle='-.', 
                linewidth=4.0, 
                label="True Posterior"
            )
            
        # 2. Plot each model's Amortized and Refined
        # We want to plot all models that have refinement info
        for model_name, result_dict in all_results_dict.items():
            if result_dict is None: continue
            
            # Check for Amortized
            amortized = result_dict.get('Amortized')
            refined = result_dict.get('Refined')
            
            # Key lookup
            # Fix case sensitivity
            if model_name.lower() == "smmd":
                 c_key_base = f"SMMD (Baseline)"
                 c_key_ref = f"SMMD (Refined)"
                 m_label = "SMMD"
            elif model_name.lower() == "mmd":
                 c_key_base = f"MMD (Baseline)"
                 c_key_ref = f"MMD (Refined)"
                 m_label = "MMD"
            elif model_name.lower() == "bayesflow":
                 c_key_base = f"BayesFlow (Baseline)"
                 c_key_ref = f"BayesFlow (Refined)"
                 m_label = "BayesFlow"
            else:
                 continue # Skip non-refinable models like NPE/DNNABC for this specific plot if not in color dict
            
            # Plot Amortized
            if amortized is not None:
                c = colors.get(c_key_base, "#333333")
                sns.kdeplot(x=amortized[:, dim_idx], label=f"{m_label} (Amortized)", ax=ax, color=c, fill=False, linestyle='--', linewidth=3.5)
                
            # Plot Refined
            if refined is not None:
                c = colors.get(c_key_ref, "#333333")
                sns.kdeplot(x=refined[:, dim_idx], label=f"{m_label} (Refined)", ax=ax, color=c, fill=True, alpha=0.15, linewidth=4.5)
        
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
            # ax.tick_params(axis='y', labelleft=False)
            
        ax.set_xlim(x_limits[i])
        
        # sns.despine(ax=ax, left=True if i > 0 else False)
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.0)
        
    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Desired order: 
    # SMMD(Ref), SMMD(Base), MMD(Ref), MMD(Base), BF(Ref), BF(Base), TruePost
    order = [
        "SMMD (Refined)", "SMMD (Amortized)",
        "MMD (Refined)", "MMD (Amortized)",
        "BayesFlow (Refined)", "BayesFlow (Amortized)",
        "True Posterior"
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
        ncol=min(len(sorted_handles), 4),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1.5
    )
    
    plt.savefig(os.path.join(output_dir, filename), dpi=350, bbox_inches='tight')
    plt.close()

def format_time(seconds):
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def run_single_experiment(
    model_type,
    task,
    train_loader,
    theta_true,
    x_obs,
    round_id,
    epochs=None,
    device=DEVICE,
    n_obs=n,
    initial_training_data=None,
    true_posterior_samples=None,
):
    """
    Runs a single experiment for a specific model type.
    """
    model_cfg = MODELS_CONFIG.get(model_type, {})
    if epochs is None:
        epochs = model_cfg.get("epochs", 30)
    print(f"\n=== Running Experiment for {model_type.upper()} (Round {round_id}, Epochs={epochs}, n={n_obs}) ===")
    
    # 1. Model Initialization & Training
    model = None
    loss_history = []
    training_time = 0.0
    refinement_time = 0.0
    total_time = 0.0 # For DNNABC/W2ABC
    
    t_start_total = time.time()
    
    if model_type == "smmd":
        summary_dim = model_cfg.get("summary_dim", 10)
        lr = model_cfg.get("learning_rate", LEARNING_RATE)
        model = SMMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs)
        loss_history, training_time = train_smmd_mmd(
            model, train_loader, epochs, device, model_type="smmd", lr=lr
        )
    elif model_type == "mmd":
        summary_dim = model_cfg.get("summary_dim", 10)
        lr = model_cfg.get("learning_rate", LEARNING_RATE)
        model = MMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs)
        loss_history, training_time = train_smmd_mmd(
            model, train_loader, epochs, device, model_type="mmd", lr=lr
        )
    elif model_type == "bayesflow":
        if BAYESFLOW_AVAILABLE:
            # Force CPU for BayesFlow to avoid MPS/Standardization issues
            bf_device = torch.device("cpu")
            print("Forcing BayesFlow to use CPU to avoid MPS/Keras compatibility issues.")
            summary_dim = model_cfg.get("summary_dim", 10)
            lr = model_cfg.get("learning_rate", LEARNING_RATE)
            model, loss_history, training_time = train_bayesflow(
                train_loader, epochs, bf_device, summary_dim=summary_dim, learning_rate=lr
            )
        else:
            print("BayesFlow unavailable, skipping.")
            return None
    elif model_type == "dnnabc":
        model = DNNABC_Model(d=d, d_x=d_x, n_points=n_obs)
        loss_history = train_dnnabc(model, train_loader, epochs, device)
    elif model_type == "w2abc":
        print("W2-ABC does not require neural network training. Proceeding to inference...")
        loss_history = []
    elif model_type in ["snpe_a", "npe", "snpe", "snpe_c"]:
        sbi_rounds = model_cfg.get("sbi_rounds", 2)
        sims_per_round = model_cfg.get("sims_per_round", 1000)
        posterior_samples = run_sbi_model(
            model_type=model_type,
            train_loader=None,
            x_obs=x_obs,
            theta_true=theta_true,
            task=task,
            device=str(device),
            num_rounds=sbi_rounds,
            sims_per_round=sims_per_round,
            max_epochs=epochs,
            n_samples=N_SAMPLES_POSTERIOR,
            initial_training_data=initial_training_data,
        )
        loss_history = []
            
    if loss_history:
        plot_loss(loss_history, title=f"{model_type.upper()} Loss", round_id=round_id)
    
    # 2. Model Inference (Amortized)
    print(f"--- Amortized Inference ({model_type}) ---")
    
    # Sample from model
    n_samples = N_SAMPLES_POSTERIOR
    
    if model_type == "w2abc":
        # W2ABC does inference via SMC, which takes time
        max_populations = model_cfg.get("max_populations", 2)
        posterior_samples = run_w2abc(
            task, x_obs, n_samples=n_samples, max_populations=max_populations
        )
        model = "W2ABC_SMC_Sampler" # Placeholder
    elif model_type in ["snpe_a", "npe", "snpe", "snpe_c"]:
        # Already sampled in training step
        pass
    elif model_type == "dnnabc":
        dnn_cfg = MODELS_CONFIG.get("dnnabc", {})
        n_pool = dnn_cfg.get("n_pool", 100000)
        abc_batch_size = dnn_cfg.get("abc_batch_size", 10000)
        posterior_samples = abc_rejection_sampling(
            model,
            x_obs,
            task,
            n_samples=n_samples,
            n_pool=n_pool,
            batch_size=abc_batch_size,
            device=device,
        )
    elif hasattr(model, "sample_posterior"):
        posterior_samples = model.sample_posterior(x_obs, n_samples)
    elif model_type == "bayesflow":
        # Prepare conditions for sampling
        # BayesFlow expects conditions as dict (via adapter)
        # IMPORTANT: Adapter requires CPU/Numpy data
        if isinstance(x_obs, torch.Tensor):
             x_obs_cpu = x_obs.detach().cpu().numpy()
        else:
             x_obs_cpu = np.asarray(x_obs)
             
        # Add batch dim if needed
        if x_obs_cpu.ndim == 2:
            x_obs_cpu = x_obs_cpu[np.newaxis, ...]
            
        conditions_dict = {"summary_variables": x_obs_cpu}
        
        # Sample
        posterior_samples = model.sample(conditions=conditions_dict, num_samples=n_samples)
        
        # Output might be dict
        if isinstance(posterior_samples, dict):
             posterior_samples = posterior_samples["inference_variables"]
             
        # Shape: (1, n_samples, d) -> (n_samples, d)
        posterior_samples = posterior_samples.reshape(-1, d)
    else:
        # Fallback (should typically not happen if wrappers are correct)
        print("Model does not support sample_posterior, skipping inference.")
        return None
        
    print(f"Posterior Mean: {np.mean(posterior_samples, axis=0)}")
    
    # Save Amortized Samples
    save_dir_samples = f"{RESULTS_BASE}/posterior_samples/{model_type}/round_{round_id}"
    os.makedirs(save_dir_samples, exist_ok=True)
    np.save(f"{save_dir_samples}/amortized.npy", posterior_samples)
    print(f"Saved amortized samples to {save_dir_samples}/amortized.npy")
    
    # Plot Amortized Posterior
    plot_dir = f"{RESULTS_BASE}/plots/round_{round_id}/{model_type}"
    plot_posterior(posterior_samples, theta_true, plot_dir, "amortized_posterior.png")
    
    # 3. Local Refinement
    refined_samples_flat = None
    refinement_time = 0.0
    
    if model_type in ["dnnabc", "w2abc", "snpe_a", "npe", "snpe", "snpe_c"]:
        print(f"--- Skipping Local Refinement for {model_type} (as requested) ---")
        refined_samples_flat = posterior_samples
        total_time = time.time() - t_start_total
    else:
        print(f"--- Local Refinement ({model_type}) ---")
        
        t_refine_start = time.time()
        
        # Refinement parameters
        n_chains = REFINE_CONFIG.get("n_chains", 1000)
        burn_in = REFINE_CONFIG.get("burn_in", 99)
        thin = REFINE_CONFIG.get("thin", 1)
        nsims = REFINE_CONFIG.get("nsims", 50)
        epsilon = REFINE_CONFIG.get("epsilon", None)
        
        refined_samples = refine_posterior(
            model,
            x_obs,
            task,
            n_chains=n_chains,
            n_samples=1,  # Take the last sample
            burn_in=burn_in,
            thin=thin,
            nsims=nsims,
            epsilon=epsilon,
            device=str(device),
        )
        
        refinement_time = time.time() - t_refine_start
        print(f"Refinement finished in {refinement_time:.2f}s")
        
        # Reshape refined samples: (n_chains * n_samples, d)
        refined_samples_flat = refined_samples.reshape(-1, d)
        
        print(f"Refined Mean: {np.mean(refined_samples_flat, axis=0)}")
        
        # Save Refined Samples
        np.save(f"{save_dir_samples}/refined.npy", refined_samples_flat)
        print(f"Saved refined samples to {save_dir_samples}/refined.npy")
        
        # Plot Refined Posterior
        plot_posterior(refined_samples_flat, theta_true, plot_dir, "refined_posterior.png")
        
        # Plot Comparison (Amortized vs Refined)
        samples_comp = {
            "Amortized": posterior_samples,
            "Refined": refined_samples_flat
        }
        plot_refinement_comparison(samples_comp, theta_true, plot_dir, model_type, true_posterior_samples=true_posterior_samples)
        
        total_time = training_time + refinement_time # Though for these we log separately
    
    return {
        "model": model,
        "loss_history": loss_history,
        "amortized_samples": posterior_samples,
        "refined_samples": refined_samples_flat,
        "training_time": training_time,
        "refinement_time": refinement_time,
        "total_time": total_time
    }

def save_model(model, model_type, round_id):
    """Saves the model state."""
    save_dir = f"saved_models_{n}/{model_type}"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/round_{round_id}.pt"
    
    if model_type == "bayesflow":
        try:
             model.save(f"{save_dir}/round_{round_id}.keras")
             print(f"Saved BayesFlow model to {save_dir}/round_{round_id}.keras")
        except Exception as e:
             print(f"Failed to save BayesFlow model (full): {e}")
             try:
                 model.save_weights(f"{save_dir}/round_{round_id}.weights.h5")
                 print(f"Saved BayesFlow weights to {save_dir}/round_{round_id}.weights.h5")
             except Exception as e2:
                 print(f"Failed to save BayesFlow weights: {e2}")
                 
                 try:
                     if hasattr(model, "summary_network"):
                         model.summary_network.save_weights(f"{save_dir}/round_{round_id}_summary.weights.h5")
                         print(f"Saved Summary Network weights to {save_dir}/round_{round_id}_summary.weights.h5")
                 except Exception as e3:
                     print(f"Failed to save Summary Network weights: {e3}")
                 
                 try:
                     if hasattr(model, "inference_network"):
                         dummy_x = torch.zeros((1, n, d_x), dtype=torch.float32)
                         x_emb = model.summary_network(dummy_x)
                         dummy_theta = torch.zeros((1, d), dtype=torch.float32)
                         if hasattr(model.inference_network, "log_prob"):
                             _ = model.inference_network.log_prob(dummy_theta, conditions=x_emb)
                         else:
                             _ = model.inference_network(dummy_theta, conditions=x_emb)
                         torch.save(model.inference_network.state_dict(), f"{save_dir}/round_{round_id}_inference.pt")
                         print(f"Saved Inference Network state_dict to {save_dir}/round_{round_id}_inference.pt")
                 except Exception as e4:
                     print(f"Failed to save Inference Network state_dict: {e4}")
    elif isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), filename)
        print(f"Saved {model_type} model to {filename}")
    else:
        print(f"Skipping save for {model_type} (not a torch module or bayesflow).")

def main():
    global RESULTS_BASE, n
    os.makedirs("saved_models", exist_ok=True)

    print(f"Starting Multi-Round Experiment (Total Rounds: {NUM_ROUNDS})")
    print(f"n_observation settings: {N_OBS_LIST}")

    for n_curr in N_OBS_LIST:
        n = n_curr
        RESULTS_BASE = f"results_{n_curr}"
        os.makedirs(RESULTS_BASE, exist_ok=True)

        all_results = []
        model_results_table = {}

        print(f"\n******** Running experiments for n_observation = {n_curr} ********")

        for round_idx in range(1, NUM_ROUNDS + 1):
            print(f"\n{'=' * 20} ROUND {round_idx}/{NUM_ROUNDS} {'=' * 20}")

            print("=== Step 1: Data Generation ===")
            task = GaussianTask(n=n)
            print(f"Generating training data (size={DATASET_SIZE}, n={n})...")
            theta_train, x_train = generate_dataset(
                task, n_sims=DATASET_SIZE, n_obs=n
            )

            dataset = TensorDataset(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(theta_train).float(),
            )
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            print("Generating observation...")
            theta_true, x_obs = task.get_ground_truth()
            print(f"True Params: {theta_true}")

            obs_save_dir = f"{RESULTS_BASE}/true_observations/round_{round_idx}"
            os.makedirs(obs_save_dir, exist_ok=True)
            np.save(f"{obs_save_dir}/observation.npy", x_obs)
            np.save(f"{obs_save_dir}/theta_true.npy", theta_true)
            print(f"Saved true observation to {obs_save_dir}")

            print("\n=== Step 2: True Posterior Sampling ===")
            pymc_samples = None
            try:
                X_comp = x_obs[:, 0]
                Y_comp = x_obs[:, 1]
                Z_comp = x_obs[:, 2]

                denom = 1.0 - Z_comp
                denom[np.abs(denom) < 1e-6] = 1e-6

                x_2d = X_comp / denom
                y_2d = Y_comp / denom
                x_obs_2d = np.stack([x_2d, y_2d], axis=-1)

                print(
                    f"Inverted 3D obs to 2D for true posterior sampling. "
                    f"Shape: {x_obs_2d.shape}"
                )

                method = TRUE_POSTERIOR_CONFIG.get("method", "mcmc").lower()
                if method == "pymc":
                    pymc_samples = run_pymc(
                        x_obs_2d,
                        n_draws=TRUE_POSTERIOR_CONFIG.get("n_draws", 2000),
                        n_tune=TRUE_POSTERIOR_CONFIG.get("n_tune_chain", 1000),
                        chains=TRUE_POSTERIOR_CONFIG.get("chains", 30),
                    )
                elif method == "mcmc":
                    base_scale = TRUE_POSTERIOR_CONFIG.get("proposal_scale", 0.06)
                    scale_factor = float(25.0 / n) ** 0.5
                    proposal_scale = base_scale * scale_factor
                    pymc_samples = run_gaussian_posterior_mcmc(
                        x_obs_2d,
                        task,
                        n_draws=TRUE_POSTERIOR_CONFIG.get("n_draws", 2000),
                        n_tune_chain=TRUE_POSTERIOR_CONFIG.get(
                            "n_tune_chain", 1000
                        ),
                        chains=TRUE_POSTERIOR_CONFIG.get("chains", 30),
                        proposal_scale=proposal_scale,
                    )
                else:
                    raise ValueError(
                        f"Unknown true_posterior_config.method '{method}'"
                    )

                print(f"PyMC Samples Shape: {pymc_samples.shape}")
                pymc_plot_dir = f"{RESULTS_BASE}/plots/round_{round_idx}/pymc"
                plot_posterior(pymc_samples, theta_true, pymc_plot_dir, "posterior.png")

                pymc_save_dir = (
                    f"{RESULTS_BASE}/posterior_samples/pymc_reference/round_{round_idx}"
                )
                os.makedirs(pymc_save_dir, exist_ok=True)
                np.save(f"{pymc_save_dir}/samples.npy", pymc_samples)
                print(f"Saved PyMC samples to {pymc_save_dir}/samples.npy")
            except Exception as e:
                print(f"PyMC Sampling failed: {e}")
                pymc_samples = None

            print("\n=== Step 3: Run Model Experiments ===")
            round_samples_for_plot = {}
            if pymc_samples is not None:
                round_samples_for_plot["PyMC"] = pymc_samples

            current_models_to_run = MODELS_TO_RUN
            if isinstance(current_models_to_run, str):
                current_models_to_run = [current_models_to_run]

            for model_type in current_models_to_run:
                if model_type not in MODELS_CONFIG:
                    print(f"Warning: No config for {model_type}, skipping.")
                    continue

                initial_data = (x_train, theta_train)

                result = run_single_experiment(
                    model_type,
                    task,
                    train_loader,
                    theta_true,
                    x_obs,
                    round_id=round_idx,
                    device=DEVICE,
                    n_obs=n,
                    initial_training_data=initial_data,
                    true_posterior_samples=pymc_samples,
                )

                if result is None:
                    continue

                if result["amortized_samples"] is not None:
                    round_samples_for_plot[model_type] = result["amortized_samples"]

                mmd_amortized = np.nan
                mmd_refined = np.nan

                if pymc_samples is not None:
                    if result["amortized_samples"] is not None:
                        mmd_amortized = compute_mmd_metric(
                            result["amortized_samples"], pymc_samples
                        )
                        print(
                            f"MMD ({model_type} Amortized): {mmd_amortized:.4f}"
                        )

                    if (
                        model_type in ["smmd", "mmd", "bayesflow"]
                        and result["refined_samples"] is not None
                    ):
                        mmd_refined = compute_mmd_metric(
                            result["refined_samples"], pymc_samples
                        )
                        print(
                            f"MMD ({model_type} Refined): {mmd_refined:.4f}"
                        )
                else:
                    print("Skipping MMD calculation (No PyMC samples).")

                save_model(result["model"], model_type, round_idx)

                record = {
                    "round": round_idx,
                    "model_name": model_type,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "training_time": result["training_time"],
                    "refinement_time": result["refinement_time"],
                    "total_time": result["total_time"],
                    "training_time_hms": format_time(result["training_time"]),
                    "refinement_time_hms": format_time(result["refinement_time"]),
                    "total_time_hms": format_time(result["total_time"]),
                    "mmd_amortized": mmd_amortized,
                    "mmd_refined": mmd_refined,
                }
                all_results.append(record)

                if model_type not in model_results_table:
                    model_results_table[model_type] = []
                model_results_table[model_type].append(record)

                pd.DataFrame(all_results).to_csv(
                    f"{RESULTS_BASE}/experiment_results.csv", index=False
                )

            print(f"Generating combined posterior plot for Round {round_idx}...")
            combined_plot_dir = f"{RESULTS_BASE}/plots/round_{round_idx}"
            plot_combined_posteriors(
                round_samples_for_plot,
                theta_true,
                combined_plot_dir,
                "combined_posteriors.png",
            )
            
            # --- New: Plot all refinement comparisons ---
            # Need to gather refinement data for all models
            # We don't have a direct dict for this yet, so let's reconstruct it from file or previous loop logic
            # Actually, `run_single_experiment` returns a dict with refined samples.
            # But we didn't store the full result dicts in a way that's easy to access all at once here 
            # (we appended to all_results list for CSV, but didn't keep the return values in a keyed dict)
            # Let's fix this by storing the return values in a dict during the loop
            
            # (Note: we need to change the loop above slightly to store the full result objects if we want to do this cleanly)
            # OR, we can just reload them from the saved .npy files, which is safer and decoupled.
            
            print(f"Generating all-model refinement comparison plot for Round {round_idx}...")
            all_refinement_data = {}
            refinable_models = ["smmd", "mmd", "bayesflow"]
            
            for m in refinable_models:
                # Paths
                base_sample_dir = f"{RESULTS_BASE}/posterior_samples/{m}/round_{round_idx}"
                p_amortized = f"{base_sample_dir}/amortized.npy"
                p_refined = f"{base_sample_dir}/refined.npy"
                
                if os.path.exists(p_amortized) and os.path.exists(p_refined):
                    try:
                        sam_amortized = np.load(p_amortized)
                        sam_refined = np.load(p_refined)
                        all_refinement_data[m] = {
                            "Amortized": sam_amortized,
                            "Refined": sam_refined
                        }
                    except Exception as e:
                        print(f"Error loading samples for {m}: {e}")
            
            if all_refinement_data:
                plot_all_refinement_comparisons(
                    all_refinement_data,
                    theta_true,
                    combined_plot_dir,
                    true_posterior_samples=pymc_samples
                )
            else:
                print("No refinement data found for all-model comparison.")

        print("\n=== Experiment Summary ===")
        if not model_results_table:
            print("Warning: No results to summarize.")
        else:
            os.makedirs(f"{RESULTS_BASE}/tables", exist_ok=True)
            summary_rows = []

            for model_name, rows in model_results_table.items():
                if not rows:
                    continue

                df_model = pd.DataFrame(rows)
                table_path = f"{RESULTS_BASE}/tables/{model_name}_results.csv"
                df_model.to_csv(table_path, index=False)
                print(f"Saved per-round table for {model_name} to {table_path}")

                per_round_rows = []
                for r in rows:
                    per_round_rows.append(
                        {
                            "round": r["round"],
                            "MMD_Amortized": r.get("mmd_amortized", np.nan),
                            "MMD_Refined": r.get("mmd_refined", np.nan),
                        }
                    )

                model_dir = f"{RESULTS_BASE}/models/{model_name}"
                os.makedirs(model_dir, exist_ok=True)
                per_round_df = pd.DataFrame(per_round_rows)
                per_round_metrics_path = (
                    f"{model_dir}/{model_name}_per_round_metrics.csv"
                )
                per_round_df.to_csv(per_round_metrics_path, index=False)
                print(
                    f"Saved per-round metrics for {model_name} to "
                    f"{per_round_metrics_path}"
                )

                mmd_am_vals = per_round_df["MMD_Amortized"].dropna().values
                mmd_ref_vals = per_round_df["MMD_Refined"].dropna().values

                mmd_am_mean = (
                    float(np.mean(mmd_am_vals)) if len(mmd_am_vals) > 0 else np.nan
                )
                mmd_am_median = (
                    float(np.median(mmd_am_vals))
                    if len(mmd_am_vals) > 0
                    else np.nan
                )
                mmd_ref_mean = (
                    float(np.mean(mmd_ref_vals)) if len(mmd_ref_vals) > 0 else np.nan
                )
                mmd_ref_median = (
                    float(np.median(mmd_ref_vals))
                    if len(mmd_ref_vals) > 0
                    else np.nan
                )

                summary_rows.append(
                    {
                        "Model": model_name,
                        "MMD_Amortized_Mean": mmd_am_mean,
                        "MMD_Amortized_Median": mmd_am_median,
                        "MMD_Refined_Mean": mmd_ref_mean,
                        "MMD_Refined_Median": mmd_ref_median,
                    }
                )

            if summary_rows:
                df_summary = pd.DataFrame(summary_rows)
                summary_path = f"{RESULTS_BASE}/final_summary.csv"
                df_summary.to_csv(summary_path, index=False)
                print(f"Saved final summary to {summary_path}")
            else:
                print("No valid summary rows were generated.")

    print("\nAll Experiments Completed.")

if __name__ == "__main__":
    main()
