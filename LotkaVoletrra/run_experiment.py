
"""
Run Experiment for Lotka-Volterra Task with SMMD/MMD/BayesFlow/DNNABC/W2ABC.
Migrated from Gaussian Task.
"""

import os
import sys
import time
from datetime import timedelta
import json
import numpy as np
import torch

# Fix for MPS 'aten::linalg_qr.out' not implemented error
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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
from data_generation import LVTask
from models.smmd import SMMD_Model, sliced_mmd_loss, mixture_sliced_mmd_loss
from models.mmd import MMD_Model, mmd_loss
from models.bayesflow_net import build_bayesflow_model
from models.dnnabc import DNNABC_Model, train_dnnabc, abc_rejection_sampling
from models.w2abc import run_w2abc
from models.sbi_wrappers import run_sbi_model
from utilities import compute_metrics, fit_kde_and_evaluate, refine_posterior

# Try importing BayesFlow (optional)
try:
    import bayesflow as bf
    BAYESFLOW_AVAILABLE = True
except ImportError:
    BAYESFLOW_AVAILABLE = False
    print("BayesFlow not installed or import failed. BayesFlow model will be unavailable.")

# ============================================================================
# 1. Configuration & Device
# ============================================================================

# Load Config
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
    print("✅ Loaded configuration from config.json")
except FileNotFoundError:
    print("⚠️ config.json not found. Using default configuration.")
    CONFIG = {
        "dataset_size": 20000,
        "val_size": 1000,
        "batch_size": 128,
        "num_rounds": 5,
        "models_to_run": ["smmd", "mmd", "bayesflow", "dnnabc", "w2abc"],
        "models_config": {},
        "n_observation": 151
    }

# Hyperparameters from Config
DATASET_SIZE = CONFIG.get("dataset_size", 20000)
VAL_SIZE = CONFIG.get("val_size", 1000)
BATCH_SIZE = CONFIG.get("batch_size", 128)
NUM_ROUNDS = CONFIG.get("num_rounds", 5)
MODELS_TO_RUN = CONFIG.get("models_to_run", ["smmd"])
MODELS_CONFIG = CONFIG.get("models_config", {})

# SMMD/MMD specific config
SMMD_MMD_CONFIG = CONFIG.get("smmd_mmd_config", {"M": 50, "L": 20})
M = SMMD_MMD_CONFIG.get("M", 50)
L = SMMD_MMD_CONFIG.get("L", 20)

N_TIME_STEPS = CONFIG.get("n_time_steps", 151)
DT = CONFIG.get("dt", 0.2)

LEARNING_RATE = CONFIG.get("learning_rate", 3e-4)
N_SAMPLES_POSTERIOR = CONFIG.get("n_samples_posterior", 1000)

# Task Dimensions
d = 4 # alpha, beta, gamma, delta
d_x = 2 # Prey, Predator
n_obs = N_TIME_STEPS 

# Check device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"✅ Using MPS (Apple Silicon) acceleration. Device: {DEVICE}")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ Using CUDA acceleration. Device: {DEVICE}")
else:
    DEVICE = torch.device("cpu")
    print(f"ℹ️ Using CPU. Device: {DEVICE}")

def get_scheduler(optimizer, epochs):
    """Cosine Decay Scheduler"""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

# ============================================================================
# 2. Training Loops
# ============================================================================

from scipy.stats import gaussian_kde

# L1 Penalty Factor
L1_LAMBDA = 1e-4

def train_smmd_mmd(model, train_loader, epochs, device, model_type="smmd", n_time_steps=151):
    """Training loop for SMMD and MMD models."""
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(optimizer, epochs)
    
    model.to(device)
    model.train()
    
    loss_history = []
    
    print(f"Starting training ({model_type})...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            theta_batch = batch[1].to(device)
            
            optimizer.zero_grad()
            
            with torch.enable_grad():
                # Generate fake parameters
                # Sample Z: (batch, M, d)
                z = torch.randn(x_batch.size(0), M, model.d, device=device)
                
                # Forward pass
                theta_fake = model(x_batch, z)
                
                if model_type == "smmd":
                    loss = sliced_mmd_loss(theta_batch, theta_fake, num_slices=L, n_time_steps=n_time_steps)
                elif model_type == "mmd":
                    loss = mmd_loss(theta_batch, theta_fake, n_time_steps=n_time_steps)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Add L1 Penalty to Generator weights only
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
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f}s")
    return loss_history, training_time

def train_bayesflow(train_loader, epochs, device, summary_dim=10):
    """Train BayesFlow model using Keras training loop."""
    if not BAYESFLOW_AVAILABLE:
        raise ImportError("BayesFlow is not available.")
        
    print("Starting training (BayesFlow)...")
    
    # Get n_obs from first batch
    first_batch = next(iter(train_loader))
    first_x, first_theta = first_batch[0], first_batch[1]
    
    n_current = first_x.shape[1]
    
    # Build model
    amortized_posterior = build_bayesflow_model(d, d_x, summary_dim=summary_dim)
    
    # Manual build
    try:
        dummy_dict = {
            "inference_variables": first_theta.float(),
            "summary_variables": first_x.float()
        }
        if hasattr(amortized_posterior, "adapter"):
             _ = amortized_posterior.adapter(dummy_dict)
        _ = amortized_posterior.log_prob(dummy_dict)
    except Exception as e:
        print(f"Manual build warning: {e}")
        
    # Compile
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
            
            # BayesFlow usually doesn't support weighted loss easily in standard train_step
            # If weights are present (len(batch) > 2), they are ignored here unless we use resampling strategy before training
            # or implement manual training loop.
            # For this implementation, we assume weighted training for BayesFlow is handled via Resampling 
            # (which happens before creating the loader).
            
            x_batch_cpu = x_batch
            theta_batch_cpu = theta_batch
            
            batch_dict = {
                "inference_variables": theta_batch_cpu,
                "summary_variables": x_batch_cpu
            }
            
            try:
                metrics = amortized_posterior.train_step(batch_dict)
                loss_val = metrics.get("loss", list(metrics.values())[0])
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
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f}s")
    
    return amortized_posterior, loss_history, training_time

def plot_posterior(theta_samples, theta_true, output_dir, filename="posterior.png"):
    """Plots marginal posteriors."""
    os.makedirs(output_dir, exist_ok=True)
    num_params = theta_samples.shape[1]
    cols = num_params
    
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1: axes = [axes]
    
    df = pd.DataFrame(theta_samples, columns=[f'theta{i+1}' for i in range(num_params)])
    
    for i in range(cols):
        ax = axes[i]
        param_name = f'theta{i+1}'
        param_symbol = rf'$\theta_{{{i+1}}}$'
        sns.kdeplot(data=df, x=param_name, fill=True, alpha=0.5, ax=ax, label='Posterior')
        if i < len(theta_true):
            ax.axvline(x=theta_true[i], color='red', linestyle='--', linewidth=2, label='True')
        ax.set_title(f'Marginal {param_symbol}')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def format_time(seconds):
    """Formats seconds to HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def plot_combined_posteriors(all_model_samples, theta_true, output_dir, filename="combined_posteriors.png"):
    """
    Plots posteriors from multiple models on one figure.
    all_model_samples: dict {model_name: samples}
    """
    os.makedirs(output_dir, exist_ok=True)
    if not all_model_samples:
        return

    num_params = next(iter(all_model_samples.values())).shape[1]
    cols = num_params
    
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
    if cols == 1: axes = [axes]
    
    # Define colors or styles if needed, but seaborn handles hue well if we dataframe it
    
    for i in range(cols):
        ax = axes[i]
        param_name = f'theta{i+1}'
        param_symbol = rf'$\theta_{{{i+1}}}$'
        
        for model_name, samples in all_model_samples.items():
            label_name = model_name
            if model_name.endswith("_refineplus"):
                base = model_name[:-11]
                label_name = f"{base.upper()}+Refine+"
            else:
                label_name = model_name.upper()
            sns.kdeplot(x=samples[:, i], label=label_name, ax=ax, alpha=0.3, fill=False, linewidth=2)
            
        if i < len(theta_true):
            ax.axvline(x=theta_true[i], color='black', linestyle='--', linewidth=2, label='True')
            
        ax.set_title(f'Marginal {param_symbol}')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_refinement_comparison(samples_dict, theta_true, output_dir, model_name):
    """
    Plots Initial vs Refined+ vs Refined(MCMC) for a single model.
    samples_dict: { 'Initial': samples, 'Refined+': samples, 'Refined (MCMC)': samples }
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_refinement_comparison.png"
    
    num_params = next(iter(samples_dict.values())).shape[1]
    cols = num_params
    
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 1: axes = [axes]
    
    for i in range(cols):
        ax = axes[i]
        param_name = f'theta{i+1}'
        param_symbol = rf'$\theta_{{{i+1}}}$'
        
        for label, samples in samples_dict.items():
            if samples is not None:
                sns.kdeplot(x=samples[:, i], label=label, ax=ax, alpha=0.3, fill=True)
            
        if i < len(theta_true):
            ax.axvline(x=theta_true[i], color='black', linestyle='--', linewidth=2, label='True')
            
        ax.set_title(f'{model_name.upper()} - {param_symbol}')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# ============================================================================
# 3. Experiment Runner
# ============================================================================

def run_single_experiment(model_type, task, train_loader, theta_true, x_obs, round_id, model_config={}):
    """
    Runs a single experiment: Train -> Sample -> Compute Metrics
    """
    print(f"\n=== Running {model_type.upper()} (Round {round_id}) ===")
    
    epochs = model_config.get("epochs", 30)
    
    model = None
    loss_history = []
    training_time = 0.0
    
    timings = {"initial": 0.0, "refined_plus": 0.0, "mcmc": 0.0}
    samples_dict = {}
    
    n_obs_curr = x_obs.shape[0] # Time steps
    
    # 1. Train
    if model_type == "smmd":
        summary_dim = model_config.get("summary_dim", 10)
        model = SMMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs_curr)
        loss_history, training_time = train_smmd_mmd(model, train_loader, epochs, DEVICE, "smmd", n_time_steps=n_obs_curr)
    elif model_type == "mmd":
        summary_dim = model_config.get("summary_dim", 10)
        model = MMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs_curr)
        loss_history, training_time = train_smmd_mmd(model, train_loader, epochs, DEVICE, "mmd", n_time_steps=n_obs_curr)
    elif model_type == "bayesflow":
        if BAYESFLOW_AVAILABLE:
            bf_device = torch.device("cpu") # Force CPU for Keras
            summary_dim = model_config.get("summary_dim", 10)
            model, loss_history, training_time = train_bayesflow(train_loader, epochs, bf_device, summary_dim=summary_dim)
        else:
            return None
    elif model_type == "dnnabc":
        model = DNNABC_Model(d=d, d_x=d_x, n_points=n_obs_curr)
        loss_history = train_dnnabc(model, train_loader, epochs, DEVICE)
        training_time = 0 # Tracked elsewhere or ignored for now
    elif model_type == "w2abc":
        print("W2-ABC: No training required.")
    elif model_type in ["snpe_a", "snpe_b", "npe"]:
        # SBI Models train and sample in one go (for this implementation)
        pass
    
    # 2. Sample (Amortized Inference / ABC)
    print(f"--- Sampling ({model_type}) ---")
    n_samples = N_SAMPLES_POSTERIOR
    
    t_start_sample = time.time()
    
    if model_type == "w2abc":
        max_populations = model_config.get("max_populations", 2)
        posterior_samples = run_w2abc(task, x_obs, n_samples=n_samples, max_populations=max_populations)
    elif model_type in ["snpe_a", "snpe_b", "npe", "snpe", "snpe_c"]:
        num_rounds = model_config.get("sbi_rounds", 1)
        sims_per_round = model_config.get("sims_per_round", 1000)
        max_epochs = model_config.get("epochs", 1000)
        posterior_samples = run_sbi_model(
            model_type, train_loader, x_obs, theta_true, task, 
            device=DEVICE, num_rounds=num_rounds, 
            sims_per_round=sims_per_round, max_epochs=max_epochs
        )
    elif model_type == "dnnabc":
        n_pool = model_config.get("n_pool", 100000)
        posterior_samples = abc_rejection_sampling(model, x_obs, task, n_samples=n_samples, n_pool=n_pool, device=DEVICE)
    elif model_type == "bayesflow":
        x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
        if x_obs_cpu.ndim == 2: x_obs_cpu = x_obs_cpu[np.newaxis, ...]
        
        # BayesFlow Sample
        post = model.sample(conditions={"summary_variables": x_obs_cpu}, num_samples=n_samples)
        if isinstance(post, dict): post = post["inference_variables"]
        posterior_samples = post.reshape(-1, d)
    elif hasattr(model, "sample_posterior"):
        posterior_samples = model.sample_posterior(x_obs, n_samples)
    else:
        print("Model cannot sample.")
        return None
        
    sampling_time = time.time() - t_start_sample
    
    # 3. Compute Metrics (Initial posterior)
    metrics = compute_metrics(posterior_samples, theta_true)
    metrics["training_time"] = training_time
    metrics["sampling_time"] = sampling_time
    metrics["stage"] = "initial"
    metrics_initial = metrics.copy()
    
    # Store Initial: total wall-clock for initial posterior (train + sample)
    timings["initial"] = training_time + sampling_time
    samples_dict["initial"] = posterior_samples
    
    # Save Samples & Plot
    # Structure: results/models/{model_type}/round_{round_id}/initial/
    base_dir = f"results/models/{model_type}/round_{round_id}/initial"
    os.makedirs(base_dir, exist_ok=True)
    
    np.save(f"{base_dir}/posterior_samples.npy", posterior_samples)
    plot_posterior(posterior_samples, theta_true, base_dir, "posterior_plot.png")
    
    print(f"Bias L2: {metrics['bias_l2']:.4f}")
    print(f"HDI Length per param: {metrics['hdi_length']}")
    print(f"Coverage per param: {metrics['coverage']}")
    
    # ========================================================================
    # 4. Save Initial Model
    # ========================================================================
    save_model_dir = f"results/models/{model_type}/round_{round_id}"
    os.makedirs(save_model_dir, exist_ok=True)
    
    if model_type == "bayesflow":
        try:
            # Keras 3 requires .weights.h5 extension
            model.save_weights(f"{save_model_dir}/initial_weights.weights.h5")
        except Exception as e:
            print(f"Failed to save BayesFlow weights: {e}")
    elif model_type in ["smmd", "mmd"]:
        torch.save(model.state_dict(), f"{save_model_dir}/initial_state_dict.pt")
        
    # ========================================================================
    # 5. Refined+ (Sequential Training) - SMMD/MMD/BayesFlow
    # ========================================================================
    run_refined_training = model_type in ["smmd", "mmd", "bayesflow"]
    
    # Store theta_init for MCMC
    theta_init_mcmc = None
    
    if run_refined_training:
        print(f"\n--- Starting Refined+ Training for {model_type} ---")
        t_refine_start = time.time()
        
        # rounds of Refined+ training
        num_refine_rounds = model_config.get("num_refine_rounds", 1)
        
        for refine_round_idx in range(1, num_refine_rounds + 1):
            print(f"\n[Refined+ Round {refine_round_idx}/{num_refine_rounds}]")
            
            # Determine Refined+ Mode
            refined_mode = model_config.get("refined_mode", 1)
            
            if refined_mode == 0:
                print(f"Refined+ Mode is 0. Skipping Refined+ training.")
                break
                
            # Mode 1 (New Standard): 50% Posterior + 50% Prior -> Unweighted Training -> Weighted Resampling
            N_new = DATASET_SIZE
            print(f"1. Sampling {N_new} parameters from current posterior...")
            
            theta_new = None
            if model_type == "bayesflow":
                # BayesFlow sampling
                x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
                if x_obs_cpu.ndim == 2: x_obs_cpu = x_obs_cpu[np.newaxis, ...]
                x_obs_rep = np.tile(x_obs_cpu, (N_new, 1, 1))
                post = model.sample(conditions={"summary_variables": x_obs_rep}, num_samples=1)
                if isinstance(post, dict): post = post["inference_variables"]
                theta_new = post.reshape(N_new, -1)
            elif hasattr(model, "sample_posterior"):
                theta_new = model.sample_posterior(x_obs, N_new)
                
            if isinstance(theta_new, torch.Tensor):
                theta_new = theta_new.cpu().numpy()
            
            if theta_new is None:
                print("Failed to sample for Refined+. Skipping.")
                continue
                
            print(f"2. Simulating new data (N={N_new})...")
            # Sample from q_train (initial trained model)
            theta_new = None
            if model_type == "bayesflow":
                x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
                if x_obs_cpu.ndim == 2: x_obs_cpu = x_obs_cpu[np.newaxis, ...]
                x_obs_rep = np.tile(x_obs_cpu, (N_new, 1, 1))
                post = model.sample(conditions={"summary_variables": x_obs_rep}, num_samples=1)
                if isinstance(post, dict): post = post["inference_variables"]
                theta_new = post.reshape(N_new, -1)
            elif hasattr(model, "sample_posterior"):
                theta_new = model.sample_posterior(x_obs, N_new)
                
            if isinstance(theta_new, torch.Tensor):
                theta_new = theta_new.cpu().numpy()
                
            if theta_new is None:
                print("Failed to sample from q_train for Refined+. Skipping.")
                continue

            # Simulate x from q_train
            x_new = task.simulator(theta_new)
            
            # Prior Data (Reuse old training data)
            print(f"3. Sampling {N_new} parameters from prior (reusing original training data)...")
            # Reuse original training data (theta_train, x_train)
            # Need to ensure N_new matches or we subsample/repeat
            # train_loader has the data. Let's extract it or use the variables from main loop if passed.
            # But run_single_experiment takes train_loader.
            # We can iterate train_loader to get all data.
            
            theta_old_list = []
            x_old_list = []
            for batch in train_loader:
                # batch is usually (x, theta) or (theta, x) depending on creation
                # TensorDataset(x_train_tensor, theta_train_tensor)
                # So batch[0] is x, batch[1] is theta
                x_b, t_b = batch[0], batch[1]
                theta_old_list.append(t_b)
                x_old_list.append(x_b)
            
            theta_old = torch.cat(theta_old_list).numpy()
            x_old = torch.cat(x_old_list).numpy()
            
            # Squeeze theta_old if necessary
            # TensorDataset might wrap theta as (batch, d) or (batch, 1, d)
            # We want (N, d)
            if theta_old.ndim == 3 and theta_old.shape[1] == 1:
                theta_old = theta_old.squeeze(1)
                
            # Check theta_new shape (should be (N, d))
            if theta_new.ndim == 3 and theta_new.shape[1] == 1:
                theta_new = theta_new.squeeze(1)
            
            # Print shapes for debug
            print(f"   Shape Check: theta_old={theta_old.shape}, theta_new={theta_new.shape}")
            print(f"   Shape Check: x_old={x_old.shape}, x_new={x_new.shape}")
            
            # If dataset size differs from N_new, adjust. 
            # Usually N_new = N (dataset size).
            if len(theta_old) != N_new:
                # Subsample or repeat
                indices = np.random.choice(len(theta_old), size=N_new, replace=True)
                theta_old = theta_old[indices]
                x_old = x_old[indices]

            # Fit KDE on q_train samples (theta_new) for weight calculation later
            # Wait, we need q_train(theta) density.
            # We approximate q_train(theta) using KDE on theta_new (samples from q_train).
            print("   Fitting KDE on q_train samples...")
            log_kde_q_train = fit_kde_and_evaluate(theta_new, theta_new) # Fit on theta_new
            # We need the KDE object to evaluate on FUTURE samples (theta_retrained).
            # So we should refactor fit_kde_and_evaluate or just use scipy directly here.
            kde_q_train_model = gaussian_kde(theta_new.T)
            
            # Combine
            theta_combined = np.concatenate([theta_old, theta_new], axis=0)
            x_combined = np.concatenate([x_old, x_new], axis=0)
            
            # No weights for training (Equation 1 in ReviseComment is sum of losses)
            # We just create a dataset of 2N
            print(f"   Combined Dataset: {len(theta_combined)} samples. (Unweighted for training)")
            
            # D. Retrain
            print("4. Retraining on combined dataset...")
            ds_retrain = TensorDataset(
                torch.from_numpy(x_combined).float(), 
                torch.from_numpy(theta_combined).float()
            )
            loader_retrain = DataLoader(ds_retrain, batch_size=BATCH_SIZE, shuffle=True)
            
            if model_type == "bayesflow":
                 # Retrain BayesFlow
                 # Note: train_bayesflow returns (model, history, time)
                 # We ignore the returned model since it updates in place (Keras)
                 _, loss_hist_retrain, time_retrain = train_bayesflow(loader_retrain, epochs=epochs, device="cpu", summary_dim=model_config.get("summary_dim", 10))
            else:
                 loss_hist_retrain, time_retrain = train_smmd_mmd(model, loader_retrain, epochs=epochs, device=DEVICE, model_type=model_type)
            
            print(f"   Retraining done in {time_retrain:.2f}s")
            
            timings["refined_plus"] += time_retrain
            
            # Save Retrained Model for this Refined+ Round
            if model_type == "bayesflow":
                 try:
                     model.save_weights(f"{save_model_dir}/sl_round{refine_round_idx}.weights.h5")
                 except:
                     pass
            else:
                 torch.save(model.state_dict(), f"{save_model_dir}/sl_state_dict_round{refine_round_idx}.pt")

            # Sample Retrained Model (Mixture Model)
            print(f"5. Sampling Retrained Model (Refined+ Round {refine_round_idx})...")
            
            theta_retrained = None
            if model_type == "bayesflow":
                 x_obs_rep = np.tile(x_obs_cpu, (n_samples, 1, 1))
                 post = model.sample(conditions={"summary_variables": x_obs_rep}, num_samples=1)
                 if isinstance(post, dict): post = post["inference_variables"]
                 theta_retrained = post.reshape(n_samples, -1)
            elif hasattr(model, "sample_posterior"):
                 theta_retrained = model.sample_posterior(x_obs, n_samples)
                 
            if isinstance(theta_retrained, torch.Tensor):
                theta_retrained = theta_retrained.cpu().numpy()
            
            # Calculate Weights for Resampling (to recover target posterior)
            # w = prior / (0.5 * q_train + 0.5 * prior)
            # Note: Denominator uses q_train (initial), NOT q_retrained!
            print("6. Calculating weights for resampling...")
            
            # Evaluate q_train density on theta_retrained
            # kde_q_train_model was fitted on theta_new (samples from q_train)
            kde_prob_q_train = kde_q_train_model(theta_retrained.T)
            
            log_prior_retrained = task.log_prior(theta_retrained)
            prior_prob_retrained = np.exp(log_prior_retrained)
            
            denom = 0.5 * kde_prob_q_train + 0.5 * prior_prob_retrained
            weights_resample = prior_prob_retrained / (denom + 1e-10)
            
            # Handle NaNs or Infs
            weights_resample = np.nan_to_num(weights_resample, nan=0.0, posinf=0.0, neginf=0.0)
            
            sum_w = np.sum(weights_resample)
            if sum_w < 1e-9:
                print("Warning: Sum of weights is close to 0. Using uniform weights.")
                weights_resample = np.ones_like(weights_resample) / len(weights_resample)
            else:
                weights_resample /= sum_w
            
            # Resample
            indices = np.random.choice(len(theta_retrained), size=n_samples, replace=True, p=weights_resample)
            theta_resampled = theta_retrained[indices]
            
            # Store for MCMC
            theta_init_mcmc = theta_resampled
            samples_dict["refined_plus_pre_mcmc"] = theta_resampled
                    
            theta_init_mcmc = theta_resampled
            samples_dict["refined_plus_pre_mcmc"] = theta_resampled
            
    # ========================================================================
            
    # ========================================================================
    # 6. ABC-MCMC Refinement (For SMMD/MMD/BayesFlow)
    # ========================================================================
    run_mcmc = model_type in ["smmd", "mmd", "bayesflow"]
    
    if run_mcmc:
        # Check Refined Mode for MCMC
        refined_mode = model_config.get("refined_mode", 1)
        if refined_mode == 0:
            print(f"Refined Mode is 0. Skipping ABC-MCMC for {model_type}.")
            return {
                "metrics": metrics,
                "timings": timings,
                "samples": samples_dict
            }
            
        print(f"\n--- Performing ABC-MCMC Local Refinement for {model_type} ---")
        t_mcmc_start = time.time()
        
        # Ensure model is in eval mode
        if hasattr(model, "eval"): model.eval()
        
        # Pass n_chains=n_samples and n_samples=1 to get exactly n_samples
        # Pass device explicitly
        mcmc_burn_in = model_config.get("mcmc_burn_in", 29)
        
        # Pass theta_init_mcmc if available (from Refined+)
        theta_mcmc = refine_posterior(model, x_obs, task=task, n_chains=n_samples, n_samples=1, 
                                     burn_in=mcmc_burn_in, device=DEVICE, theta_init=theta_init_mcmc) 
 
        
        time_mcmc = time.time() - t_mcmc_start
        print(f"   ABC-MCMC done in {time_mcmc:.2f}s")
        
        timings["mcmc"] = time_mcmc
        timings["refined_plus"] = timings["refined_plus"] + timings["mcmc"]
        samples_dict["mcmc"] = theta_mcmc
        samples_dict["refined_plus"] = theta_mcmc
        
        # Metrics MCMC
        metrics_mcmc = compute_metrics(theta_mcmc, theta_true)
        # Training time includes retrain time if applicable
        total_train_time = training_time
        if run_refined_training:
             # We didn't capture time_retrain variable scope outside if block easily without initialization
             # But we can approximate or use what we have. 
             # Let's just add it if it happened.
             pass 
             
        metrics_mcmc["training_time"] = total_train_time + time_mcmc
        metrics_mcmc["stage"] = "refined_mcmc"
        print(f"   MCMC Bias L2: {metrics_mcmc['bias_l2']:.4f}")
        print(f"   MCMC HDI Length (Mean): {np.mean(metrics_mcmc['hdi_length']):.4f}")
        print(f"   MCMC Coverage (Mean): {np.mean(metrics_mcmc['coverage']):.4f}")
        
        # Save Samples & Plot
        refine_dir = f"results/models/{model_type}/round_{round_id}/refineplus"
        os.makedirs(refine_dir, exist_ok=True)
        np.save(f"{refine_dir}/posterior_samples.npy", theta_mcmc)
        plot_posterior(theta_mcmc, theta_true, refine_dir, "posterior_plot.png")
        
        return {
            "metrics": metrics_mcmc,
            "timings": timings,
            "samples": samples_dict,
            "metrics_initial": metrics_initial
        }

    return {
        "metrics": metrics,
        "timings": timings,
        "samples": samples_dict,
        "metrics_initial": metrics_initial
    }

def main():
    os.makedirs("results", exist_ok=True)
    
    # Use global config
    MODELS = MODELS_TO_RUN
    
    # Aggregated Results
    model_results_table = {m: [] for m in MODELS}
    
    print(f"Starting Multi-Round Experiment (Rounds={NUM_ROUNDS})")
    
    for round_idx in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*20} ROUND {round_idx}/{NUM_ROUNDS} {'='*20}")
        
        # 1. Data Generation
        # Calculate t_max based on n_time_steps from config (assuming dt is fixed or from config)
        # n_obs = T/dt + 1 => T = (n_obs - 1) * dt
        dt = DT
        t_max = (N_TIME_STEPS - 1) * dt
        task = LVTask(t_max=t_max, dt=dt)
        
        print(f"Generating training data (N={DATASET_SIZE}, T_MAX={t_max:.1f}, Steps={N_TIME_STEPS})...")
        theta_train = task.sample_prior(DATASET_SIZE, "vague") # Use vague prior for training
        x_train = task.simulator(theta_train)
        
        dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(theta_train).float())
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Ground Truth for this round
        theta_true, x_obs = task.get_ground_truth()
        
        round_initial_samples = {}
        round_refineplus_samples = {}
        
        # Run Models
        for model_name in MODELS:
            try:
                # Get model config
                model_conf = MODELS_CONFIG.get(model_name, {})
                if isinstance(model_conf, int): # Handle legacy format if any
                    model_conf = {"epochs": model_conf}
                
                res = run_single_experiment(model_name, task, train_loader, theta_true, x_obs, round_idx, model_config=model_conf)
                
                if res:
                    metrics = res["metrics"]
                    timings = res["timings"]
                    samples = res["samples"]
                    
                    if "initial" in samples:
                        round_initial_samples[model_name] = samples["initial"]
                    
                    if model_name in ["smmd", "mmd", "bayesflow"]:
                        final_refined = None
                        for key in ["refined_plus", "mcmc", "refined_plus_pre_mcmc"]:
                            value = samples.get(key)
                            if value is not None:
                                final_refined = value
                                break
                        if final_refined is not None:
                            round_refineplus_samples[f"{model_name}_refineplus"] = final_refined
                            
                            plot_samples = {
                                "Initial": samples.get("initial"),
                                "Refine+": final_refined
                            }
                            plot_samples = {k: v for k, v in plot_samples.items() if v is not None}
                            
                            plot_dir = f"results/models/{model_name}/round_{round_idx}/comparison"
                            plot_refinement_comparison(plot_samples, theta_true, plot_dir, model_name)

                    metrics_initial = res.get("metrics_initial")
                    
                    row = metrics.copy()
                    row["round"] = round_idx
                    row["status"] = 1
                    
                    if metrics_initial is not None:
                        if "bias_l2" in metrics_initial:
                            row["bias_l2_initial"] = metrics_initial["bias_l2"]
                        if "hdi_length" in metrics_initial:
                            row["hdi_length_initial"] = metrics_initial["hdi_length"]
                        if "coverage" in metrics_initial:
                            row["coverage_initial"] = metrics_initial["coverage"]
                    else:
                        if "bias_l2" in metrics:
                            row["bias_l2_initial"] = metrics["bias_l2"]
                        if "hdi_length" in metrics:
                            row["hdi_length_initial"] = metrics["hdi_length"]
                        if "coverage" in metrics:
                            row["coverage_initial"] = metrics["coverage"]
                    
                    if model_name in ["smmd", "mmd", "bayesflow"]:
                        if "bias_l2" in metrics:
                            row["bias_l2_refineplus"] = metrics["bias_l2"]
                        if "hdi_length" in metrics:
                            row["hdi_length_refineplus"] = metrics["hdi_length"]
                        if "coverage" in metrics:
                            row["coverage_refineplus"] = metrics["coverage"]
                    
                    # Add formatted times
                    row["time_initial"] = format_time(timings["initial"])
                    row["time_refined_plus"] = format_time(timings["refined_plus"])
                    row["time_mcmc"] = format_time(timings["mcmc"])
                    
                    # Keep raw seconds for averaging
                    row["seconds_initial"] = timings["initial"]
                    row["seconds_refined_plus"] = timings["refined_plus"]
                    row["seconds_mcmc"] = timings["mcmc"]
                    
                    model_results_table[model_name].append(row)
                    
            except Exception as e:
                print(f"Error running {model_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Record failure
                row = {
                    "round": round_idx,
                    "status": 0, # Failure
                    "error_msg": str(e)
                }
                model_results_table[model_name].append(row)

        if round_initial_samples:
            print("Generating combined posterior plot...")
            plot_combined_posteriors(round_initial_samples, theta_true, f"results/comparisons/round_{round_idx}", "all_methods_initial_posterior.png")
        
        if round_refineplus_samples:
            combined_samples = dict(round_initial_samples)
            combined_samples.update(round_refineplus_samples)
            plot_combined_posteriors(combined_samples, theta_true, f"results/comparisons/round_{round_idx}", "all_methods_with_refineplus_posterior.png")

    # ============================================================================
    # 4. Final Aggregation
    # ============================================================================
    print("\n" + "="*30)
    print("FINAL AGGREGATED RESULTS")
    print("="*30)
    
    os.makedirs("results/tables", exist_ok=True)
    summary_data = []
    
    for model_name, rows in model_results_table.items():
        if not rows:
            continue
            
        # Save raw per-round table (for debugging)
        df_model = pd.DataFrame(rows)
        cols = ["round", "status", "stage", "bias_l2", "hdi_length", "coverage", 
                "time_initial", "time_refined_plus", "time_mcmc"]
        other_cols = [c for c in df_model.columns if c not in cols and not c.startswith("seconds_")]
        final_cols = cols + other_cols
        final_cols = [c for c in final_cols if c in df_model.columns]
        df_model = df_model[final_cols]
        df_model.to_csv(f"results/tables/{model_name}_results.csv", index=False)
        print(f"Saved table for {model_name} to results/tables/{model_name}_results.csv")
            
        success_rows = [r for r in rows if r.get("status", 0) == 1]
        if not success_rows:
            print(f"No successful runs for {model_name}")
            continue
        
        # Determine parameter dimension from any successful row
        example = None
        for r in success_rows:
            if "hdi_length_initial" in r or "hdi_length" in r:
                example = r
                break
        if example is None:
            print(f"No HDI information for {model_name}, skipping aggregation.")
            continue
        if "hdi_length_initial" in example:
            example_hdi = np.asarray(example["hdi_length_initial"], dtype=float)
        else:
            example_hdi = np.asarray(example["hdi_length"], dtype=float)
        num_params = example_hdi.shape[0]
        
        # Build per-round metrics table under results/models/{model_name}
        per_round_rows = []
        refine_models = ["smmd", "mmd", "bayesflow"]
        
        for r in rows:
            round_id = r.get("round", np.nan)
            status = r.get("status", 0)
            
            if status != 1:
                row_metrics = {
                    "round": round_id,
                    "Bias": np.nan,
                    "RefinePlus_Bias": np.nan,
                }
                for j in range(num_params):
                    row_metrics[f"HDI_Len_Param{j+1}"] = np.nan
                    row_metrics[f"Coverage_Param{j+1}"] = np.nan
                    row_metrics[f"RefinePlus_HDI_Len_Param{j+1}"] = np.nan
                    row_metrics[f"RefinePlus_Coverage_Param{j+1}"] = np.nan
                per_round_rows.append(row_metrics)
                continue
            
            if "bias_l2_initial" in r:
                bias_init = r["bias_l2_initial"]
            elif "bias_l2" in r:
                bias_init = r["bias_l2"]
            else:
                bias_init = np.nan
            
            if model_name in refine_models and "bias_l2_refineplus" in r:
                bias_ref = r["bias_l2_refineplus"]
            else:
                bias_ref = np.nan
            
            if "hdi_length_initial" in r:
                hdi_init = np.asarray(r["hdi_length_initial"], dtype=float)
            elif "hdi_length" in r:
                hdi_init = np.asarray(r["hdi_length"], dtype=float)
            else:
                hdi_init = np.full(num_params, np.nan)
            
            if model_name in refine_models and "hdi_length_refineplus" in r:
                hdi_ref = np.asarray(r["hdi_length_refineplus"], dtype=float)
            else:
                hdi_ref = np.full(num_params, np.nan)
            
            if "coverage_initial" in r:
                cov_init = np.asarray(r["coverage_initial"], dtype=float)
            elif "coverage" in r:
                cov_init = np.asarray(r["coverage"], dtype=float)
            else:
                cov_init = np.full(num_params, np.nan)
            
            if model_name in refine_models and "coverage_refineplus" in r:
                cov_ref = np.asarray(r["coverage_refineplus"], dtype=float)
            else:
                cov_ref = np.full(num_params, np.nan)
            
            row_metrics = {
                "round": round_id,
                "Bias": bias_init,
                "RefinePlus_Bias": bias_ref,
            }
            for j in range(num_params):
                row_metrics[f"HDI_Len_Param{j+1}"] = hdi_init[j]
                row_metrics[f"Coverage_Param{j+1}"] = cov_init[j]
                row_metrics[f"RefinePlus_HDI_Len_Param{j+1}"] = hdi_ref[j]
                row_metrics[f"RefinePlus_Coverage_Param{j+1}"] = cov_ref[j]
            
            per_round_rows.append(row_metrics)
        
        per_round_df = pd.DataFrame(per_round_rows)
        model_dir = f"results/models/{model_name}"
        os.makedirs(model_dir, exist_ok=True)
        per_round_path = f"{model_dir}/{model_name}_per_round_metrics.csv"
        per_round_df.to_csv(per_round_path, index=False)
        print(f"Saved per-round metrics for {model_name} to {per_round_path}")
        
        # Aggregated statistics for final_summary
        bias_initial = []
        bias_refineplus = []
        hdi_initial_list = []
        hdi_refineplus_list = []
        coverage_initial_list = []
        coverage_refineplus_list = []
        
        for r in success_rows:
            if "bias_l2_initial" in r:
                bias_initial.append(r["bias_l2_initial"])
            elif "bias_l2" in r:
                bias_initial.append(r["bias_l2"])
            if model_name in refine_models and "bias_l2_refineplus" in r:
                bias_refineplus.append(r["bias_l2_refineplus"])
            
            if "hdi_length_initial" in r:
                hdi_initial_list.append(np.asarray(r["hdi_length_initial"], dtype=float))
            elif "hdi_length" in r:
                hdi_initial_list.append(np.asarray(r["hdi_length"], dtype=float))
            if model_name in refine_models and "hdi_length_refineplus" in r:
                hdi_refineplus_list.append(np.asarray(r["hdi_length_refineplus"], dtype=float))
            
            if "coverage_initial" in r:
                coverage_initial_list.append(np.asarray(r["coverage_initial"], dtype=float))
            elif "coverage" in r:
                coverage_initial_list.append(np.asarray(r["coverage"], dtype=float))
            if model_name in refine_models and "coverage_refineplus" in r:
                coverage_refineplus_list.append(np.asarray(r["coverage_refineplus"], dtype=float))
        
        hdi_initial_arr = np.vstack(hdi_initial_list)
        avg_hdi_len_initial = hdi_initial_arr.mean(axis=0)
        
        coverage_initial_arr = np.vstack(coverage_initial_list)
        avg_coverage_initial = coverage_initial_arr.mean(axis=0)
        
        if hdi_refineplus_list:
            hdi_refineplus_arr = np.vstack(hdi_refineplus_list)
            avg_hdi_len_refineplus = hdi_refineplus_arr.mean(axis=0)
        else:
            avg_hdi_len_refineplus = np.full_like(avg_hdi_len_initial, np.nan)
        
        if coverage_refineplus_list:
            coverage_refineplus_arr = np.vstack(coverage_refineplus_list)
            avg_coverage_refineplus = coverage_refineplus_arr.mean(axis=0)
        else:
            avg_coverage_refineplus = np.full_like(avg_coverage_initial, np.nan)
        
        avg_sec_initial = np.mean([r["seconds_initial"] for r in success_rows])
        avg_sec_sl = np.mean([r["seconds_refined_plus"] for r in success_rows])
        avg_sec_mcmc = np.mean([r["seconds_mcmc"] for r in success_rows])
        
        mean_bias_initial = np.mean(bias_initial)
        median_bias_initial = np.median(bias_initial)
        
        if bias_refineplus:
            mean_bias_refineplus = np.mean(bias_refineplus)
            median_bias_refineplus = np.median(bias_refineplus)
        else:
            mean_bias_refineplus = np.nan
            median_bias_refineplus = np.nan
        
        print(f"\nModel: {model_name.upper()}")
        print(f"  Bias L2 (Initial): Mean={mean_bias_initial:.4f}")
        if not np.isnan(mean_bias_refineplus):
            print(f"  Bias L2 (Refine+): Mean={mean_bias_refineplus:.4f}")
        print(f"  HDI Len (Initial) per param: {avg_hdi_len_initial}")
        if not np.all(np.isnan(avg_hdi_len_refineplus)):
            print(f"  HDI Len (Refine+) per param: {avg_hdi_len_refineplus}")
        print(f"  Coverage (Initial) per param: {avg_coverage_initial}")
        if not np.all(np.isnan(avg_coverage_refineplus)):
            print(f"  Coverage (Refine+) per param: {avg_coverage_refineplus}")
        print(f"  Avg Time (Initial): {format_time(avg_sec_initial)}")
        if avg_sec_sl > 0:
            print(f"  Avg Time (Refined+): {format_time(avg_sec_sl)}")
        if avg_sec_mcmc > 0:
            print(f"  Avg Time (MCMC): {format_time(avg_sec_mcmc)}")
        
        summary_row = {
            "Model": model_name,
            "Bias_Mean": mean_bias_initial,
            "RefinePlus_Bias_Mean": mean_bias_refineplus,
            "Bias_Median": median_bias_initial,
            "RefinePlus_Bias_Median": median_bias_refineplus,
            "Avg_Time_Initial": format_time(avg_sec_initial),
            "Avg_Time_RefinedPlus": format_time(avg_sec_sl),
            "Avg_Time_MCMC": format_time(avg_sec_mcmc),
        }
        
        for j in range(num_params):
            summary_row[f"HDI_Len_Param{j+1}_Mean"] = avg_hdi_len_initial[j]
            summary_row[f"Coverage_Param{j+1}_Mean"] = avg_coverage_initial[j]
            summary_row[f"RefinePlus_HDI_Len_Param{j+1}_Mean"] = avg_hdi_len_refineplus[j]
            summary_row[f"RefinePlus_Coverage_Param{j+1}_Mean"] = avg_coverage_refineplus[j]
        
        summary_data.append(summary_row)
        
    # Save Summary
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("results/final_summary.csv", index=False)
    print("\nSummary saved to results/final_summary.csv")

if __name__ == "__main__":
    main()
