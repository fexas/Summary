
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
from models.smmd import SMMD_Model, sliced_mmd_loss
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

LEARNING_RATE = CONFIG.get("learning_rate", 1e-3)
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

def train_smmd_mmd(model, train_loader, epochs, device, model_type="smmd"):
    """Training loop for SMMD and MMD models."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            # Check for weights
            weights_batch = batch[2].to(device) if len(batch) > 2 else None
            
            optimizer.zero_grad()
            
            with torch.enable_grad():
                # Generate fake parameters
                # Sample Z: (batch, M, d)
                z = torch.randn(x_batch.size(0), M, model.d, device=device)
                
                # Forward pass
                theta_fake = model(x_batch, z)
                
                # Compute Loss
                if model_type == "smmd":
                    loss = sliced_mmd_loss(theta_batch, theta_fake, num_slices=L, n_points=M, weights=weights_batch)
                elif model_type == "mmd":
                    loss = mmd_loss(theta_batch, theta_fake, n_points=M, weights=weights_batch)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
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
        sns.kdeplot(data=df, x=param_name, fill=True, alpha=0.5, ax=ax, label='Posterior')
        if i < len(theta_true):
            ax.axvline(x=theta_true[i], color='red', linestyle='--', linewidth=2, label='True')
        ax.set_title(f'Marginal {param_name}')
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
    
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 1: axes = [axes]
    
    # Define colors or styles if needed, but seaborn handles hue well if we dataframe it
    
    for i in range(cols):
        ax = axes[i]
        param_name = f'theta{i+1}'
        
        for model_name, samples in all_model_samples.items():
            sns.kdeplot(x=samples[:, i], label=model_name.upper(), ax=ax, alpha=0.3, fill=False, linewidth=2)
            
        if i < len(theta_true):
            ax.axvline(x=theta_true[i], color='black', linestyle='--', linewidth=2, label='True')
            
        ax.set_title(f'Marginal {param_name}')
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
        
        for label, samples in samples_dict.items():
            if samples is not None:
                sns.kdeplot(x=samples[:, i], label=label, ax=ax, alpha=0.3, fill=True)
            
        if i < len(theta_true):
            ax.axvline(x=theta_true[i], color='black', linestyle='--', linewidth=2, label='True')
            
        ax.set_title(f'{model_name.upper()} - {param_name}')
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
    
    # Trackers
    timings = {"initial": 0.0, "refined_plus": 0.0, "mcmc": 0.0}
    samples_dict = {}
    
    n_obs_curr = x_obs.shape[0] # Time steps
    
    # 1. Train
    if model_type == "smmd":
        summary_dim = model_config.get("summary_dim", 10)
        model = SMMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs_curr)
        loss_history, training_time = train_smmd_mmd(model, train_loader, epochs, DEVICE, "smmd")
    elif model_type == "mmd":
        summary_dim = model_config.get("summary_dim", 10)
        model = MMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=n_obs_curr)
        loss_history, training_time = train_smmd_mmd(model, train_loader, epochs, DEVICE, "mmd")
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
    elif model_type in ["snpe_a", "snpe_b", "npe"]:
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
    
    # 3. Compute Metrics
    metrics = compute_metrics(posterior_samples, theta_true)
    metrics["training_time"] = training_time
    metrics["sampling_time"] = sampling_time
    metrics["stage"] = "initial"
    
    # Store Initial
    timings["initial"] = training_time
    samples_dict["initial"] = posterior_samples
    
    # Save Samples & Plot
    # Structure: results/models/{model_type}/round_{round_id}/initial/
    base_dir = f"results/models/{model_type}/round_{round_id}/initial"
    os.makedirs(base_dir, exist_ok=True)
    
    np.save(f"{base_dir}/posterior_samples.npy", posterior_samples)
    plot_posterior(posterior_samples, theta_true, base_dir, "posterior_plot.png")
    
    print(f"Bias L2: {metrics['bias_l2']:.4f}")
    print(f"HDI Length (Mean): {np.mean(metrics['hdi_length']):.4f}")
    print(f"Coverage (Mean): {np.mean(metrics['coverage']):.4f}")
    
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
    # 5. Refined+ (Sequential Training) - Only for SMMD/MMD
    # ========================================================================
    run_refined_training = model_type in ["smmd", "mmd"]
    
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
                
            print(f"Refined+ Mode: {refined_mode}")
            
            theta_combined = None
            x_combined = None
            weights_combined = None
            
            if refined_mode == 1:
                # Mode 1: 50% Posterior, 50% Prior (Original)
                N_new = DATASET_SIZE
                print(f"1. Sampling {N_new} parameters from current posterior (Mode 1)...")
                
                theta_new = None
                if hasattr(model, "sample_posterior"):
                    theta_new = model.sample_posterior(x_obs, N_new)
                    if isinstance(theta_new, torch.Tensor):
                        theta_new = theta_new.cpu().numpy()
                
                if theta_new is None:
                    print("Failed to sample for Refined+. Skipping.")
                    continue
                    
                print(f"2. Simulating new data (N={N_new})...")
                x_new = task.simulator(theta_new)
                
                print("3. Calculating weights (Mode 1)...")
                
                # Fit KDE on theta_new
                log_kde_new = fit_kde_and_evaluate(theta_new, theta_new) # log q(theta)
                kde_prob_new = np.exp(log_kde_new)
                
                # Prior prob
                log_prior_new = task.log_prior(theta_new)
                prior_prob_new = np.exp(log_prior_new)
                
                # Weights for NEW data
                # Denominator: 0.5 * KDE + 0.5 * Prior
                denom_new = 0.5 * kde_prob_new + 0.5 * prior_prob_new
                weights_new = prior_prob_new / (denom_new + 1e-10)
                
                # Weights for OLD data (from Prior)
                print(f"   Reusing previous training data...")
                # Extract from train_loader
                if hasattr(train_loader.dataset, 'tensors'):
                    x_old = train_loader.dataset.tensors[0].numpy()
                    theta_old = train_loader.dataset.tensors[1].numpy()
                else:
                    # Fallback if not TensorDataset or other structure
                    print("Warning: Could not extract data from loader. Resampling from prior.")
                    theta_old = task.sample_prior(N_new, "vague")
                    x_old = task.simulator(theta_old)
                
                log_kde_old = fit_kde_and_evaluate(theta_new, theta_old) 
                kde_prob_old = np.exp(log_kde_old)
                
                prior_prob_old = np.exp(task.log_prior(theta_old))
                denom_old = 0.5 * kde_prob_old + 0.5 * prior_prob_old
                weights_old = prior_prob_old / (denom_old + 1e-10)
                
                # Combine
                theta_combined = np.concatenate([theta_old, theta_new], axis=0)
                x_combined = np.concatenate([x_old, x_new], axis=0)
                weights_combined = np.concatenate([weights_old, weights_new], axis=0)
                
            elif refined_mode == 2:
                # Mode 2: 100% Posterior
                N_new = DATASET_SIZE
                print(f"1. Sampling {N_new} parameters from current posterior (Mode 2)...")
                
                theta_new = None
                if hasattr(model, "sample_posterior"):
                    theta_new = model.sample_posterior(x_obs, N_new)
                    if isinstance(theta_new, torch.Tensor):
                        theta_new = theta_new.cpu().numpy()
                        
                if theta_new is None:
                    print("Failed to sample for Refined+. Skipping.")
                    continue
                    
                print(f"2. Simulating new data (N={N_new})...")
                x_new = task.simulator(theta_new)
                
                print("3. Calculating weights (Mode 2)...")
                
                # Fit KDE on theta_new
                log_kde_new = fit_kde_and_evaluate(theta_new, theta_new)
                kde_prob_new = np.exp(log_kde_new)
                
                log_prior_new = task.log_prior(theta_new)
                prior_prob_new = np.exp(log_prior_new)
                
                # Weights = p(theta) / q(theta)
                weights_new = prior_prob_new / (kde_prob_new + 1e-10)
                
                theta_combined = theta_new
                x_combined = x_new
                weights_combined = weights_new
                
            else:
                print(f"Unknown refined_mode: {refined_mode}. Skipping Refined+ round.")
                continue
            
            # Normalize weights
            weights_combined = weights_combined / np.mean(weights_combined)
            
            print(f"   Combined Dataset: {len(theta_combined)} samples. Weights Mean: {np.mean(weights_combined):.4f}")
            
            # D. Retrain
            print("4. Retraining with weighted dataset...")
            ds_retrain = TensorDataset(
                torch.from_numpy(x_combined).float(), 
                torch.from_numpy(theta_combined).float(),
                torch.from_numpy(weights_combined).float()
            )
            loader_retrain = DataLoader(ds_retrain, batch_size=BATCH_SIZE, shuffle=True)
            
            loss_hist_retrain, time_retrain = train_smmd_mmd(model, loader_retrain, epochs=epochs, device=DEVICE, model_type=model_type)
            print(f"   Retraining done in {time_retrain:.2f}s")
            
            timings["refined_plus"] += time_retrain
            
            # Save Retrained Model for this Refined+ Round
            save_path = f"{save_model_dir}/sl_state_dict.pt"
            torch.save(model.state_dict(), save_path)
            print(f"   Saved model to {save_path}")
            
            # Sample Retrained Model
            print(f"5. Sampling Retrained Model (Refined+ Round {refine_round_idx})...")
            theta_retrained = model.sample_posterior(x_obs, n_samples)
            if isinstance(theta_retrained, torch.Tensor):
                theta_retrained = theta_retrained.cpu().numpy()
            
            samples_dict["refined_plus"] = theta_retrained
                    
            # Metrics Retrained
            metrics_retrained = compute_metrics(theta_retrained, theta_true)
            metrics_retrained["training_time"] = training_time + time_retrain # Accumulate? Ideally yes but simplified
            metrics_retrained["stage"] = f"refined_round{refine_round_idx}"
            print(f"   Retrained (Round {refine_round_idx}) Bias L2: {metrics_retrained['bias_l2']:.4f}")
            
            # Save Samples & Plot
            refined_dir = f"results/models/{model_type}/round_{round_id}/refined_round_{refine_round_idx}"
            os.makedirs(refined_dir, exist_ok=True)
            np.save(f"{refined_dir}/posterior_samples.npy", theta_retrained)
            plot_posterior(theta_retrained, theta_true, refined_dir, "posterior_plot.png")
            
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
        theta_mcmc = refine_posterior(model, x_obs, task=task, n_chains=n_samples, n_samples=1, burn_in=mcmc_burn_in, device=DEVICE) 
        
        time_mcmc = time.time() - t_mcmc_start
        print(f"   ABC-MCMC done in {time_mcmc:.2f}s")
        
        timings["mcmc"] = time_mcmc
        samples_dict["mcmc"] = theta_mcmc
        
        # Metrics MCMC
        metrics_mcmc = compute_metrics(theta_mcmc, theta_true)
        # Training time includes retrain time if applicable
        total_train_time = training_time
        if run_refined_training:
             # We didn't capture time_retrain variable scope outside if block easily without initialization
             # But we can approximate or use what we have. 
             # Let's just add it if it happened.
             pass 
             
        metrics_mcmc["training_time"] = total_train_time + time_mcmc # + retrain time ideally
        metrics_mcmc["stage"] = "refined_mcmc"
        print(f"   MCMC Bias L2: {metrics_mcmc['bias_l2']:.4f}")
        
        # Save Samples & Plot
        mcmc_dir = f"results/models/{model_type}/round_{round_id}/abc_mcmc"
        os.makedirs(mcmc_dir, exist_ok=True)
        np.save(f"{mcmc_dir}/posterior_samples.npy", theta_mcmc)
        plot_posterior(theta_mcmc, theta_true, mcmc_dir, "posterior_plot.png")
        
        return {
            "metrics": metrics_mcmc,
            "timings": timings,
            "samples": samples_dict
        }

    return {
        "metrics": metrics,
        "timings": timings,
        "samples": samples_dict
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
        
        # Storage for Combined Plot (Initial Posteriors)
        round_initial_samples = {}
        
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
                    
                    # Store for Plot 1 (Combined Initial)
                    if "initial" in samples:
                        round_initial_samples[model_name] = samples["initial"]
                        
                    # Plot 2: Refinement Comparison (Initial vs SL vs MCMC)
                    # For SMMD/MMD/Bayesflow
                    if model_name in ["smmd", "mmd", "bayesflow"]:
                         # Check if we have refinement samples
                         if "refined_plus" in samples or "mcmc" in samples:
                             # Create labels mapping
                             plot_samples = {
                                 "Initial": samples.get("initial"),
                                 "Refined+": samples.get("refined_plus"),
                                 "Refined (MCMC)": samples.get("mcmc")
                             }
                             # Remove None
                             plot_samples = {k: v for k, v in plot_samples.items() if v is not None}
                             
                             plot_dir = f"results/models/{model_name}/round_{round_idx}/comparison"
                             plot_refinement_comparison(plot_samples, theta_true, plot_dir, model_name)

                    # Prepare Row for Table
                    row = metrics.copy()
                    row["round"] = round_idx
                    row["status"] = 1 # Success
                    # Add formatted times
                    row["time_initial"] = format_time(timings["initial"])
                    row["time_refined_plus"] = format_time(timings["refined_plus"])
                    row["time_mcmc"] = format_time(timings["mcmc"])
                    
                    # Keep raw seconds for averaging if needed later, or just rely on re-parsing
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

        # End of Round: Plot Combined Posteriors
        if round_initial_samples:
            print("Generating combined posterior plot...")
            plot_combined_posteriors(round_initial_samples, theta_true, f"results/comparisons/round_{round_idx}", "all_methods_initial_posterior.png")

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
            
        # Save per-model table
        df_model = pd.DataFrame(rows)
        # Reorder columns for clarity
        cols = ["round", "status", "stage", "bias_l2", "hdi_length", "coverage", 
                "time_initial", "time_refined_plus", "time_mcmc"]
        # Add other cols if exist
        other_cols = [c for c in df_model.columns if c not in cols and not c.startswith("seconds_")]
        final_cols = cols + other_cols
        # Filter existing columns only
        final_cols = [c for c in final_cols if c in df_model.columns]
        
        df_model = df_model[final_cols]
        df_model.to_csv(f"results/tables/{model_name}_results.csv", index=False)
        print(f"Saved table for {model_name} to results/tables/{model_name}_results.csv")
            
        # Compute Summary Stats
        bias_l2 = [r["bias_l2"] for r in rows]
        hdi_length = [np.mean(r["hdi_length"]) for r in rows]
        coverage = [np.mean(r["coverage"]) for r in rows]
        
        # Average Times
        avg_sec_initial = np.mean([r["seconds_initial"] for r in rows])
        avg_sec_sl = np.mean([r["seconds_refined_plus"] for r in rows])
        avg_sec_mcmc = np.mean([r["seconds_mcmc"] for r in rows])
        
        mean_bias = np.mean(bias_l2)
        median_bias = np.median(bias_l2)
        avg_hdi_len = np.mean(hdi_length)
        avg_coverage = np.mean(coverage)
        
        print(f"\nModel: {model_name.upper()}")
        print(f"  Bias L2: Mean={mean_bias:.4f}")
        print(f"  Avg Time (Initial): {format_time(avg_sec_initial)}")
        if avg_sec_sl > 0: print(f"  Avg Time (Refined+): {format_time(avg_sec_sl)}")
        if avg_sec_mcmc > 0: print(f"  Avg Time (MCMC): {format_time(avg_sec_mcmc)}")
        
        summary_data.append({
            "Model": model_name,
            "Bias_Mean": mean_bias,
            "Bias_Median": median_bias,
            "HDI_Len_Mean": avg_hdi_len,
            "Coverage_Mean": avg_coverage,
            "Avg_Time_Initial": format_time(avg_sec_initial),
            "Avg_Time_RefinedPlus": format_time(avg_sec_sl),
            "Avg_Time_MCMC": format_time(avg_sec_mcmc)
        })
        
    # Save Summary
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv("results/final_summary.csv", index=False)
    print("\nSummary saved to results/final_summary.csv")

if __name__ == "__main__":
    main()
