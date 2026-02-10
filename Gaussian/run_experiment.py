
"""
Run Experiment for Gaussian Task with SMMD/MMD/BayesFlow and Refinement.
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
from utilities import refine_posterior, compute_bandwidth_torch, compute_mmd_metric
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
# Default values, will be overridden by config.json if available
M = 50      # MMD approximation samples
L = 20      # Slicing directions (for SMMD)
BATCH_SIZE = 256
DATASET_SIZE = 25600
LEARNING_RATE = 1e-3
NUM_ROUNDS = 5 # Default number of rounds
n = 50 # Default observation size, will be updated from config

# Load Configuration from JSON
CONFIG_PATH = "config.json"
if os.path.exists(CONFIG_PATH):
    print(f"Loading configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        n = config.get("n_observation", n)
        NUM_ROUNDS = config.get("num_rounds", NUM_ROUNDS)
        MODELS_CONFIG = config.get("models_config", {
            "smmd": 500,
            "mmd": 500,
            "bayesflow": 20,
            "dnnabc": 500,
            "w2abc": 0
        })
        MODELS_TO_RUN = config.get("models_to_run", list(MODELS_CONFIG.keys()))
        print(f"Configuration loaded: n={n}, rounds={NUM_ROUNDS}, models={MODELS_TO_RUN}")
else:
    print(f"Warning: {CONFIG_PATH} not found. Using default parameters.")
    MODELS_CONFIG = {
        "smmd": 500,
        "mmd": 500,
        "bayesflow": 20,
        "dnnabc": 500,
        "w2abc": 0
    }
    MODELS_TO_RUN = list(MODELS_CONFIG.keys())

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

def train_smmd_mmd(model, train_loader, epochs, device, model_type="smmd"):
    """Training loop for SMMD and MMD models."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
                # Generate fake parameters
                # Sample Z: (batch, M, d)
                z = torch.randn(x_batch.size(0), M, model.d, device=device)
                
                # Forward pass
                # Note: SMMD_Model forward takes (x, z) and returns theta_fake
                theta_fake = model(x_batch, z)
                
                # Compute Loss
                if model_type == "smmd":
                    loss = sliced_mmd_loss(theta_batch, theta_fake, num_slices=L, n_points=M)
                elif model_type == "mmd":
                    loss = mmd_loss(theta_batch, theta_fake, n_points=M)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
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

def train_bayesflow(train_loader, epochs, device):
    """
    Train BayesFlow model using Keras training loop.
    """
    if not BAYESFLOW_AVAILABLE:
        raise ImportError("BayesFlow is not available.")
        
    print("Starting training (BayesFlow with Keras Loop)...")
    
    # Build model
    amortized_posterior = build_bayesflow_model(d, d_x, summary_dim=10)
    
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
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
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
    output_dir = f"results/plots/round_{round_id}/loss"
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

    num_params = next(iter(all_model_samples.values())).shape[1]
    cols = num_params
    
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 1: axes = [axes]
    
    for i in range(cols):
        ax = axes[i]
        param_name = f'theta{i+1}'
        
        for model_name, samples in all_model_samples.items():
            if samples is not None:
                sns.kdeplot(x=samples[:, i], label=model_name.upper(), ax=ax, alpha=0.3, fill=False, linewidth=2)
            
        if i < len(theta_true):
            ax.axvline(x=theta_true[i], color='black', linestyle='--', linewidth=2, label='True')
            
        ax.set_title(f'Marginal {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_refinement_comparison(samples_dict, theta_true, output_dir, model_name):
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

    num_params = valid_samples[0].shape[1]
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
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def run_single_experiment(model_type, task, train_loader, theta_true, x_obs, round_id, epochs=30, device=DEVICE, n_obs=n):
    """
    Runs a single experiment for a specific model type.
    """
    print(f"\n=== Running Experiment for {model_type.upper()} (Round {round_id}, Epochs={epochs}, n={n_obs}) ===")
    
    # 1. Model Initialization & Training
    model = None
    loss_history = []
    training_time = 0.0
    refinement_time = 0.0
    total_time = 0.0 # For DNNABC/W2ABC
    
    t_start_total = time.time()
    
    if model_type == "smmd":
        model = SMMD_Model(summary_dim=10, d=d, d_x=d_x, n=n_obs)
        loss_history, training_time = train_smmd_mmd(model, train_loader, epochs, device, model_type="smmd")
    elif model_type == "mmd":
        model = MMD_Model(summary_dim=10, d=d, d_x=d_x, n=n_obs)
        loss_history, training_time = train_smmd_mmd(model, train_loader, epochs, device, model_type="mmd")
    elif model_type == "bayesflow":
        if BAYESFLOW_AVAILABLE:
            # Force CPU for BayesFlow to avoid MPS/Standardization issues
            bf_device = torch.device("cpu")
            print("Forcing BayesFlow to use CPU to avoid MPS/Keras compatibility issues.")
            model, loss_history, training_time = train_bayesflow(train_loader, epochs, bf_device)
        else:
            print("BayesFlow unavailable, skipping.")
            return None
    elif model_type == "dnnabc":
        model = DNNABC_Model(d=d, d_x=d_x, n_points=n_obs)
        # We treat this as training time, but for final logging we only care about "total time"
        loss_history = train_dnnabc(model, train_loader, epochs, device)
    elif model_type == "w2abc":
        print("W2-ABC does not require neural network training. Proceeding to inference...")
        loss_history = []
    elif model_type == "snpe_a":
        # SNPE-A runs training and sampling together
        # Get sbi_rounds from global config if available, else default to 2
        sbi_rounds = config.get("sbi_rounds", 2) if "config" in globals() else 2
        
        posterior_samples = run_sbi_model(
            model_type="snpe_a",
            train_loader=None, # SBI generates its own simulations
            x_obs=x_obs,
            theta_true=theta_true,
            task=task,
            device=str(device), # sbi expects string or torch.device
            num_rounds=sbi_rounds,
            sims_per_round=1000,
            max_epochs=epochs
        )
        loss_history = [] # SBI wrapper doesn't return loss history yet
            
    if loss_history:
        plot_loss(loss_history, title=f"{model_type.upper()} Loss", round_id=round_id)
    
    # 2. Model Inference (Amortized)
    print(f"--- Amortized Inference ({model_type}) ---")
    
    # Sample from model
    n_samples = 1000
    
    if model_type == "w2abc":
        # W2ABC does inference via SMC, which takes time
        posterior_samples = run_w2abc(task, x_obs, n_samples=n_samples, max_populations=2) 
        model = "W2ABC_SMC_Sampler" # Placeholder
    elif model_type == "snpe_a":
        # Already sampled in training step
        pass
    elif model_type == "dnnabc":
        # DNNABC does ABC rejection sampling using trained summary net
        posterior_samples = abc_rejection_sampling(model, x_obs, task, n_samples=n_samples, device=device)
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
    save_dir_samples = f"results/posterior_samples/{model_type}/round_{round_id}"
    os.makedirs(save_dir_samples, exist_ok=True)
    np.save(f"{save_dir_samples}/amortized.npy", posterior_samples)
    print(f"Saved amortized samples to {save_dir_samples}/amortized.npy")
    
    # Plot Amortized Posterior
    plot_dir = f"results/plots/round_{round_id}/{model_type}"
    plot_posterior(posterior_samples, theta_true, plot_dir, "amortized_posterior.png")
    
    # 3. Local Refinement
    refined_samples_flat = None
    
    if model_type in ["dnnabc", "w2abc", "snpe_a"]:
        print(f"--- Skipping Local Refinement for {model_type} (as requested) ---")
        refined_samples_flat = posterior_samples # Use amortized/SMC samples as final
        
        # For these models, we only care about total time
        total_time = time.time() - t_start_total
    else:
        print(f"--- Local Refinement ({model_type}) ---")
        
        t_refine_start = time.time()
        
        # Refinement parameters
        n_chains = 1000
        burn_in = 99
        
        refined_samples = refine_posterior(
            model, x_obs, task,
            n_chains=n_chains,
            n_samples=1,       # Take the last sample
            burn_in=burn_in,
            thin=1,
            nsims=50,
            epsilon=None,      # Auto-compute bandwidth
            device=str(device) 
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
        plot_refinement_comparison(samples_comp, theta_true, plot_dir, model_type)
        
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
    save_dir = f"saved_models/{model_type}"
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
    # Create results directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # Configuration is already loaded at module level
    # MODELS_CONFIG, MODELS_TO_RUN, NUM_ROUNDS, n are set
    
    # Store all results for final CSV
    all_results = []
    
    print(f"Starting Multi-Round Experiment (Total Rounds: {NUM_ROUNDS})")

    for round_idx in range(1, NUM_ROUNDS + 1):
        print(f"\n{'='*20} ROUND {round_idx}/{NUM_ROUNDS} {'='*20}")
        
        # 1. Data Generation (Shared across models for fair comparison in one round)
        print("=== Step 1: Data Generation ===")
        # Use n from config
        task = GaussianTask(n=n)
        
        # Generate Training Data
        print(f"Generating training data (size={DATASET_SIZE}, n={n})...")
        theta_train, x_train = generate_dataset(task, n_sims=DATASET_SIZE, n_obs=n)
        
        # Create DataLoader
        dataset = TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(theta_train).float()
        )
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Generate Observation (Ground Truth)
        print("Generating observation...")
        theta_true, x_obs = task.get_ground_truth()
        print(f"True Params: {theta_true}")
        
        # Save True Observation
        obs_save_dir = f"results/true_observations/round_{round_idx}"
        os.makedirs(obs_save_dir, exist_ok=True)
        np.save(f"{obs_save_dir}/observation.npy", x_obs)
        np.save(f"{obs_save_dir}/theta_true.npy", theta_true)
        print(f"Saved true observation to {obs_save_dir}")
        
        # 2. PyMC Reference Sampling
        print("\n=== Step 2: PyMC Reference Sampling ===")
        
        pymc_samples = None
        try:
            # Invert 3D observation to 2D for PyMC (Inverse Stereographic Projection)
            X_comp = x_obs[:, 0]
            Y_comp = x_obs[:, 1]
            Z_comp = x_obs[:, 2]
            
            denom = 1.0 - Z_comp
            denom[np.abs(denom) < 1e-6] = 1e-6 # Avoid division by zero
            
            x_2d = X_comp / denom
            y_2d = Y_comp / denom
            x_obs_2d = np.stack([x_2d, y_2d], axis=-1)
            
            print(f"Inverted 3D obs to 2D for PyMC. Shape: {x_obs_2d.shape}")
            
            # Reduce draws/tune slightly for speed in multi-round (optional, keeping high for quality)
            pymc_samples = run_pymc(x_obs_2d, n_draws=2000, n_tune=3000, chains=30)
            print(f"PyMC Samples Shape: {pymc_samples.shape}")
            pymc_plot_dir = f"results/plots/round_{round_idx}/pymc"
            plot_posterior(pymc_samples, theta_true, pymc_plot_dir, "posterior.png")
            
            # Save PyMC Samples
            pymc_save_dir = f"results/posterior_samples/pymc_reference/round_{round_idx}"
            os.makedirs(pymc_save_dir, exist_ok=True)
            np.save(f"{pymc_save_dir}/samples.npy", pymc_samples)
            print(f"Saved PyMC samples to {pymc_save_dir}/samples.npy")
            
        except Exception as e:
            print(f"PyMC Sampling failed: {e}")
            pymc_samples = None
    
        # 3. Run Experiments
        print("\n=== Step 3: Run Model Experiments ===")
        
        # Collect samples for combined plot
        round_samples_for_plot = {}
        if pymc_samples is not None:
            round_samples_for_plot["PyMC"] = pymc_samples
        
        current_models_to_run = MODELS_TO_RUN
        if isinstance(current_models_to_run, str):
            current_models_to_run = [current_models_to_run]
            
        for model_type in current_models_to_run:
            if model_type not in MODELS_CONFIG:
                print(f"Warning: No epoch config for {model_type}, skipping.")
                continue
                
            epochs = MODELS_CONFIG[model_type]
            
            # Run Experiment
            # Pass n from config
            result = run_single_experiment(model_type, task, train_loader, theta_true, x_obs, round_id=round_idx, epochs=epochs, device=DEVICE, n_obs=n)
            
            if result is None:
                continue
            
            # Collect amortized samples for combined plot
            if result["amortized_samples"] is not None:
                round_samples_for_plot[model_type] = result["amortized_samples"]
                
            # Compute MMD Metrics
            mmd_amortized = 0.0
            mmd_refined = 0.0
            
            if pymc_samples is not None:
                # Amortized MMD
                if result["amortized_samples"] is not None:
                    mmd_amortized = compute_mmd_metric(result["amortized_samples"], pymc_samples)
                    print(f"MMD ({model_type} Amortized): {mmd_amortized:.4f}")
                    
                # Refined MMD
                if result["refined_samples"] is not None:
                    mmd_refined = compute_mmd_metric(result["refined_samples"], pymc_samples)
                    print(f"MMD ({model_type} Refined): {mmd_refined:.4f}")
            else:
                print("Skipping MMD calculation (No PyMC samples).")

            # Save Model
            save_model(result["model"], model_type, round_idx)
            
            # Record Results
            record = {
                "round": round_idx,
                "model_name": model_type,
                "training_time": result["training_time"],
                "refinement_time": result["refinement_time"],
                "total_time": result["total_time"],
                "mmd_amortized": mmd_amortized,
                "mmd_refined": mmd_refined
            }
            all_results.append(record)
            
            # Save CSV incrementally (optional, but good for safety)
            pd.DataFrame(all_results).to_csv("results/experiment_results.csv", index=False)
            
        # End of Model Loop - Plot Combined Posteriors
        print(f"Generating combined posterior plot for Round {round_idx}...")
        combined_plot_dir = f"results/plots/round_{round_idx}"
        plot_combined_posteriors(round_samples_for_plot, theta_true, combined_plot_dir, "combined_posteriors.png")
    
    # 4. Summary Statistics
    print("\n=== Experiment Summary ===")
    if not all_results:
        print("Warning: No results to summarize.")
    else:
        df = pd.DataFrame(all_results)
        
        # Calculate Mean and Median
        if "model_name" in df.columns:
            summary_mean = df.groupby("model_name").mean(numeric_only=True)
            summary_median = df.groupby("model_name").median(numeric_only=True)
            
            # Drop 'round' column if present in summary (it's averaged, so maybe not useful)
            if "round" in summary_mean.columns:
                summary_mean = summary_mean.drop(columns=["round"])
            if "round" in summary_median.columns:
                summary_median = summary_median.drop(columns=["round"])
            
            print("Mean Metrics:")
            print(summary_mean)
            print("\nMedian Metrics:")
            print(summary_median)
            
            # Save Summary
            os.makedirs("results", exist_ok=True)
            summary_mean.to_csv("results/summary_mean.csv")
            summary_median.to_csv("results/summary_median.csv")
        else:
            print("Error: 'model_name' column missing in results DataFrame.")
            print(df.head())
    
    print("\nAll Experiments Completed. Results saved to 'results/' directory.")

if __name__ == "__main__":
    main()
