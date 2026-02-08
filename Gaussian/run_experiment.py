
"""
Run Experiment for Gaussian Task with SMMD/MMD/BayesFlow and Refinement.
"""

import os
import time
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
    n, d, d_x, p
)
from models.smmd import SMMD_Model, sliced_mmd_loss
from models.mmd import MMD_Model, mmd_loss
from models.bayesflow_net import build_bayesflow_model
from utilities import refine_posterior, compute_bandwidth_torch
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
MODEL_TYPE = "smmd"  # Options: "smmd", "mmd", "bayesflow"
M = 50      # MMD approximation samples
L = 20      # Slicing directions (for SMMD)
BATCH_SIZE = 256
EPOCHS = 500
LEARNING_RATE = 0.001
DATASET_SIZE = 25600

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
            
    print(f"Training finished in {time.time() - start_time:.2f}s")
    return loss_history

def train_bayesflow(train_loader, epochs, device):
    """
    Train BayesFlow model using Keras training loop.
    """
    if not BAYESFLOW_AVAILABLE:
        raise ImportError("BayesFlow is not available.")
        
    print("Starting training (BayesFlow with Keras Loop)...")
    
    # Build model (ensure it's created on the correct device)
    # Keras 3 with Torch backend creates weights on the default torch device or current device context
    if device.type == "cuda":
        # Use context manager to ensure parameters are initialized on GPU
        print(f"Initializing BayesFlow model on {device}...")
        with torch.device(device):
             amortized_posterior = build_bayesflow_model(d, d_x, summary_dim=10)
    else:
        # For MPS/CPU, standard build is usually fine, but explicit doesn't hurt
        amortized_posterior = build_bayesflow_model(d, d_x, summary_dim=10)
    
    # Verify device placement (optional check)
    try:
        # Access a weight to check device
        # Note: Keras 3 weights might need .value to get the tensor
        if hasattr(amortized_posterior, "summary_network") and hasattr(amortized_posterior.summary_network, "weights") and len(amortized_posterior.summary_network.weights) > 0:
            first_weight = amortized_posterior.summary_network.weights[0]
            if hasattr(first_weight, "value"):
                w_device = first_weight.value.device
            else:
                w_device = first_weight.device
            print(f"Model weights initialized on: {w_device}")
    except Exception as e:
        # This is just a check, don't fail if it doesn't work (e.g. weights not yet created)
        pass
    
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
        
    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    return amortized_posterior, loss_history

def plot_loss(loss_history, title="Training Loss"):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"results/loss_{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_posterior(theta_samples, theta_true, method_name):
    """
    Plots the marginal posterior for each of the 5 dimensions in a single row.
    """
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
    plt.savefig(f"results/posterior_{method_name.lower().replace(' ', '_')}.png")
    plt.close()

def run_single_experiment(model_type, task, train_loader, theta_true, x_obs, epochs=30, device=DEVICE):
    """
    Runs a single experiment for a specific model type.
    """
    print(f"\n=== Running Experiment for {model_type.upper()} (Epochs={epochs}) ===")
    
    # 1. Model Initialization & Training
    model = None
    loss_history = []
    
    if model_type == "smmd":
        model = SMMD_Model(summary_dim=10, d=d, d_x=d_x, n=n)
        loss_history = train_smmd_mmd(model, train_loader, epochs, device, model_type="smmd")
    elif model_type == "mmd":
        model = MMD_Model(summary_dim=10, d=d, d_x=d_x, n=n)
        loss_history = train_smmd_mmd(model, train_loader, epochs, device, model_type="mmd")
    elif model_type == "bayesflow":
        if BAYESFLOW_AVAILABLE:
            model, loss_history = train_bayesflow(train_loader, epochs, device)
        else:
            print("BayesFlow unavailable, skipping.")
            return None
            
    plot_loss(loss_history, title=f"{model_type.upper()} Loss")
    
    # 2. Model Inference (Amortized)
    print(f"--- Amortized Inference ({model_type}) ---")
    
    # Sample from model
    n_samples = 1000
    if hasattr(model, "sample_posterior"):
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
    
    plot_posterior(posterior_samples, theta_true, f"{model_type}_Amortized")
    
    # 3. Local Refinement
    print(f"--- Local Refinement ({model_type}) ---")
    
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
    
    # Reshape refined samples: (n_chains * n_samples, d)
    refined_samples_flat = refined_samples.reshape(-1, d)
    
    print(f"Refined Mean: {np.mean(refined_samples_flat, axis=0)}")
    
    plot_posterior(refined_samples_flat, theta_true, f"{model_type}_Refined")
    
    return {
        "model": model,
        "loss_history": loss_history,
        "amortized_samples": posterior_samples,
        "refined_samples": refined_samples_flat
    }

def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Configuration
    # Models to run and their epochs
    MODELS_CONFIG = {
        "smmd": 500,
        "mmd": 500,
        "bayesflow": 20
    }
    # MODELS_TO_RUN = list(MODELS_CONFIG.keys())
    MODELS_TO_RUN = ["bayesflow"]

    # 1. Data Generation (Shared across models for fair comparison in one round)
    print("=== Step 1: Data Generation ===")
    task = GaussianTask()
    
    # Generate Training Data
    print(f"Generating training data (size={DATASET_SIZE})...")
    theta_train, x_train = generate_dataset(task, n_sims=DATASET_SIZE)
    
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
    
    # 2. PyMC Reference Sampling
    print("\n=== Step 2: PyMC Reference Sampling ===")
    
    # Invert 3D observation to 2D for PyMC (Inverse Stereographic Projection)
    # x = X / (1 - Z), y = Y / (1 - Z)
    # x_obs is (n, 3)
    try:
        X_comp = x_obs[:, 0]
        Y_comp = x_obs[:, 1]
        Z_comp = x_obs[:, 2]
        
        # Handle numerical stability if Z is close to 1 (shouldn't be for Gaussian on plane)
        denom = 1.0 - Z_comp
        denom[np.abs(denom) < 1e-6] = 1e-6 # Avoid division by zero
        
        x_2d = X_comp / denom
        y_2d = Y_comp / denom
        x_obs_2d = np.stack([x_2d, y_2d], axis=-1)
        
        print(f"Inverted 3D obs to 2D for PyMC. Shape: {x_obs_2d.shape}")
        
        pymc_samples = run_pymc(x_obs_2d, n_draws=2000, n_tune=3000, chains=20)
        print(f"PyMC Samples Shape: {pymc_samples.shape}")
        plot_posterior(pymc_samples, theta_true, "PyMC_Reference")
    except Exception as e:
        print(f"PyMC Sampling failed: {e}")
        pymc_samples = None

    # 3. Run Experiments
    print("\n=== Step 3: Run Model Experiments ===")
    
    if isinstance(MODELS_TO_RUN, str):
        MODELS_TO_RUN = [MODELS_TO_RUN]
        
    for model_type in MODELS_TO_RUN:
        if model_type not in MODELS_CONFIG:
            print(f"Warning: No epoch config for {model_type}, skipping.")
            continue
            
        epochs = MODELS_CONFIG[model_type]
        run_single_experiment(model_type, task, train_loader, theta_true, x_obs, epochs=epochs, device=DEVICE)
    
    print("\nAll Experiments Completed.")

if __name__ == "__main__":
    main()
