
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

# Import from local modules
from data_generation import (
    GaussianTask,
    generate_dataset,
    TRUE_PARAMS,
    n, d, d_x, p
)
from models.smmd import SMMD_Model, sliced_mmd_loss
from models.mmd import MMD_Model, mmd_loss
from utilities import refine_posterior, compute_bandwidth_torch
from pymc_sampler import run_pymc

# Try importing BayesFlow (optional)
try:
    import bayesflow as bf
    from models.bayesflow_wrapper import BayesFlowWrapper
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
DATASET_SIZE = 12800

# Check for MPS (Apple Silicon)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon) acceleration.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU.")

def get_scheduler(optimizer, epochs):
    """Cosine Decay Scheduler"""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

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
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
    print(f"Training finished in {time.time() - start_time:.2f}s")
    return loss_history

def train_bayesflow(train_loader, epochs, device):
    """Training loop for BayesFlow."""
    if not BAYESFLOW_AVAILABLE:
        raise ImportError("BayesFlow is not available.")
        
    # Define BayesFlow components
    # 1. Summary Network (same architecture as SMMD if possible, or standard)
    summary_net = bf.networks.DeepSet(summary_dim=10) # Using built-in DeepSet for simplicity or define custom
    
    # 2. Inference Network (Invertible Network)
    inference_net = bf.networks.InferenceNetwork(num_params=d)
    
    # 3. Amortized Posterior
    amortized_posterior = bf.amortizers.AmortizedPosterior(inference_net, summary_net)
    
    # 4. Trainer
    # We need to adapt the data loader or use BayesFlow's trainer
    # BayesFlow trainer expects a generative model or a data loader with dicts
    
    # Let's manually train for flexibility with existing dataloader
    optimizer = optim.Adam(amortized_posterior.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(optimizer, epochs)
    
    amortized_posterior.to(device)
    loss_history = []
    
    print("Starting training (BayesFlow)...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, theta_batch in train_loader:
            x_batch = x_batch.to(device)
            theta_batch = theta_batch.to(device)
            
            # BayesFlow expects input dict: {"parameters": ..., "summary_conditions": ...}
            # x_batch needs to be (batch, n_obs, d_x)
            # Our x_batch is (batch, n_obs, d_x)
            
            batch_dict = {
                "parameters": theta_batch,
                "summary_conditions": x_batch
            }
            
            optimizer.zero_grad()
            loss = amortized_posterior.compute_loss(batch_dict)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # Wrap in our wrapper
    wrapper = BayesFlowWrapper(amortized_posterior, d=d)
    return wrapper, loss_history

def plot_loss(loss_history, title="Training Loss"):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"Gaussian/results/loss_{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_posterior(theta_samples, theta_true, method_name):
    # Select first 2 dimensions for plotting
    df = pd.DataFrame(theta_samples[:, :2], columns=['theta1', 'theta2'])
    
    plt.figure(figsize=(8, 8))
    sns.kdeplot(data=df, x='theta1', y='theta2', fill=True, alpha=0.5, label='Posterior')
    plt.scatter(theta_true[0], theta_true[1], c='red', marker='*', s=100, label='True Parameter')
    plt.title(f"Posterior Approximation ({method_name})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Gaussian/results/posterior_{method_name.lower().replace(' ', '_')}.png")
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
    os.makedirs("Gaussian/results", exist_ok=True)
    
    # Configuration
    # Models to run and their epochs
    MODELS_CONFIG = {
        "smmd": 30,
        "mmd": 30,
        "bayesflow": 30
    }
    # To run specific models, just comment out others in the dict or filter keys
    # MODELS_TO_RUN = ["smmd", "mmd", "bayesflow"] 
    MODELS_TO_RUN = list(MODELS_CONFIG.keys())

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
        
        pymc_samples = run_pymc(x_obs_2d, n_draws=2000, n_tune=3000, chains=200)
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
