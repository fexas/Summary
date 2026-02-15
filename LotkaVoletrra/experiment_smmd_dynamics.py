
import os
import sys
import time
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Fix for MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import local modules
from data_generation import LVTask
from models.smmd import SMMD_Model, sliced_mmd_loss

# Config & Constants
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {}

# Override or set defaults for this specific experiment
DATASET_SIZE = 12800 
BATCH_SIZE = CONFIG.get("batch_size", 128)
N_TIME_STEPS = CONFIG.get("n_time_steps", 151)
DT = CONFIG.get("dt", 0.2)
LEARNING_RATE = CONFIG.get("learning_rate", 3e-4)

# SMMD Config
SMMD_MMD_CONFIG = CONFIG.get("smmd_mmd_config", {"M": 50, "L": 20})
M = SMMD_MMD_CONFIG.get("M", 50)
L = SMMD_MMD_CONFIG.get("L", 20)

d = 4 # alpha, beta, gamma, delta
d_x = 2 # Prey, Predator

# Device
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
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

def train_smmd_dynamics(model, train_loader, epochs, device, x_obs, theta_true, output_dir):
    """
    Train SMMD model and sample posterior every 50 epochs.
    """
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_scheduler(optimizer, epochs)
    
    model.to(device)
    model.train()
    
    loss_history = []
    dynamics_samples = {} # epoch -> samples
    
    print(f"Starting SMMD training for {epochs} epochs...")
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Training Step
        for batch in train_loader:
            x_batch = batch[0].to(device)
            theta_batch = batch[1].to(device)
            
            optimizer.zero_grad()
            
            with torch.enable_grad():
                # Sample Z
                z = torch.randn(x_batch.size(0), M, model.d, device=device)
                
                # Forward
                theta_fake = model(x_batch, z)
                
                loss = sliced_mmd_loss(theta_batch, theta_fake, num_slices=L, n_time_steps=N_TIME_STEPS)
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
        # Periodic Sampling (Every 50 epochs)
        if (epoch + 1) % 50 == 0:
            print(f"--- Sampling at Epoch {epoch+1} ---")
            
            # Use model's sample_posterior method
            samples = model.sample_posterior(x_obs, n_samples=1000)
            dynamics_samples[epoch+1] = samples
            
            # Save intermediate
            np.save(os.path.join(output_dir, f"samples_epoch_{epoch+1}.npy"), samples)
            
            # Switch back to train mode (sample_posterior might use eval but usually it's fine, verify if it changes mode)
            model.train()

    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f}s")
    
    return loss_history, dynamics_samples

def plot_dynamics(dynamics_samples, theta_true, output_dir):
    """
    Plot evolution of posterior approximation.
    """
    print("Plotting dynamics...")
    param_names = ['alpha', 'beta', 'gamma', 'delta']
    num_params = d
    
    # Setup colors/palette
    epochs = sorted(dynamics_samples.keys())
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(epochs))]
    
    fig, axes = plt.subplots(1, num_params, figsize=(5*num_params, 5))
    if num_params == 1: axes = [axes]
    
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        
        # Plot each epoch
        for idx, epoch in enumerate(epochs):
            samples = dynamics_samples[epoch]
            sns.kdeplot(samples[:, i], ax=ax, color=colors[idx], label=f'Epoch {epoch}', alpha=0.3)
            
        # Plot True
        ax.axvline(theta_true[i], color='red', linestyle='--', linewidth=2, label='True')
        
        ax.set_title(f'Evolution of {param_name}')
        if i == 0: # Only legend on first plot to avoid clutter
            ax.legend()
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, "posterior_dynamics.png")
    plt.savefig(save_path)
    print(f"Dynamics plot saved to {save_path}")

def main():
    # Setup Output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/smmd_dynamics_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Results will be saved to {output_dir}")
    
    # 1. Data Generation
    print("Generating data...")
    task = LVTask(N_TIME_STEPS, DT)
    
    # Generate Ground Truth
    theta_true = task.sample_prior(1, "vague").flatten()
    x_obs = task.simulator(theta_true[None, :])[0]
    
    # Save Ground Truth
    np.save(os.path.join(output_dir, "theta_true.npy"), theta_true)
    np.save(os.path.join(output_dir, "x_obs.npy"), x_obs)
    
    # Generate Training Data
    print(f"Generating training dataset (Size: {DATASET_SIZE})...")
    theta_train = task.sample_prior(DATASET_SIZE, "vague")
    x_train = task.simulator(theta_train)
    
    # Create Loader
    dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(theta_train).float()
    )
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model Setup
    smmd_config = CONFIG.get("models_config", {}).get("smmd", {})
    epochs = smmd_config.get("epochs", 500)
    summary_dim = smmd_config.get("summary_dim", 10)
    
    print(f"Initializing SMMD Model (Epochs: {epochs}, Summary Dim: {summary_dim})...")
    model = SMMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=N_TIME_STEPS)
    
    # 3. Train & Sample
    loss_history, dynamics_samples = train_smmd_dynamics(
        model, train_loader, epochs, DEVICE, x_obs, theta_true, output_dir
    )
    
    # 4. Plot
    plot_dynamics(dynamics_samples, theta_true, output_dir)
    
    # Save Loss
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SMMD Training Loss")
    plt.savefig(os.path.join(output_dir, "loss_history.png"))
    print("Loss plot saved.")

if __name__ == "__main__":
    main()
