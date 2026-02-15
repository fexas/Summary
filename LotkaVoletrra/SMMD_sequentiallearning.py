
"""
SMMD Sequential Learning Experiment Script.
Tests SMMD with sequential data addition (doubling strategy) and validation-based early stopping.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.neighbors import KernelDensity

# Import from local modules
from data_generation import LVTask
from models.smmd import SMMD_Model, sliced_mmd_loss
from utilities import compute_metrics

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Experiment Parameters
INITIAL_SAMPLES = 1000
ROUNDS = 5  # Adjust as needed, usually 10 for SNPE comparison, but let's start with a few
BATCH_SIZE = 50
LEARNING_RATE = 3e-4
EPOCHS = 1000
VAL_FRACTION = 0.1
PATIENCE = 50
SMMD_M = 50
SMMD_L = 20
SUMMARY_DIM = 10

# LV Task Params
T_MAX = 30.0
DT = 0.2
N_TIME_STEPS = 151
D_THETA = 4
D_X = 2

def get_scheduler(optimizer, epochs):
    """Cosine Decay Scheduler"""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

def fit_kde_and_evaluate(theta_train, theta_eval):
    """
    Fit KDE on theta_train and evaluate log_density on theta_eval.
    Returns density (exp(log_density)).
    """
    # Use GridSearch or simple heuristic for bandwidth?
    # Simple heuristic: Scott's Rule or Silvermans
    # sklearn defaults to 1.0, which might be too wide.
    # Let's use a simple heuristic or a fixed value if dimensions are small.
    # For d=4, maybe 0.2 or 0.5?
    # Let's try to optimize or use a reasonable default.
    # We can use Cross Validation if needed, but for speed let's use a rule of thumb.
    # Scott's factor: n**(-1./(d+4))
    n, d = theta_train.shape
    bandwidth = n**(-1./(d+4))
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(theta_train)
    
    log_density = kde.score_samples(theta_eval)
    return np.exp(log_density)

def train_smmd_with_val(model, dataset, epochs, device, batch_size=BATCH_SIZE, lr=LEARNING_RATE, val_fraction=VAL_FRACTION, patience=PATIENCE):
    """
    Training loop with validation and early stopping.
    """
    # Split dataset
    total_len = len(dataset)
    val_len = int(total_len * val_fraction)
    train_len = total_len - val_len
    
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(optimizer, epochs)
    
    model.to(device)
    
    loss_history = []
    val_loss_history = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Starting training: Train={train_len}, Val={val_len}, Epochs={epochs}, Patience={patience}")
    start_time = time.time()
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            x_batch = batch[0].to(device)
            theta_batch = batch[1].to(device)
            
            optimizer.zero_grad()
            
            with torch.enable_grad():
                # Generate fake parameters
                z = torch.randn(x_batch.size(0), SMMD_M, model.d, device=device)
                theta_fake = model(x_batch, z)
                
                loss = sliced_mmd_loss(theta_batch, theta_fake, num_slices=SMMD_L, n_time_steps=N_TIME_STEPS)
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        
        # --- Validation ---
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch[0].to(device)
                theta_batch = batch[1].to(device)
                
                z = torch.randn(x_batch.size(0), SMMD_M, model.d, device=device)
                theta_fake = model(x_batch, z)
                
                loss = sliced_mmd_loss(theta_batch, theta_fake, num_slices=SMMD_L, n_time_steps=N_TIME_STEPS)
                val_epoch_loss += loss.item()
        
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        scheduler.step()
        
        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            # print(f"Epoch {epoch+1}: New best val loss {best_val_loss:.6f}")
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f}")
            break
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f}s")
    
    return loss_history, val_loss_history, training_time

def main():
    # Setup
    task = LVTask(t_max=T_MAX, dt=DT)
    theta_true, x_obs = task.get_ground_truth()
    
    # Create results directory
    res_dir = "results/smmd_sequential"
    os.makedirs(res_dir, exist_ok=True)
    
    # Store accumulated data
    # We will keep them as numpy arrays and merge them
    theta_pool = None
    x_pool = None
    
    # Ground Truth Plotting
    # ...
    
    print(f"Ground Truth Theta: {theta_true}")
    
    # Initialize Model (Re-initialize each round or fine-tune? Usually re-init or continue? 
    # SNPE re-trains from scratch or fine-tunes. 
    
    model = SMMD_Model(summary_dim=SUMMARY_DIM, d=D_THETA, d_x=D_X, n=N_TIME_STEPS)
    
    # Store samples from each round for final plot
    all_rounds_samples = {}
    
    for round_idx in range(1, ROUNDS + 1):
        print(f"\n{'='*20} ROUND {round_idx} {'='*20}")
        
        # 1. Generate/Prepare Data
        if round_idx == 1:
            # Initial Round: Sample from Prior
            print(f"Generating {INITIAL_SAMPLES} samples from Prior...")
            theta_new = task.sample_prior(INITIAL_SAMPLES, "vague")
            x_new = task.simulator(theta_new)
            
            theta_pool = theta_new
            x_pool = x_new
            
        else:
            # Subsequent Rounds
            # N_new = Current Pool Size (Doubling)
            n_new = len(theta_pool)
            print(f"Sampling {n_new} samples from Posterior (Doubling)...")
            
            # Sample from current model
            # We need to sample from posterior given x_obs
            # Wait, sequential learning in SNPE usually generates data based on parameters sampled from the current posterior 
            # conditioned on the OBSERVATION x_obs.
            # Yes. "based on approximate posterior".
            # So we sample theta ~ q(theta | x_obs).
            
            # Handle x_obs dimension
            x_obs_input = x_obs
            
            # Sample
            theta_new = model.sample_posterior(x_obs_input, n_new)
            
            # Simulate
            x_new = task.simulator(theta_new)
            
            # Aggregate
            theta_pool = np.concatenate([theta_pool, theta_new], axis=0)
            x_pool = np.concatenate([x_pool, x_new], axis=0)
            
            print(f"New Pool Size: {len(theta_pool)}")
            
            # Calculate Weights
            # w = p(theta) / q_KDE(theta)
            # p(theta) is constant (Uniform).
            # q_KDE(theta) is density of theta_pool.
            
            print("Calculating weights via KDE...")
            kde_density = fit_kde_and_evaluate(theta_pool, theta_pool) # Evaluate on itself
            
            # Avoid division by zero
            kde_density = np.clip(kde_density, 1e-12, None)
            
            pass

        dataset = TensorDataset(
            torch.from_numpy(x_pool).float(),
            torch.from_numpy(theta_pool).float()
        )
        
        # Train
        loss_hist, val_hist, time_taken = train_smmd_with_val(
            model, dataset, EPOCHS, DEVICE, 
            batch_size=BATCH_SIZE, lr=LEARNING_RATE, 
            val_fraction=VAL_FRACTION, patience=PATIENCE
        )
        
        # 3. Sample & Evaluate
        print("Sampling for evaluation...")
        posterior_samples = model.sample_posterior(x_obs, 1000)
        
        # Metrics
        metrics = compute_metrics(posterior_samples, theta_true)
        print(f"Round {round_idx} Metrics: {metrics}")
        
        # Save Plot
        from run_experiment import plot_posterior, plot_combined_posteriors
        plot_dir = f"{res_dir}/round_{round_idx}"
        os.makedirs(plot_dir, exist_ok=True)
        plot_posterior(posterior_samples, theta_true, plot_dir, "posterior.png")
        
        # Store samples
        all_rounds_samples[f"Round {round_idx}"] = posterior_samples
        
        # Save Model
        torch.save(model.state_dict(), f"{plot_dir}/model.pt")

    # Final Combined Plot
    print("\nGenerating final combined posterior plot for all rounds...")
    plot_combined_posteriors(all_rounds_samples, theta_true, res_dir, "all_rounds_posterior_comparison.png")
    
    # Save all samples
    np.savez(f"{res_dir}/all_rounds_samples.npz", **all_rounds_samples)
    print(f"Saved all samples to {res_dir}/all_rounds_samples.npz")

if __name__ == "__main__":
    main()
