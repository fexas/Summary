"""
PyTorch implementation of Slicing MMD-based Amortized Inference for Stochastic SIR.
Optimized for Mac M1 Max (MPS).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import from local data_generation
try:
    from data_generation import (
        simulator, 
        sample_prior,
        TRUE_PARAMS,
        PRIOR_MIN,
        PRIOR_MAX,
        d, d_x, NUM_OBS
    )
except ImportError:
    from stochastic_sir.data_generation import (
        simulator, 
        sample_prior,
        TRUE_PARAMS,
        PRIOR_MIN,
        PRIOR_MAX,
        d, d_x, NUM_OBS
    )

# ============================================================================
# 1. Hyperparameters & Device
# ============================================================================
p = 10      # Summary statistics dimension
M = 50      # MMD approximation samples
L = 20      # Slicing directions
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.001

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

# ============================================================================
# 2. Neural Networks (DeepSets + Generator)
# ============================================================================

class InvariantModule(nn.Module):
    def __init__(self, settings):
        super().__init__()
        
        # S1: Dense layers before pooling
        s1_layers = []
        in_dim = settings["input_dim"]
        for _ in range(settings["num_dense_s1"]):
            s1_layers.append(nn.Linear(in_dim, settings["dense_s1_args"]["units"]))
            s1_layers.append(nn.ReLU())
            in_dim = settings["dense_s1_args"]["units"]
        self.s1 = nn.Sequential(*s1_layers)
        self.s1_out_dim = in_dim
        
        # S2: Dense layers after pooling
        s2_layers = []
        in_dim = self.s1_out_dim
        for _ in range(settings["num_dense_s2"]):
            s2_layers.append(nn.Linear(in_dim, settings["dense_s2_args"]["units"]))
            s2_layers.append(nn.ReLU())
            in_dim = settings["dense_s2_args"]["units"]
        self.s2 = nn.Sequential(*s2_layers)
        self.output_dim = in_dim

    def forward(self, x):
        # x: (batch, n_points, input_dim)
        x_s1 = self.s1(x) # (batch, n_points, s1_out_dim)
        x_reduced = torch.mean(x_s1, dim=1) # (batch, s1_out_dim)
        return self.s2(x_reduced) # (batch, s2_out_dim)

class EquivariantModule(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.invariant_module = InvariantModule(settings)
        
        # S3: Dense layers combining original x and invariant features
        s3_layers = []
        # Input to S3 is original input_dim + invariant_output_dim
        in_dim = settings["input_dim"] + self.invariant_module.output_dim
        for _ in range(settings["num_dense_s3"]):
            s3_layers.append(nn.Linear(in_dim, settings["dense_s3_args"]["units"]))
            s3_layers.append(nn.ReLU())
            in_dim = settings["dense_s3_args"]["units"]
        self.s3 = nn.Sequential(*s3_layers)
        self.output_dim = in_dim

    def forward(self, x):
        # x: (batch, n_points, input_dim)
        batch_size, n_points, _ = x.shape
        
        # Invariant path
        out_inv = self.invariant_module(x) # (batch, inv_out_dim)
        
        # Expand and tile: (batch, n_points, inv_out_dim)
        out_inv_rep = out_inv.unsqueeze(1).expand(-1, n_points, -1)
        
        # Concatenate: (batch, n_points, input_dim + inv_out_dim)
        out_c = torch.cat([x, out_inv_rep], dim=-1)
        
        return self.s3(out_c) # (batch, n_points, s3_out_dim)

class SummaryNet(nn.Module):
    def __init__(self, n_points=NUM_OBS, input_dim=d_x, output_dim=p):
        super().__init__()
        
        settings = dict(
            num_dense_s1=2, num_dense_s2=2, num_dense_s3=2,
            dense_s1_args={"units": 32},
            dense_s2_args={"units": 32},
            dense_s3_args={"units": 32},
            input_dim=input_dim
        )
        
        # Layer 1: Equivariant
        self.equiv1 = EquivariantModule(settings)
        
        # Layer 2: Equivariant (update input dim)
        settings_l2 = settings.copy()
        settings_l2["input_dim"] = self.equiv1.output_dim
        self.equiv2 = EquivariantModule(settings_l2)
        
        # Layer 3: Invariant (update input dim)
        settings_l3 = settings.copy()
        settings_l3["input_dim"] = self.equiv2.output_dim
        self.inv = InvariantModule(settings_l3)
        
        # Output Layer
        self.out_layer = nn.Linear(self.inv.output_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, n_points, input_dim)
        x = self.equiv1(x)
        x = self.equiv2(x)
        x = self.inv(x)
        return self.out_layer(x)

class Generator(nn.Module):
    def __init__(self, z_dim=d, stats_dim=p, out_dim=d):
        super().__init__()
        input_dim = z_dim + stats_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            # Output activation: Scaled Sigmoid to match Prior Range?
            # Or just raw and clip? Or Softplus for positive?
            # Prior is U[0.01, 1.0].
            # Let's use Sigmoid * (Max - Min) + Min
        )
        self.prior_min = torch.tensor(PRIOR_MIN, device=DEVICE, dtype=torch.float32)
        self.prior_max = torch.tensor(PRIOR_MAX, device=DEVICE, dtype=torch.float32)
        self.prior_range = self.prior_max - self.prior_min
        
    def forward(self, z, stats):
        # z: (batch, M, z_dim)
        # stats: (batch, stats_dim)
        
        # Expand stats: (batch, M, stats_dim)
        stats_exp = stats.unsqueeze(1).expand(-1, z.size(1), -1)
        
        # Concat: (batch, M, z_dim + stats_dim)
        gen_input = torch.cat([z, stats_exp], dim=-1)
        
        raw_out = self.net(gen_input) # (batch, M, out_dim)
        
        # Map to prior range
        return torch.sigmoid(raw_out) * self.prior_range + self.prior_min

class SMMD_Model(nn.Module):
    def __init__(self, summary_dim=p):
        super().__init__()
        self.T = SummaryNet(output_dim=summary_dim)
        self.G = Generator(stats_dim=summary_dim)
        
    def forward(self, x_obs, z):
        stats = self.T(x_obs)
        theta_fake = self.G(z, stats)
        return theta_fake

# ============================================================================
# 3. Loss Function (Sliced MMD)
# ============================================================================

def sliced_mmd_loss(theta_true, theta_fake, num_slices=L):
    # theta_true: (batch, d) -> unsqueeze to (batch, 1, d)
    # theta_fake: (batch, M, d)
    
    batch_size, M, dim = theta_fake.shape
    theta_true = theta_true.unsqueeze(1) # (batch, 1, d)
    
    # Generate random directions: (num_slices, d)
    directions = torch.randn(num_slices, dim, device=theta_fake.device)
    directions = directions / torch.norm(directions, dim=1, keepdim=True)
    
    # Project: (batch, M, d) @ (d, num_slices) -> (batch, M, num_slices)
    proj_fake = torch.matmul(theta_fake, directions.T)
    # Project: (batch, 1, d) @ (d, num_slices) -> (batch, 1, num_slices)
    proj_true = torch.matmul(theta_true, directions.T)
    
    # Sort projections along sample dimension M
    # Note: theta_true has only 1 sample, so it's a Dirac delta.
    # Sliced MMD usually compares two distributions. Here one is a Dirac.
    # The Wasserstein-1 distance between a distribution and a Dirac at x0 is E[|x - x0|].
    # But SMMD uses sorted differences?
    # If M > 1 and True is 1, sorting True is trivial.
    # Comparing a distribution to a point mass:
    # We want the distribution to collapse to the point mass (if consistent).
    # Or matches the posterior.
    # Wait, in Amortized Inference (e.g. BayesFlow), we minimize distance between
    # Joint (theta, x) and Product (theta_fake, x).
    # Here we are feeding x_obs and generating theta_fake.
    # We want theta_fake ~ p(theta|x_obs).
    # Since we know theta_true for x_obs, and x_obs was generated from theta_true...
    # The posterior should be centered around theta_true (with some spread due to noise).
    # Minimizing MMD(theta_fake, theta_true) where theta_true is a SINGLE point
    # forces theta_fake to collapse to theta_true (mode collapse).
    # This trains a point estimator, NOT a posterior estimator, UNLESS there is noise in x_obs
    # that makes theta_true uncertain?
    # Actually, for a fixed x_obs, there is a true posterior.
    # If we use a single sample theta_true, we are effectively minimizing Expected Loss over joint p(theta, x).
    # E_{theta, x} [ D( q(theta|x), delta(theta) ) ] ?
    # If D is MMD, and target is Dirac, we learn to predict the mean/mode.
    # To learn the posterior variance, we usually need something else (like KL in flow).
    # But let's stick to the SMMD implementation provided in G_and_K/smmd_torch.py.
    # It seems to treat theta_true as the target distribution (of size 1).
    
    sorted_fake, _ = torch.sort(proj_fake, dim=1)
    sorted_true, _ = torch.sort(proj_true, dim=1) # Trivial sort for dim 1
    
    # Broadcast true to M (replicate)
    sorted_true = sorted_true.expand(-1, M, -1)
    
    # L2 distance between sorted projections
    loss = torch.mean((sorted_fake - sorted_true)**2)
    return loss

# ============================================================================
# 4. Training Loop
# ============================================================================

def train_smmd():
    print("Generating training data...")
    # Generate training data
    # 2000 batches of size BATCH_SIZE? Or fixed dataset?
    # Let's generate a fixed dataset for stability
    N_TRAIN = 10000
    
    # Sample Prior
    theta_train = sample_prior(N_TRAIN)
    # Simulate
    x_train = simulator(theta_train, n_points=NUM_OBS)
    
    # Convert to Tensor
    dataset = TensorDataset(
        torch.from_numpy(theta_train).float(),
        torch.from_numpy(x_train).float()
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    model = SMMD_Model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    losses = []
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for theta_batch, x_batch in dataloader:
            theta_batch = theta_batch.to(DEVICE)
            x_batch = x_batch.to(DEVICE)
            
            # Sample noise z
            z = torch.randn(x_batch.size(0), M, d, device=DEVICE)
            
            # Forward
            theta_fake = model(x_batch, z)
            
            # Loss
            loss = sliced_mmd_loss(theta_batch, theta_fake)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
            
    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # Save Loss Plot
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Sliced MMD Loss")
    plt.title("SMMD Training Loss (SIR)")
    os.makedirs("stochastic_sir/smmd_result", exist_ok=True)
    plt.savefig("stochastic_sir/smmd_result/loss.png")
    plt.close()
    
    return model

# ============================================================================
# 5. Evaluation
# ============================================================================

def evaluate(model):
    print("Evaluating on ground truth...")
    # Generate observation from TRUE_PARAMS
    # We use a single observation for inference
    x_obs = simulator(TRUE_PARAMS, n_points=NUM_OBS) # (1, n_points, 2)
    x_obs_tensor = torch.from_numpy(x_obs).float().to(DEVICE)
    
    # Generate posterior samples
    N_SAMPLES = 1000
    # We can generate M samples at once, need N_SAMPLES total
    # Run in batches of 1 (since x_obs is 1)
    
    model.eval()
    with torch.no_grad():
        # z: (1, N_SAMPLES, d)
        z = torch.randn(1, N_SAMPLES, d, device=DEVICE)
        posterior_samples = model(x_obs_tensor, z)
        
    samples = posterior_samples[0].cpu().numpy()
    
    # Plot pairplot
    df = pd.DataFrame(samples, columns=["beta", "gamma"])
    
    g = sns.pairplot(df, diag_kind="kde", corner=True)
    g.fig.suptitle("SMMD Posterior (SIR)", y=1.02)
    
    # Plot True Params
    # axes is a 2x2 grid (or smaller due to corner=True)
    # beta is col 0, gamma is col 1
    
    # Subplot [0,0] (beta hist), [1,0] (gamma vs beta), [1,1] (gamma hist)
    # True params: beta=0.4, gamma=0.1
    
    # We can manually add the red dot
    # Extract axes
    axes = g.axes
    
    # Beta distribution (0,0)
    axes[0,0].axvline(TRUE_PARAMS[0], color='r', linestyle='--')
    
    # Gamma vs Beta (1,0)
    axes[1,0].scatter(TRUE_PARAMS[0], TRUE_PARAMS[1], color='r', s=50, zorder=5)
    
    # Gamma distribution (1,1)
    axes[1,1].axvline(TRUE_PARAMS[1], color='r', linestyle='--')
    
    plt.savefig("stochastic_sir/smmd_result/posterior.png")
    print("Posterior plot saved.")

if __name__ == "__main__":
    model = train_smmd()
    evaluate(model)
i