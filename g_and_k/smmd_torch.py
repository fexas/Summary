"""
PyTorch implementation of Slicing MMD-based Amortized Inference for g-and-k distribution.
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
        PRIOR_CONFIGS, 
        TRUE_PARAMS,
        n, d, d_x
    )
except ImportError:
    from G_and_K.data_generation import (
        simulator, 
        PRIOR_CONFIGS, 
        TRUE_PARAMS,
        n, d, d_x
    )

# ============================================================================
# 1. Hyperparameters & Device
# ============================================================================
p = 10      # Summary statistics dimension
M = 50      # MMD approximation samples
L = 20      # Slicing directions
BATCH_SIZE = 256
EPOCHS = 100
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
    def __init__(self, n_points=n, input_dim=d_x, output_dim=p):
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
            nn.Linear(128, out_dim)
        )
        
    def forward(self, z, stats):
        # z: (batch, M, z_dim)
        # stats: (batch, stats_dim)
        
        # Expand stats: (batch, M, stats_dim)
        stats_exp = stats.unsqueeze(1).expand(-1, z.size(1), -1)
        
        # Concat: (batch, M, z_dim + stats_dim)
        gen_input = torch.cat([z, stats_exp], dim=-1)
        
        return self.net(gen_input) # (batch, M, out_dim)

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
    
    # 1. Random Projections
    # (dim, L)
    unit_vectors = torch.randn(dim, num_slices, device=theta_fake.device)
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=0, keepdim=True)
    
    # Projections
    # theta_true: (batch, 1, d) @ (d, L) -> (batch, 1, L)
    proj_T = torch.matmul(theta_true.unsqueeze(1), unit_vectors)
    
    # theta_fake: (batch, M, d) @ (d, L) -> (batch, M, L)
    proj_G = torch.matmul(theta_fake, unit_vectors)
    
    # 2. Compute MMD on projections (using Gaussian Kernel)
    # Bandwidth
    bandwidth = 1.0
    
    # Diff matrices: (batch, M, M, L) or (batch, 1, M, L) etc.
    # To compute efficiently:
    # Kernel(X, Y) = exp(-0.5 * (X-Y)^2 / h)
    
    # G vs G
    # (batch, M, 1, L) - (batch, 1, M, L) -> (batch, M, M, L)
    diff_GG = proj_G.unsqueeze(2) - proj_G.unsqueeze(1)
    K_GG = torch.exp(-0.5 * diff_GG.pow(2) / bandwidth)
    loss_GG = torch.mean(K_GG, dim=(1, 2, 3)) # Mean over samples and slices
    
    # T vs T (Since T has 1 sample, this is just 1.0, but for generality/batching)
    # If theta_true is (batch, 1, d), we treat it as 1 sample.
    # diff_TT is 0, exp(0) is 1.
    loss_TT = torch.tensor(1.0, device=theta_fake.device) 
    
    # G vs T
    # (batch, M, L) - (batch, 1, L) -> (batch, M, L) (broadcasting 1 to M)
    diff_GT = proj_G - proj_T # (batch, M, L)
    K_GT = torch.exp(-0.5 * diff_GT.pow(2) / bandwidth)
    loss_GT = torch.mean(K_GT, dim=(1, 2)) # Mean over M and L
    
    # MMD Loss = E[K_GG] + E[K_TT] - 2*E[K_GT]
    loss = loss_GG + loss_TT - 2 * loss_GT
    
    return torch.mean(loss) # Mean over batch

# ============================================================================
# 4. Training & Plotting
# ============================================================================

def train_smmd_torch(theta_train, x_train, prior_name, result_dir, summary_dim=p):
    print(f"Training SMMD (PyTorch) for {prior_name} with p={summary_dim} on {DEVICE}...")
    
    # Prepare Data
    theta_tensor = torch.from_numpy(theta_train).float()
    x_tensor = torch.from_numpy(x_train).float()
    
    dataset = TensorDataset(theta_tensor, x_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Init Model
    model = SMMD_Model(summary_dim=summary_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    loss_history = []
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_theta, batch_x in loader:
            batch_theta = batch_theta.to(DEVICE)
            batch_x = batch_x.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Sample Z
            curr_batch_size = batch_theta.size(0)
            z = torch.randn(curr_batch_size, M, d, device=DEVICE)
            
            # Forward
            theta_fake = model(batch_x, z)
            
            # Loss
            loss = sliced_mmd_loss(batch_theta, theta_fake, num_slices=L)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
            
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    # Plot Loss
    plt.figure()
    plt.plot(loss_history)
    plt.title(f"Loss (PyTorch) - {prior_name}")
    plt.savefig(os.path.join(result_dir, f"loss_torch_{prior_name}.png"))
    plt.close()
    
    return model

def evaluate_posterior_torch(model, x_obs, true_params, prior_name, result_dir):
    """
    Generate posterior samples and plot.
    """
    model.eval()
    n_post_samples = 2000
    
    # x_obs: (1, n, d_x)
    x_obs_tensor = torch.from_numpy(x_obs).float().to(DEVICE)
    
    # 1. Get Summary Stats
    with torch.no_grad():
        stats = model.T(x_obs_tensor) # (1, p)
        
        # Replicate stats is efficient, or we can just pass batch of Z
        # We can pass batch of Z and broadcast stats in Generator
        
        # Z: (1, n_post_samples, d) -> treat as batch=1, M=n_post_samples
        # But our generator expects (batch, M, ...).
        # We can just do one pass.
        Z = torch.randn(1, n_post_samples, d, device=DEVICE)
        
        post_samples = model.G(Z, stats) # (1, n_post_samples, d)
        post_samples = post_samples.squeeze(0).cpu().numpy()
        
    # Plot Corner Plot
    param_names = ['A', 'B', 'g', 'k']
    
    # Create DataFrame
    df = {}
    for i in range(d):
        df[param_names[i]] = post_samples[:, i]
    
    df = pd.DataFrame(df)
    
    # Get Prior Limits for Plotting (matching existing logic)
    # Handle composite names like "weak_informative_p5"
    base_prior_name = prior_name
    for key in PRIOR_CONFIGS.keys():
        if prior_name.startswith(key):
            base_prior_name = key
            break
            
    limit = PRIOR_CONFIGS[base_prior_name]['bounds_limit'] if 'bounds_limit' in PRIOR_CONFIGS[base_prior_name] else 10.0
    # Fallback if bounds_limit not in dict
    if 'bounds_limit' not in PRIOR_CONFIGS[base_prior_name]:
         # Rough estimation based on A
         limit = max(abs(PRIOR_CONFIGS[base_prior_name]['A'][0]), abs(PRIOR_CONFIGS[base_prior_name]['A'][1])) * 1.5

    plot_limit = limit * 1.1
    
    g = sns.PairGrid(df, diag_sharey=False, corner=True)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.kdeplot, fill=True)
    
    # Set limits
    for i in range(d):
        for j in range(i + 1):
             if i == j:
                 g.diag_axes[i].set_xlim(-plot_limit, plot_limit)
             else:
                 g.axes[i, j].set_xlim(-plot_limit, plot_limit)
                 g.axes[i, j].set_ylim(-plot_limit, plot_limit)

    # Mark True Params
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                g.diag_axes[i].axvline(true_params[i], color='r', linestyle='--')
            else:
                g.axes[i, j].scatter(true_params[j], true_params[i], color='r', marker='*', s=100)
                
    g.fig.suptitle(f"Posterior PyTorch ({prior_name})", y=1.02)
    plt.savefig(os.path.join(result_dir, f"posterior_torch_{prior_name}.png"))
    plt.close()
    
    return post_samples

if __name__ == "__main__":
    pass
