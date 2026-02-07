
"""
PyTorch implementation of Slicing MMD-based Amortized Inference for Gaussian Toy Example.
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
        GaussianTask,
        generate_dataset,
        TRUE_PARAMS,
        n, d, d_x, p
    )
except ImportError:
    from Gaussian.data_generation import (
        GaussianTask,
        generate_dataset,
        TRUE_PARAMS,
        n, d, d_x, p
    )

# ============================================================================
# 1. Hyperparameters & Device
# ============================================================================
M = 50      # MMD approximation samples
L = 20      # Slicing directions
BATCH_SIZE = 256
EPOCHS = 500
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
            dense_s1_args={"units": 64}, # Increased slightly for 3D inputs
            dense_s2_args={"units": 64},
            dense_s3_args={"units": 64},
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
    bandwidth = 1.0 / (2.0 * n)
    
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

def train_smmd_torch(theta_train, x_train, result_dir, summary_dim=p):
    print(f"Training SMMD (PyTorch) with p={summary_dim} on {DEVICE}...")
    
    # Prepare Data
    theta_tensor = torch.from_numpy(theta_train).float()
    x_tensor = torch.from_numpy(x_train).float()
    
    dataset = TensorDataset(theta_tensor, x_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Init Model
    model = SMMD_Model(summary_dim=summary_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # LR Scheduler (Cosine Decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Train
    print(f"Training SMMD (PyTorch) for {EPOCHS} epochs...")
    
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
        
        # Step LR Scheduler
        scheduler.step()
        
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
            
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    # Plot Loss
    plt.figure()
    plt.plot(loss_history)
    plt.title(f"Loss (PyTorch)")
    plt.savefig(os.path.join(result_dir, f"loss_smmd.png"))
    plt.close()
    
    # Save Model
    torch.save(model.state_dict(), os.path.join(result_dir, "smmd_model.pth"))
    
    return model

def evaluate_posterior_torch(model, x_obs, true_params, result_dir):
    """
    Generate posterior samples and plot.
    """
    model.eval()
    n_post_samples = 10000
    
    # x_obs: (n, d_x) -> (1, n, d_x)
    x_obs = x_obs[np.newaxis, ...]
    x_obs_tensor = torch.from_numpy(x_obs).float().to(DEVICE)
    
    # 1. Get Summary Stats and Generate
    with torch.no_grad():
        stats = model.T(x_obs_tensor) # (1, p)
        Z = torch.randn(1, n_post_samples, d, device=DEVICE)
        post_samples = model.G(Z, stats) # (1, n_post_samples, d)
        post_samples = post_samples.squeeze(0).cpu().numpy()
        
    # Plot Corner Plot
    param_names = ['m0', 'm1', 's0', 's1', 'r']
    
    # Create DataFrame
    df = {}
    for i in range(d):
        df[param_names[i]] = post_samples[:, i]
    
    df = pd.DataFrame(df)
    
    plot_limit = 4.0 # Prior is -3 to 3 approx
    
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
                
    g.fig.suptitle(f"Posterior SMMD (PyTorch)", y=1.02)
    plt.savefig(os.path.join(result_dir, f"posterior_smmd.png"))
    plt.close()
    
    return post_samples

# ============================================================================
# 5. Local Refinement (ABC-MCMC)
# ============================================================================

def compute_bandwidth_torch(model, x_obs, task, n_samples=5000, quantile_level=0.005):
    """
    Compute bandwidth for likelihood estimation using quantiles of distance.
    """
    print("Computing bandwidth for refinement...")
    model.eval()
    
    # 1. Generate Theta0 from learned model
    # x_obs: (n, d_x)
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(DEVICE) # (1, n, d_x)
    
    # Replicate x_obs for batch generation
    x_obs_batch = x_obs_tensor.expand(n_samples, -1, -1) # (N0, n, d_x)
    
    with torch.no_grad():
        stats = model.T(x_obs_batch) # (N0, p)
        # We need M=1 per sample
        Z = torch.randn(n_samples, 1, d, device=DEVICE)
        Theta0 = model.G(Z, stats).squeeze(1) # (N0, d)
        
    Theta0_np = Theta0.cpu().numpy()
    
    # 2. Simulate Data from Theta0
    # Use GaussianTask simulator (batch)
    # Output: (N0, n, d_x)
    xn_0 = task.simulator(Theta0_np, n_samples=n) 
    xn_0_tensor = torch.from_numpy(xn_0).float().to(DEVICE)
    
    # 3. Compute Stats and Distance
    with torch.no_grad():
        TT = model.T(xn_0_tensor) # (N0, p)
        # Target stats (computed once)
        T_target = model.T(x_obs_tensor) # (1, p)
        
        # Distance (Euclidean)
        diff = T_target - TT
        dist_sq = torch.sum(diff**2, dim=1) # (N0,)
        dist = torch.sqrt(dist_sq)
        
    dist_np = dist.cpu().numpy()
    
    # 4. Quantile
    quan1 = np.quantile(dist_np, quantile_level)
    # Removed upper bound check as requested
    
    print(f"Computed bandwidth (epsilon): {quan1}")
    return quan1

def approximate_likelihood_torch(model, theta, x_obs_stats, task, nsims, epsilon):
    """
    Compute KDE-based likelihood for a batch of parameters.
    theta: (batch, d) numpy
    x_obs_stats: (1, p) torch tensor
    Returns: (batch,) numpy
    """
    batch_size = theta.shape[0]
    
    # 1. Simulate: (batch * nsims, n, d_x)
    # Replicate theta for nsims
    # theta: (B, d) -> (B, 1, d) -> (B, nsims, d) -> (B*nsims, d)
    theta_exp = np.repeat(theta, nsims, axis=0)
    
    sim_data = task.simulator(theta_exp, n_samples=n)
    sim_data_tensor = torch.from_numpy(sim_data).float().to(DEVICE)
    
    # 2. Compute Stats: (B*nsims, p)
    with torch.no_grad():
        sim_stats = model.T(sim_data_tensor)
        
    # 3. Reshape and Compute Distance
    # (B, nsims, p)
    sim_stats = sim_stats.view(batch_size, nsims, -1)
    
    # x_obs_stats: (1, p) -> (1, 1, p)
    target = x_obs_stats.unsqueeze(0)
    
    # Dist sq: (B, nsims)
    diff = sim_stats - target
    dist_sq = torch.sum(diff**2, dim=-1)
    
    # 4. KDE
    # kernel = exp(-dist_sq / (2 * epsilon^2))
    # likelihood ~ mean(kernel) over nsims
    kernel = torch.exp(-dist_sq / (2 * epsilon**2))
    likelihood = torch.mean(kernel, dim=1) # (B,)
    
    return likelihood.cpu().numpy()

def refine_posterior(model, x_obs, task, 
                     n_chains=1000, n_samples=1, burn_in=99, 
                     thin=1, nsims=50, epsilon=None, proposal_std=0.5):
    """
    Run Parallel MCMC Refinement.
    Default: 1000 chains, 99 burn-in, take 1 sample (the last one).
    Total steps = burn_in + (n_samples-1)*thin + 1 if we take the last.
    Wait, logic should be:
    Run burn_in steps.
    Then run n_samples * thin steps.
    Keep every thin-th sample.
    If n_samples=1, we just need 1 sample after burn_in.
    """
    print(f"Starting MCMC Refinement (chains={n_chains}, burn_in={burn_in}, samples={n_samples})...")
    model.eval()
    
    # 1. Compute Bandwidth if not provided
    if epsilon is None:
        epsilon = compute_bandwidth_torch(model, x_obs, task)
        
    # 2. Initialize Chains from Model Posterior
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(DEVICE)
    with torch.no_grad():
        x_obs_stats = model.T(x_obs_tensor) # (1, p)
        Z = torch.randn(n_chains, 1, d, device=DEVICE)
        current_theta = model.G(Z, x_obs_stats.expand(n_chains, -1)).squeeze(1).cpu().numpy() # (chains, d)
        
    # Clip to bounds initially
    current_theta = np.clip(current_theta, task.lower, task.upper)
    
    # 3. Initial Log Probabilities
    # Prior
    current_log_prior = task.log_prior(current_theta) # (chains,)
    
    # Likelihood (Linear space to match reference, but careful with zeros)
    current_likelihood = approximate_likelihood_torch(model, current_theta, x_obs_stats, task, nsims, epsilon)
    
    # Current Ratio (Prior * Likelihood)
    current_prob = np.exp(current_log_prior) * current_likelihood
    
    # Storage
    samples = []
    total_accepted = 0
    
    start_time = time.time()
    
    # MCMC Loop
    # We want to collect `n_samples` samples, separated by `thin` steps.
    # Total sampling steps = (n_samples - 1) * thin + 1? 
    # Or typically: run thin steps, collect, repeat n_samples times.
    # Total steps after burn-in = n_samples * thin.
    
    total_sampling_steps = n_samples * thin
    total_steps = burn_in + total_sampling_steps
    
    print(f"Total MCMC steps: {total_steps} (Burn-in: {burn_in}, Sampling: {total_sampling_steps})")
    
    for step in range(1, total_steps + 1):
        # Propose
        proposal_noise = np.random.randn(n_chains, d) * proposal_std
        proposed_theta = current_theta + proposal_noise
        
        # Proposed Log Prob
        proposed_log_prior = task.log_prior(proposed_theta)
        
        proposed_likelihood = approximate_likelihood_torch(model, proposed_theta, x_obs_stats, task, nsims, epsilon)
        proposed_prob = np.exp(proposed_log_prior) * proposed_likelihood
        
        # Acceptance Probability
        ratio = np.divide(proposed_prob, current_prob, out=np.zeros_like(current_prob), where=current_prob!=0)
        accept_prob = np.minimum(1.0, ratio)
        
        # Random Uniform
        u = np.random.rand(n_chains)
        accept_mask = u < accept_prob
        
        # Update
        current_theta[accept_mask] = proposed_theta[accept_mask]
        current_prob[accept_mask] = proposed_prob[accept_mask]
        
        if step > burn_in:
            total_accepted += np.sum(accept_mask)
            
            # Check if we should collect sample
            # steps relative to burn_in: 1, 2, ... total_sampling_steps
            sampling_step = step - burn_in
            if sampling_step % thin == 0:
                samples.append(current_theta.copy())
                
        if step % 10 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            print(f"Step {step}/{total_steps} | Accepted: {np.mean(accept_mask):.2%} | Time: {elapsed:.2f}s")
            
    # Collect Samples
    if not samples:
        # Fallback if logic fails (e.g. n_samples=0?)
        samples.append(current_theta.copy())
        
    # samples: List of (chains, d) -> (n_saved, chains, d) -> (n_saved * chains, d)
    posterior_samples = np.vstack(samples)
    
    acceptance_rate = total_accepted / (n_chains * total_sampling_steps) if total_sampling_steps > 0 else 0
    print(f"Refinement Complete. Acceptance Rate: {acceptance_rate:.2%}")
    
    return posterior_samples

if __name__ == "__main__":
    pass
