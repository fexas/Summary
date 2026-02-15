
import torch
import torch.nn as nn
import numpy as np


def _init_normal_0_2(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0.0, std=0.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

class InvariantModule(nn.Module):
    def __init__(self, settings):
        super().__init__()
        
        s1_layers = []
        in_dim = settings["input_dim"]
        for _ in range(settings["num_dense_s1"]):
            units = settings["dense_s1_args"]["units"]
            s1_layers.append(nn.Linear(in_dim, units))
            s1_layers.append(RMSNorm(units))
            s1_layers.append(nn.ReLU())
            in_dim = units
        self.s1 = nn.Sequential(*s1_layers)
        self.s1_out_dim = in_dim
        
        s2_layers = []
        in_dim = self.s1_out_dim
        for _ in range(settings["num_dense_s2"]):
            units = settings["dense_s2_args"]["units"]
            s2_layers.append(nn.Linear(in_dim, units))
            s2_layers.append(RMSNorm(units))
            s2_layers.append(nn.ReLU())
            in_dim = units
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
        
        s3_layers = []
        in_dim = settings["input_dim"] + self.invariant_module.output_dim
        for _ in range(settings["num_dense_s3"]):
            units = settings["dense_s3_args"]["units"]
            s3_layers.append(nn.Linear(in_dim, units))
            s3_layers.append(RMSNorm(units))
            s3_layers.append(nn.ReLU())
            in_dim = units
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

class TimeSeriesSummaryNet(nn.Module):
    def __init__(self, n_points=None, input_dim=2, output_dim=10, hidden_dim=64):
        super().__init__()
        # 1D CNN for Time Series Summary
        # Input: (Batch, Seq_Len, Channels)
        # We will permute to (Batch, Channels, Seq_Len) for Conv1d
        
        self.conv_net = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=10, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 2
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=10, padding=2),
            nn.ReLU(),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.post_conv_norm = RMSNorm(hidden_dim*2)
        self.apply(_init_normal_0_2)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1) # (batch, input_dim, seq_len)
        
        features = self.conv_net(x) # (batch, hidden*2, 1)
        features = features.flatten(1) # (batch, hidden*2)
        features = self.post_conv_norm(features)
        
        return self.fc(features)

# Alias for backward compatibility (and usage in DNNABC/SMMD)
SummaryNet = TimeSeriesSummaryNet

class DeepSetSummaryNet(nn.Module):
    def __init__(self, n_points=50, input_dim=3, output_dim=10):
        super().__init__()
        
        settings = dict(
            num_dense_s1=2, num_dense_s2=2, num_dense_s3=2,
            dense_s1_args={"units": 64}, 
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
        self.apply(_init_normal_0_2)
        
    def forward(self, x):
        # x: (batch, n_points, input_dim)
        x = self.equiv1(x)
        x = self.equiv2(x)
        x = self.inv(x)
        return self.out_layer(x)

class Generator(nn.Module):
    def __init__(self, z_dim=5, stats_dim=10, out_dim=5):
        super().__init__()
        input_dim = z_dim + stats_dim
        layers = []
        hidden_dim = 64
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(RMSNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(RMSNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(RMSNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(RMSNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(_init_normal_0_2)
        
    def forward(self, z, stats):
        # z: (batch, M, z_dim)
        # stats: (batch, stats_dim)
        
        # Expand stats: (batch, M, stats_dim)
        stats_exp = stats.unsqueeze(1).expand(-1, z.size(1), -1)
        
        # Concat: (batch, M, z_dim + stats_dim)
        gen_input = torch.cat([z, stats_exp], dim=-1)
        
        return self.net(gen_input) # (batch, M, out_dim)

class SMMD_Model(nn.Module):
    def __init__(self, summary_dim=10, d=5, d_x=3, n=50):
        super().__init__()
        self.T = SummaryNet(n_points=n, input_dim=d_x, output_dim=summary_dim)
        self.G = Generator(z_dim=d, stats_dim=summary_dim, out_dim=d)
        self.d = d
        
    def forward(self, x_obs, z):
        stats = self.T(x_obs)
        theta_fake = self.G(z, stats)
        return theta_fake
    
    def compute_stats(self, x):
        return self.T(x)
    
    def sample_posterior(self, x_obs, n_samples):
        # x_obs: (1, n, d_x) numpy or tensor
        # returns numpy (n_samples, d)
        
        # Check input type
        if not isinstance(x_obs, torch.Tensor):
            device = next(self.parameters()).device
            x_obs = torch.from_numpy(x_obs).float().to(device)
            if x_obs.ndim == 2:
                x_obs = x_obs.unsqueeze(0)
                
        device = x_obs.device
        with torch.no_grad():
            stats = self.T(x_obs) # (1, p)
            Z = torch.randn(1, n_samples, self.d, device=device)
            samples = self.G(Z, stats).squeeze(0) # (n_samples, d)
        
        return samples.cpu().numpy()

def sliced_mmd_loss(theta_true, theta_fake, num_slices=20, n_time_steps=151):
    # theta_true: (batch, d) -> unsqueeze to (batch, 1, d)
    # theta_fake: (batch, M, d)
    
    batch_size, M, dim = theta_fake.shape
    device = theta_fake.device
    
    # 1. Random Projections
    # (dim, L)
    unit_vectors = torch.randn(dim, num_slices, device=device)
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=0, keepdim=True)
    
    # Projections
    # theta_true: (batch, 1, d) @ (d, L) -> (batch, 1, L)
    proj_T = torch.matmul(theta_true.unsqueeze(1), unit_vectors)
    
    # theta_fake: (batch, M, d) @ (d, L) -> (batch, M, L)
    proj_G = torch.matmul(theta_fake, unit_vectors)
    
    # 2. Compute MMD on projections (using Gaussian Kernel)
    # Bandwidth
    bandwidth = 5.0 / (1.0 * n_time_steps)
    
    # Diff matrices: (batch, M, M, L) or (batch, 1, M, L) etc.
    # To compute efficiently:
    # Kernel(X, Y) = exp(-0.5 * (X-Y)^2 / h)
    
    # G vs G
    # (batch, M, 1, L) - (batch, 1, M, L) -> (batch, M, M, L)
    diff_GG = proj_G.unsqueeze(2) - proj_G.unsqueeze(1)
    K_GG = torch.exp(-0.5 * diff_GG.pow(2) / bandwidth)
    loss_GG = torch.mean(K_GG, dim=(1, 2, 3)) # Mean over samples and slices
    
    # T vs T (Since T has 1 sample, this is just 1.0, but for generality/batching)
    loss_TT = torch.tensor(1.0, device=device) 
    
    # G vs T
    # (batch, M, L) - (batch, 1, L) -> (batch, M, L) (broadcasting 1 to M)
    diff_GT = proj_G - proj_T # (batch, M, L)
    K_GT = torch.exp(-0.5 * diff_GT.pow(2) / bandwidth)
    loss_GT = torch.mean(K_GT, dim=(1, 2)) # Mean over M and L
    
    loss = loss_GG + loss_TT - 2 * loss_GT
    loss = torch.mean(loss)
    return loss

def mixture_sliced_mmd_loss(theta_true, theta_fake, bandwidths=None, num_slices=20, n_time_steps=151):
    """
    Mixture Bandwidth Sliced MMD Loss.
    
    Args:
        theta_true: (batch, d) Ground truth parameters
        theta_fake: (batch, M, d) Simulated parameters
        bandwidths: (K,) or list of floats. Bandwidths (sigma^2) for the kernel.
                    If None, defaults to [5.0/n_time_steps, 15.0/n_time_steps].
        num_slices: int, number of random slices
        n_time_steps: int, number of time steps (used for default bandwidths)
        weights: (batch,) Optional weights for importance sampling
        
    Returns:
        loss: scalar tensor
    """
    # theta_true: (batch, d) -> unsqueeze to (batch, 1, d)
    # theta_fake: (batch, M, d)
    
    batch_size, M, dim = theta_fake.shape
    device = theta_fake.device
    
    # Default bandwidths if not provided
    if bandwidths is None:
        bandwidths = [20.0 / n_time_steps, 20.0 / n_time_steps]
    
    # Ensure bandwidths is a tensor
    if not isinstance(bandwidths, torch.Tensor):
        bandwidths = torch.tensor(bandwidths, device=device, dtype=torch.float32)
    else:
        bandwidths = bandwidths.to(device)
        
    # bandwidths: (K,)
    num_kernels = bandwidths.shape[0]
    
    # Coefficients for Gaussian Kernel normalization: 1 / sqrt(2 * pi * sigma^2)
    # bandwidths here are treated as sigma^2 (variance), matching the existing code's usage
    # where exp(-0.5 * dist^2 / bandwidth).
    # So sigma = sqrt(bandwidth).
    # coef = 1 / (sqrt(2*pi) * sigma) = 1 / sqrt(2*pi*bandwidth)
    coefs = 1.0 / torch.sqrt(2 * np.pi * bandwidths) # (K,)
    
    # Reshape for broadcasting: (1, 1, 1, 1, K) or similar depending on dimensions
    
    # 1. Random Projections
    # (dim, L)
    unit_vectors = torch.randn(dim, num_slices, device=device)
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=0, keepdim=True)
    
    # Projections
    # theta_true: (batch, 1, d) @ (d, L) -> (batch, 1, L)
    proj_T = torch.matmul(theta_true.unsqueeze(1), unit_vectors)
    
    # theta_fake: (batch, M, d) @ (d, L) -> (batch, M, L)
    proj_G = torch.matmul(theta_fake, unit_vectors)
    
    # 2. Compute Mixture MMD
    
    # G vs G
    # (batch, M, 1, L) - (batch, 1, M, L) -> (batch, M, M, L)
    diff_GG = proj_G.unsqueeze(2) - proj_G.unsqueeze(1)
    diff_sq_GG = diff_GG.pow(2)
    
    # Compute mixture kernel
    # Expand dims manually for broadcasting optimization
    # diff_sq_GG: (batch, M, M, L)
    # bandwidths: (K,)
    
    # (batch, M, M, L, 1)
    d_GG = diff_sq_GG.unsqueeze(-1)
    # (1, 1, 1, 1, K)
    b_view = bandwidths.view(1, 1, 1, 1, num_kernels)
    c_view = coefs.view(1, 1, 1, 1, num_kernels)
    
    # (batch, M, M, L, K)
    K_GG_all = c_view * torch.exp(-0.5 * d_GG / b_view)
    
    # Mean over samples (M, M), slices (L), and kernels (K)
    loss_GG = torch.mean(K_GG_all, dim=(1, 2, 3, 4)) # (batch,)
    
    # T vs T
    # diff is 0. Kernel is 1.0 * coef.
    # Mixture is mean(coefs)
    loss_TT = torch.mean(coefs) # Scalar (broadcasts to batch)
    
    # G vs T
    # (batch, M, L) - (batch, 1, L) -> (batch, M, L)
    diff_GT = proj_G - proj_T
    diff_sq_GT = diff_GT.pow(2) # (batch, M, L)
    
    # (batch, M, L, 1)
    d_GT = diff_sq_GT.unsqueeze(-1)
    # (1, 1, 1, K)
    b_view_GT = bandwidths.view(1, 1, 1, num_kernels)
    c_view_GT = coefs.view(1, 1, 1, num_kernels)
    
    # (batch, M, L, K)
    K_GT_all = c_view_GT * torch.exp(-0.5 * d_GT / b_view_GT)
    
    # Mean over M, L, and K
    loss_GT = torch.mean(K_GT_all, dim=(1, 2, 3)) # (batch,)
    
    loss = loss_GG + loss_TT - 2 * loss_GT
    loss = torch.mean(loss)
    return loss
