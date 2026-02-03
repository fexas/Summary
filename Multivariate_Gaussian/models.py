"""
SMMD Models for Multivariate Gaussian Experiment.
Adapted from G_and_K/smmd_torch.py.
"""

import torch
import torch.nn as nn

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
    def __init__(self, input_dim, output_dim=10):
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
        
    def forward(self, x):
        # x: (batch, n_points, input_dim)
        x = self.equiv1(x)
        x = self.equiv2(x)
        x = self.inv(x)
        return self.out_layer(x)

class Generator(nn.Module):
    def __init__(self, z_dim=1, stats_dim=10, out_dim=1):
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
        
        # Note: In the Gaussian experiment, we are inferring sigma (1D), 
        # so output is (batch, M, 1).
        # We might need to ensure output is positive for sigma, but let's assume
        # the network learns the parameter directly.
        out = self.net(gen_input)
        
        return out 

class SMMD_Model(nn.Module):
    def __init__(self, input_dim, summary_dim=10, theta_dim=1):
        super().__init__()
        self.T = SummaryNet(input_dim=input_dim, output_dim=summary_dim)
        self.G = Generator(z_dim=theta_dim, stats_dim=summary_dim, out_dim=theta_dim)
        self.theta_dim = theta_dim
        
    def forward(self, x_obs, z):
        stats = self.T(x_obs)
        theta_fake = self.G(z, stats)
        return theta_fake

def sliced_mmd_loss(theta_true, theta_fake, num_slices=20, bandwidth=1.0):
    # theta_true: (batch, d_theta) -> unsqueeze to (batch, 1, d_theta)
    # theta_fake: (batch, M, d_theta)
    
    batch_size, M, dim = theta_fake.shape
    
    # 1. Random Projections
    # (dim, L)
    unit_vectors = torch.randn(dim, num_slices, device=theta_fake.device)
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=0, keepdim=True)
    
    # Projections
    # theta_true: (batch, d) -> (batch, 1, d) @ (d, L) -> (batch, 1, L)
    proj_T = torch.matmul(theta_true.unsqueeze(1), unit_vectors)
    
    # theta_fake: (batch, M, d) @ (d, L) -> (batch, M, L)
    proj_G = torch.matmul(theta_fake, unit_vectors)
    
    # 2. Compute MMD on projections (using Gaussian Kernel)
    # Bandwidth passed as argument
    
    # G vs G
    # (batch, M, 1, L) - (batch, 1, M, L) -> (batch, M, M, L)
    diff_GG = proj_G.unsqueeze(2) - proj_G.unsqueeze(1)
    K_GG = torch.exp(-0.5 * diff_GG.pow(2) / bandwidth)
    loss_GG = torch.mean(K_GG, dim=(1, 2, 3)) 
    
    # T vs T
    loss_TT = torch.tensor(1.0, device=theta_fake.device) 
    
    # G vs T
    # (batch, M, L) - (batch, 1, L) -> (batch, M, L)
    diff_GT = proj_G - proj_T 
    K_GT = torch.exp(-0.5 * diff_GT.pow(2) / bandwidth)
    loss_GT = torch.mean(K_GT, dim=(1, 2)) 
    
    # MMD Loss
    loss = loss_GG + loss_TT - 2 * loss_GT
    
    return torch.mean(loss) 
