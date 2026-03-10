
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = (rms + self.eps).rsqrt()
        return x * rms * self.weight


def _init_normal_0_2(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

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
        
        settings_l3 = settings.copy()
        settings_l3["input_dim"] = self.equiv2.output_dim
        self.inv = InvariantModule(settings_l3)
        self.post_pool_norm = RMSNorm(self.inv.output_dim)
        self.out_layer = nn.Linear(self.inv.output_dim, output_dim)
        self.apply(_init_normal_0_2)
        
    def forward(self, x):
        x = self.equiv1(x)
        x = self.equiv2(x)
        x = self.inv(x)
        x = self.post_pool_norm(x)
        return self.out_layer(x)

class Generator(nn.Module):
    def __init__(self, z_dim=5, stats_dim=10, out_dim=5):
        super().__init__()
        input_dim = z_dim + stats_dim
        layers = []
        in_dim = input_dim
        hidden_dim = 64
        for _ in range(3):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(RMSNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(_init_normal_0_2)
        
    def forward(self, z, stats):
        stats_exp = stats.unsqueeze(1).expand(-1, z.size(1), -1)
        gen_input = torch.cat([z, stats_exp], dim=-1)
        return self.net(gen_input)

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

def sliced_mmd_loss(theta_true, theta_fake, num_slices=20, n_points=50):
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
    bandwidth = 1.0 / (1.0 * n_points)
    
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
    
    # MMD Loss = E[K_GG] + E[K_TT] - 2*E[K_GT]
    loss = loss_GG + loss_TT - 2 * loss_GT
    
    return torch.mean(loss) # Mean over batch
