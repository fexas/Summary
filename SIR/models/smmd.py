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

class TimeSeriesSummaryNet(nn.Module):
    def __init__(self, n_points=None, input_dim=2, output_dim=10, hidden_dim=64):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.post_conv_norm = RMSNorm(hidden_dim * 2)
        self.apply(_init_normal_0_2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.conv_net(x)
        features = features.flatten(1)
        features = self.post_conv_norm(features)
        return self.fc(features)

SummaryNet = TimeSeriesSummaryNet

class Generator(nn.Module):
    def __init__(self, z_dim=5, stats_dim=10, out_dim=5):
        super().__init__()
        input_dim = z_dim + stats_dim
        hidden_dim = 64
        layers = [
            nn.Linear(input_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        ]
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
        if not isinstance(x_obs, torch.Tensor):
            device = next(self.parameters()).device
            x_obs = torch.from_numpy(x_obs).float().to(device)
            if x_obs.ndim == 2:
                x_obs = x_obs.unsqueeze(0)
        device = x_obs.device
        with torch.no_grad():
            stats = self.T(x_obs)
            Z = torch.randn(1, n_samples, self.d, device=device)
            samples = self.G(Z, stats).squeeze(0)
        return samples.cpu().numpy()

def sliced_mmd_loss(theta_true, theta_fake, num_slices=20, n_time_steps=151):
    batch_size, M, dim = theta_fake.shape
    device = theta_fake.device
    unit_vectors = torch.randn(dim, num_slices, device=device)
    unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=0, keepdim=True)
    proj_T = torch.matmul(theta_true.unsqueeze(1), unit_vectors)
    proj_G = torch.matmul(theta_fake, unit_vectors)
    bandwidth = 1.0 / (1.0 * n_time_steps)
    diff_GG = proj_G.unsqueeze(2) - proj_G.unsqueeze(1)
    K_GG = torch.exp(-0.5 * diff_GG.pow(2) / bandwidth)
    loss_GG = torch.mean(K_GG, dim=(1, 2, 3))
    loss_TT = torch.tensor(1.0, device=device)
    diff_GT = proj_G - proj_T
    K_GT = torch.exp(-0.5 * diff_GT.pow(2) / bandwidth)
    loss_GT = torch.mean(K_GT, dim=(1, 2))
    loss = loss_GG + loss_TT - 2 * loss_GT
    loss = torch.mean(loss)
    return loss

def mixture_sliced_mmd_loss(theta_true, theta_fake_list, weights=None, num_slices=20, n_time_steps=151):
    if weights is None:
        weights = [1.0 / len(theta_fake_list)] * len(theta_fake_list)
    total_loss = 0.0
    for w, theta_fake in zip(weights, theta_fake_list):
        total_loss = total_loss + w * sliced_mmd_loss(
            theta_true,
            theta_fake,
            num_slices=num_slices,
            n_time_steps=n_time_steps,
        )
    return total_loss
