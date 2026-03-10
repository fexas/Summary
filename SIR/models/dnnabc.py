import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .smmd import TimeSeriesSummaryNet

class DNNABC_Model(nn.Module):
    def __init__(self, d=5, d_x=3, n_points=50):
        super().__init__()
        self.net = TimeSeriesSummaryNet(n_points=n_points, input_dim=d_x, output_dim=d)
        self.d = d

    def forward(self, x):
        return self.net(x)

def train_dnnabc(model, train_loader, epochs, device):
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = nn.MSELoss()
    model.to(device)
    model.train()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, theta_batch in train_loader:
            x_batch = x_batch.to(device)
            theta_batch = theta_batch.to(device)
            optimizer.zero_grad()
            with torch.enable_grad():
                theta_pred = model(x_batch)
                loss = criterion(theta_pred, theta_batch)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
    return loss_history

def abc_rejection_sampling(model, x_obs, task, n_samples=1000, n_pool=100000, batch_size=10000, device="cpu"):
    model.eval()
    if not isinstance(x_obs, torch.Tensor):
        x_obs = torch.from_numpy(x_obs).float()
    if x_obs.ndim == 2:
        x_obs = x_obs.unsqueeze(0)
    x_obs = x_obs.to(device)
    with torch.no_grad():
        s_obs = model(x_obs)
    N_POOL = n_pool
    all_thetas = []
    all_distances = []
    for i in range(0, N_POOL, batch_size):
        current_batch_size = min(batch_size, N_POOL - i)
        theta_batch_np = task.sample_prior(current_batch_size)
        x_batch_np = task.simulator(theta_batch_np)
        x_batch = torch.from_numpy(x_batch_np).float().to(device)
        with torch.no_grad():
            s_sim = model(x_batch)
        dists = torch.norm(s_sim - s_obs, dim=1).cpu().numpy()
        all_thetas.append(theta_batch_np)
        all_distances.append(dists)
    all_thetas = np.concatenate(all_thetas, axis=0)
    all_distances = np.concatenate(all_distances, axis=0)
    idx_sorted = np.argsort(all_distances)
    accepted_idx = idx_sorted[:n_samples]
    accepted_samples = all_thetas[accepted_idx]
    return accepted_samples

