import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.smmd import SummaryNet

class DNNABC_Model(nn.Module):
    def __init__(self, d=5, d_x=3, n_points=50):
        super().__init__()
        # DNNABC uses the SummaryNet to predict theta directly
        # So output_dim should be d
        self.net = SummaryNet(n_points=n_points, input_dim=d_x, output_dim=d)
        self.d = d
        
    def forward(self, x):
        # x: (batch, n_points, d_x)
        # returns: (batch, d) -> predicted theta
        return self.net(x)

def train_dnnabc(model, train_loader, epochs, device):
    """
    Train DNNABC Regression Model.
    Loss: MSE between predicted theta and true theta.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = nn.MSELoss()
    
    model.to(device)
    model.train()
    
    loss_history = []
    print("Starting training (DNNABC)...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, theta_batch in train_loader:
            x_batch = x_batch.to(device)
            theta_batch = theta_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            with torch.enable_grad():
                theta_pred = model(x_batch)
                
                # Loss
                loss = criterion(theta_pred, theta_batch)
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
    return loss_history

def abc_rejection_sampling(model, x_obs, task, n_samples=1000, n_pool=100000, batch_size=10000, device="cpu"):
    """
    Perform ABC Rejection Sampling using the trained DNNABC model as summary statistics.
    
    1. Compute S_obs = Model(x_obs)
    2. Sample theta_prior, simulate x_sim
    3. Compute S_sim = Model(x_sim)
    4. Accept top k samples closest to S_obs
    """
    model.eval()
    
    # 1. Compute S_obs
    # x_obs: (n, d_x) or (1, n, d_x)
    if not isinstance(x_obs, torch.Tensor):
        x_obs = torch.from_numpy(x_obs).float()
    if x_obs.ndim == 2:
        x_obs = x_obs.unsqueeze(0)
    x_obs = x_obs.to(device)
    
    with torch.no_grad():
        s_obs = model(x_obs) # (1, d)
        
    # 2. Rejection Sampling
    # Simulating N (large) samples, computing distances, taking top k (n_samples).
    
    N_POOL = n_pool
    
    all_thetas = []
    all_distances = []
    
    print(f"DNNABC: Simulating {N_POOL} samples for rejection...")
    
    # Process in batches
    num_batches = N_POOL // batch_size
    
    # If N_POOL < batch_size, run at least one batch of size N_POOL
    if num_batches == 0:
        batch_size = N_POOL
        num_batches = 1
    
    # Or better: Iterate with step
    # range(0, N_POOL, batch_size) handles the remainder automatically if we slice properly
    
    for i in range(0, N_POOL, batch_size):
        current_batch_size = min(batch_size, N_POOL - i)
        
        # Sample Prior
        theta_batch_np = task.sample_prior(current_batch_size)
        
        # Simulate
        x_batch_np = task.simulator(theta_batch_np)
        
        # To Tensor
        x_batch = torch.from_numpy(x_batch_np).float().to(device)
        
        # Compute Summary (Prediction)
        with torch.no_grad():
            s_sim = model(x_batch) # (batch, d)
            
        # Euclidean Distance between summaries
        # s_obs is (1, d)
        dists = torch.norm(s_sim - s_obs, dim=1).cpu().numpy()
        
        all_thetas.append(theta_batch_np)
        all_distances.append(dists)
        
    all_thetas = np.concatenate(all_thetas, axis=0)
    all_distances = np.concatenate(all_distances, axis=0)
    
    # 3. Select top k
    # Sort by distance
    idx_sorted = np.argsort(all_distances)
    accepted_idx = idx_sorted[:n_samples]
    
    accepted_samples = all_thetas[accepted_idx]
    eps = all_distances[accepted_idx[-1]]
    
    print(f"DNNABC: Accepted {n_samples} samples. Max Epsilon: {eps:.4f}")
    
    return accepted_samples
