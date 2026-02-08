
import torch
import torch.nn as nn
from models.smmd import SMMD_Model

def mmd_loss(theta_true, theta_fake, n_points=50):
    """
    Compute Maximum Mean Discrepancy (MMD) with Gaussian Kernel.
    theta_true: (batch, d)
    theta_fake: (batch, M, d)
    """
    batch_size, M, dim = theta_fake.shape
    device = theta_fake.device
    
    # Bandwidths for Multi-scale Kernel (following reference code if possible, or simple heuristic)
    # Reference: bandwidth = [1/n, 4/n, 9/n, 16/n, 25/n]
    # n is sample size? In reference n=50.
    
    # Let's use the reference bandwidths
    # bandwidth = tf.constant([1 / n, 4 / n, 9 / n, 16 / n, 25 / n], "float32")
    # coefficient = bandwidth ** (d / 2) -> 1/coefficient
    
    bandwidths = torch.tensor([20.0/n_points], device=device)
    # Reshape for broadcasting: (K, 1, 1, 1) where K is number of kernels
    bandwidths = bandwidths.view(-1, 1, 1, 1)
    
    # Expand Inputs
    # theta_true: (batch, 1, d)
    theta_true_exp = theta_true.unsqueeze(1)
    
    # Calculate Pairwise Squared Distances using cdist for efficiency
    # 1. G vs G: (batch, M, M)
    dist_sq_GG = torch.cdist(theta_fake, theta_fake, p=2)**2
    
    # 2. T vs T: (batch, 1, 1) - Dist is 0
    # dist_sq_TT = torch.zeros((batch_size, 1, 1), device=device)
    # Using cdist for consistency (though it's 0)
    # dist_sq_TT = torch.cdist(theta_true_exp, theta_true_exp, p=2)**2
    
    # 3. G vs T: (batch, M, 1)
    dist_sq_GT = torch.cdist(theta_fake, theta_true_exp, p=2)**2
    
    # Apply Kernels
    # K_GG: (K, batch, M, M)
    # Note: We sum kernels without normalization coefficients (like 1/sigma^d) 
    # as is common in MMD implementations (e.g. the reference article).
    K_GG = torch.exp(-0.5 * dist_sq_GG.unsqueeze(0) / bandwidths)
    
    # K_TT: (K, batch, 1, 1)
    # Since dist is 0, exp(0) = 1.
    K_TT = torch.ones((bandwidths.shape[0], batch_size, 1, 1), device=device)
    
    # K_GT: (K, batch, M, 1)
    K_GT = torch.exp(-0.5 * dist_sq_GT.unsqueeze(0) / bandwidths)
    
    # Sum over Kernels
    total_K_GG = torch.sum(K_GG, dim=0) # (batch, M, M)
    total_K_TT = torch.sum(K_TT, dim=0) # (batch, 1, 1)
    total_K_GT = torch.sum(K_GT, dim=0) # (batch, M, 1)
    
    # Means
    # Use standard V-statistic (biased estimator) as in the reference article.
    # No M/(M-1) correction.
    mean_K_GG = torch.mean(total_K_GG, dim=(1, 2))
    mean_K_GT = torch.mean(total_K_GT, dim=(1, 2))
    mean_K_TT = torch.mean(total_K_TT, dim=(1, 2))
    
    # MMD squared
    mmd = mean_K_GG + mean_K_TT - 2 * mean_K_GT
    
    return torch.mean(mmd)

class MMD_Model(SMMD_Model):
    # Same architecture, just different loss during training
    pass
