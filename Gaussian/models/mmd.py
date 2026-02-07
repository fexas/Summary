
import torch
import torch.nn as nn
from Gaussian.models.smmd import SMMD_Model

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
    
    bandwidths = torch.tensor([1.0/n_points, 4.0/n_points, 9.0/n_points, 16.0/n_points, 25.0/n_points], device=device)
    # Reshape for broadcasting: (K, 1, 1, 1) where K is number of kernels
    bandwidths = bandwidths.view(-1, 1, 1, 1)
    
    # Coefficients: (K, 1, 1, 1)
    # coefficient = 1 / (bandwidth ** (dim / 2))
    coefficients = 1.0 / (bandwidths ** (dim / 2.0))
    
    # Expand Inputs
    # theta_true: (batch, 1, d)
    theta_true_exp = theta_true.unsqueeze(1)
    
    # theta_fake: (batch, M, d)
    
    # Calculate Pairwise Squared Distances
    
    # 1. G vs G: (batch, M, M)
    # (batch, M, 1, d) - (batch, 1, M, d) -> (batch, M, M, d)
    diff_GG = theta_fake.unsqueeze(2) - theta_fake.unsqueeze(1)
    dist_sq_GG = torch.sum(diff_GG**2, dim=-1) # (batch, M, M)
    
    # 2. T vs T: (batch, 1, 1) - Dist is 0
    dist_sq_TT = torch.zeros((batch_size, 1, 1), device=device)
    
    # 3. G vs T: (batch, M, 1)
    # (batch, M, d) - (batch, 1, d)
    diff_GT = theta_fake - theta_true_exp
    dist_sq_GT = torch.sum(diff_GT**2, dim=-1).unsqueeze(2) # (batch, M, 1)
    
    # Apply Kernels
    # dist_sq: (batch, ..., ...) -> expand to (1, batch, ...)
    # bandwidths: (K, 1, ...)
    
    # K_GG: (K, batch, M, M)
    K_GG = torch.exp(-0.5 * dist_sq_GG.unsqueeze(0) / bandwidths) * coefficients
    
    # K_TT: (K, batch, 1, 1)
    K_TT = torch.exp(-0.5 * dist_sq_TT.unsqueeze(0) / bandwidths) * coefficients
    
    # K_GT: (K, batch, M, 1)
    K_GT = torch.exp(-0.5 * dist_sq_GT.unsqueeze(0) / bandwidths) * coefficients
    
    # Sum over Kernels
    total_K_GG = torch.sum(K_GG, dim=0) # (batch, M, M)
    total_K_TT = torch.sum(K_TT, dim=0) # (batch, 1, 1)
    total_K_GT = torch.sum(K_GT, dim=0) # (batch, M, 1)
    
    # Means
    # Biased estimate for MMD? Reference uses M/(M-1) correction for G-G term?
    # Reference: mmd = tf.reduce_mean(K_gg) * M / (M-1) - 2 * tf.reduce_mean(K_gt)
    # K_tt is not used in reference MMD function? 
    # Reference MMD function:
    # mmd = tf.reduce_mean(K_gg) * M / (M-1) - 2 * tf.reduce_mean(K_gt)
    # Wait, MMD should be E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)].
    # If theta_true is single sample, E[k(T,T)] is constant. Maybe omitted in optimization?
    # But for correctness, let's include it.
    
    mean_K_GG = torch.mean(total_K_GG, dim=(1, 2))
    # Correction: (M / (M-1))
    if M > 1:
        mean_K_GG = mean_K_GG * (M / (M - 1.0))
        
    mean_K_GT = torch.mean(total_K_GT, dim=(1, 2))
    
    # mean_K_TT is constant (max value of kernel), we can ignore for gradient but useful for value
    mean_K_TT = torch.mean(total_K_TT, dim=(1, 2))
    
    mmd = mean_K_GG + mean_K_TT - 2 * mean_K_GT
    
    return torch.mean(mmd)

class MMD_Model(SMMD_Model):
    # Same architecture, just different loss during training
    pass
