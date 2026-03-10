import torch
import torch.nn as nn
import numpy as np
from .smmd import SMMD_Model

def mmd_loss(theta_true, theta_fake, bandwidths=None, n_time_steps=151):
    batch_size, M, dim = theta_fake.shape
    device = theta_fake.device
    if bandwidths is None:
        bandwidths = torch.tensor([10.0 / n_time_steps, 15.0 / n_time_steps], device=device)
    elif not isinstance(bandwidths, torch.Tensor):
        bandwidths = torch.tensor(bandwidths, device=device, dtype=torch.float32)
    else:
        bandwidths = bandwidths.to(device)
    bandwidths = bandwidths.view(-1, 1, 1, 1)
    coefs = (2 * np.pi * bandwidths) ** (-dim / 2)
    theta_true_exp = theta_true.unsqueeze(1)
    dist_sq_GG = torch.cdist(theta_fake, theta_fake, p=2) ** 2
    dist_sq_GT = torch.cdist(theta_fake, theta_true_exp, p=2) ** 2
    K_GG = coefs * torch.exp(-0.5 * dist_sq_GG.unsqueeze(0) / bandwidths)
    K_TT = coefs.expand(-1, batch_size, 1, 1)
    K_GT = coefs * torch.exp(-0.5 * dist_sq_GT.unsqueeze(0) / bandwidths)
    total_K_GG = torch.sum(K_GG, dim=0)
    total_K_TT = torch.sum(K_TT, dim=0)
    total_K_GT = torch.sum(K_GT, dim=0)
    mean_K_GG = torch.mean(total_K_GG, dim=(1, 2))
    mean_K_GT = torch.mean(total_K_GT, dim=(1, 2))
    mean_K_TT = torch.mean(total_K_TT, dim=(1, 2))
    mmd = mean_K_GG + mean_K_TT - 2 * mean_K_GT
    return torch.mean(mmd)

class MMD_Model(SMMD_Model):
    pass

