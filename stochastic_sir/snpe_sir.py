"""
SNPE implementation for Stochastic SIR using sbi.
Uses handcrafted summary statistics.
"""

import os
import sys

# Hack: Set HOME to /tmp to fool Arviz
os.environ["HOME"] = "/tmp"
os.environ["ARVIZ_DATA"] = "/tmp/arviz_data"

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.analysis import pairplot

# Import local
try:
    from data_generation import (
        gillespie_sir,
        interpolate_trajectory,
        calculate_summary_statistics,
        TRUE_PARAMS,
        PRIOR_MIN,
        PRIOR_MAX,
        NUM_OBS, T_MAX, OBS_TIMES, N
    )
except ImportError:
    from stochastic_sir.data_generation import (
        gillespie_sir,
        interpolate_trajectory,
        calculate_summary_statistics,
        TRUE_PARAMS,
        PRIOR_MIN,
        PRIOR_MAX,
        NUM_OBS, T_MAX, OBS_TIMES, N
    )

# ============================================================================
# 1. Simulator Wrapper for SBI
# ============================================================================

def sbi_simulator(theta):
    """
    theta: (batch, d) or (d,) Tensor
    Returns: (batch, 4) or (4,) Tensor of summary stats
    """
    is_batched = theta.ndim > 1
    theta_batch = theta if is_batched else theta.unsqueeze(0)
    
    batch_size = theta_batch.shape[0]
    results = []
    
    for i in range(batch_size):
        theta_np = theta_batch[i].numpy()
        # Run Gillespie (Single)
        times, S, I, R = gillespie_sir(theta_np, N, T_MAX)
        
        # Interpolate
        I_interp = interpolate_trajectory(times, I, OBS_TIMES)
        
        # Create raw x: (n_points, 2)
        x_raw = np.stack([I_interp/N, OBS_TIMES/T_MAX], axis=1)
        x_raw = x_raw.astype(np.float32) # (50, 2)
        
        results.append(x_raw)
        
    # Stack results: (batch, 50, 2)
    x_batch_np = np.stack(results)
    x_batch_torch = torch.from_numpy(x_batch_np)
    
    # Calculate stats: (batch, 4)
    stats = calculate_summary_statistics(x_batch_torch)
    
    if not is_batched:
        return stats.squeeze(0)
    return stats

# ============================================================================
# 2. Main
# ============================================================================

def run_snpe():
    print("Initializing SNPE for SIR...")
    
    # Prior
    prior = BoxUniform(
        low=torch.tensor(PRIOR_MIN).float(),
        high=torch.tensor(PRIOR_MAX).float()
    )
    
    # Inference Object
    inference = SNPE(prior=prior)
    
    # Generate Training Data
    NUM_SIMULATIONS = 2000
    print(f"Simulating {NUM_SIMULATIONS} samples...")
    
    theta, x = simulate_for_sbi(sbi_simulator, proposal=prior, num_simulations=NUM_SIMULATIONS)
    
    # Train
    print("Training Density Estimator...")
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    
    # Evaluate
    print("Evaluating on ground truth...")
    theta_true = torch.tensor(TRUE_PARAMS).float()
    
    # Simulate observation
    x_obs_stats = sbi_simulator(theta_true)
    
    # Sample posterior
    print("Sampling posterior...")
    samples = posterior.sample((1000,), x=x_obs_stats)
    
    # Plot
    os.makedirs("stochastic_sir/snpe_result", exist_ok=True)
    
    fig, axes = pairplot(
        samples,
        points=theta_true,
        labels=["beta", "gamma"],
        limits=[[PRIOR_MIN[0], PRIOR_MAX[0]], [PRIOR_MIN[1], PRIOR_MAX[1]]],
        figsize=(10, 10)
    )
    plt.savefig("stochastic_sir/snpe_result/posterior.png")
    print("Posterior plot saved to stochastic_sir/snpe_result/posterior.png")

if __name__ == "__main__":
    run_snpe()
