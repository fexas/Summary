"""
SNPE for g-and-k distribution using sbi library.
Uses handcrafted summary statistics (octiles).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Hack: Set HOME to /tmp to fool Arviz into writing there
os.environ["HOME"] = "/tmp"
os.environ["ARVIZ_DATA"] = "/tmp/arviz_data"
try:
    os.makedirs(os.environ["ARVIZ_DATA"], exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create ARVIZ_DATA dir: {e}")

print(f"ARVIZ_DATA set to: {os.environ['ARVIZ_DATA']}")
print(f"HOME set to: {os.environ['HOME']}")

# Add parent directory to path to import G_and_K modules
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from data_generation import simulator as gk_simulator
    from data_generation import PRIOR_CONFIGS, TRUE_PARAMS, n
except ImportError:
    # If running from parent directory or elsewhere
    sys.path.append(os.path.dirname(__file__))
    from data_generation import simulator as gk_simulator
    from data_generation import PRIOR_CONFIGS, TRUE_PARAMS, n

from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.analysis import pairplot

# ============================================================================
# 1. Configuration
# ============================================================================

NUM_ROUNDS = 2
NUM_SIMS = 2000  # Simulations per round
PRIOR_TYPE = 'weak_informative'

# ============================================================================
# 2. Summary Statistics
# ============================================================================

def calculate_summary_statistics(x):
    """
    Calculate summary statistics for g-and-k samples.
    x: (batch_size, n_samples, 1) or (n_samples, 1)
    Returns: (batch_size, n_summary)
    
    Uses Octiles (quantiles at 12.5%, 25%, ..., 87.5%)
    """
    # Ensure x is torch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        
    # Handle shapes
    if x.ndim == 2:
        # (n_samples, 1) -> (1, n_samples, 1)
        x = x.unsqueeze(0)
    
    # x is (batch, n, 1) -> squeeze to (batch, n)
    x = x.squeeze(-1)
    
    # Calculate octiles: 12.5, 25, 37.5, 50, 62.5, 75, 87.5
    probs = torch.tensor([0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    
    # torch.quantile requires (input, q, dim)
    # x: (batch, n)
    # We want quantiles along dim 1
    # output: (n_probs, batch) -> transpose to (batch, n_probs)
    
    # Note: quantile might not support batching in older torch versions, 
    # but usually does. 
    quantiles = torch.quantile(x, probs, dim=1).T
    
    return quantiles

# ============================================================================
# 3. Simulator Wrapper
# ============================================================================

def sbi_simulator(theta):
    """
    Wrapper for g-and-k simulator compatible with sbi.
    theta: (batch_size, 4) or (4,)
    Returns: (batch_size, n_summary)
    """
    # Convert torch to numpy
    theta_np = theta.numpy()
    
    # Call original simulator
    # gk_simulator handles batching
    x_raw = gk_simulator(theta_np, n_samples=n)
    
    # Calculate summary stats
    # x_raw is (batch, n, 1)
    stats = calculate_summary_statistics(x_raw)
    
    return stats

# ============================================================================
# 4. Main SNPE Workflow
# ============================================================================

def run_snpe():
    print(f"Running SNPE on g-and-k with {PRIOR_TYPE} prior...")
    
    # 1. Define Prior
    config = PRIOR_CONFIGS[PRIOR_TYPE]
    low = torch.tensor([config['A'][0], config['B'][0], config['g'][0], config['k'][0]])
    high = torch.tensor([config['A'][1], config['B'][1], config['g'][1], config['k'][1]])
    
    prior = BoxUniform(low=low, high=high)
    
    # 2. Define Ground Truth Observation (x_o)
    # Generate one observation with true parameters
    print("Generating observed data (x_o)...")
    true_theta = torch.tensor(TRUE_PARAMS, dtype=torch.float32)
    # Expand for simulator: (1, 4)
    x_o_raw = gk_simulator(true_theta.unsqueeze(0).numpy(), n_samples=n)
    x_o = calculate_summary_statistics(x_o_raw)
    # x_o should be (1, n_summary) or (n_summary,)
    # sbi expects (n_summary,) or (1, n_summary)
    
    print(f"True Params: {TRUE_PARAMS}")
    print(f"Observed Stats: {x_o.squeeze().numpy()}")
    
    # 3. Setup Inference
    inference = NPE(prior=prior)
    
    proposal = prior
    
    # 4. Sequential Training Loop
    for r in range(NUM_ROUNDS):
        print(f"\n--- Round {r+1}/{NUM_ROUNDS} ---")
        
        # Draw samples from proposal
        theta = proposal.sample((NUM_SIMS,))
        
        # Simulate
        x = sbi_simulator(theta)
        
        # Train
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        
        # Build Posterior
        posterior = inference.build_posterior(density_estimator)
        
        # Update proposal (only if not last round)
        if r < NUM_ROUNDS - 1:
            proposal = posterior.set_default_x(x_o)
    
    # 5. Evaluate
    print("\nSampling from final posterior...")
    posterior_samples = posterior.sample((10000,), x=x_o)
    
    # 6. Plotting
    print("Plotting results...")
    
    # Define limits for plotting based on prior
    limits = [
        [config['A'][0], config['A'][1]],
        [config['B'][0], config['B'][1]],
        [config['g'][0], config['g'][1]],
        [config['k'][0], config['k'][1]]
    ]
    
    fig, axes = pairplot(
        posterior_samples,
        limits=limits,
        ticks=limits,
        figsize=(10, 10),
        labels=['A', 'B', 'g', 'k'],
        points=true_theta,
        points_colors='red',
        points_offdiag={'markersize': 6},
        title=f"SNPE Posterior (g-and-k, {PRIOR_TYPE})"
    )
    
    result_dir = os.path.join(os.path.dirname(__file__), 'snpe_result')
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, f"snpe_posterior_{PRIOR_TYPE}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    run_snpe()
