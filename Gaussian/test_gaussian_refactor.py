
try:
    from data_generation import GaussianTask, generate_dataset, run_mcmc, get_ground_truth
except ImportError:
    import sys
    sys.path.append('.')
    from data_generation import GaussianTask, generate_dataset, run_mcmc, get_ground_truth
import numpy as np
import os

# Mock ArviZ if not present or just to speed up
import sys

def test_pipeline():
    print("Testing GaussianTask...")
    task = GaussianTask(prior_type='uniform')
    
    print("Testing generate_dataset...")
    theta, x = generate_dataset(task, n_sims=10)
    print(f"Dataset shapes: theta={theta.shape}, x={x.shape}")
    
    print("Testing get_ground_truth...")
    true_params, obs_2d, obs_3d = get_ground_truth(task)
    print(f"Obs shapes: 2d={obs_2d.shape}, 3d={obs_3d.shape}")
    
    print("Testing MCMC (short run)...")
    # Using very short chain to verify logic
    samples = run_mcmc(task, obs_2d, n_chains=2, n_samples=20, burn_in=10)
    print(f"MCMC samples shape: {samples.shape}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_pipeline()
