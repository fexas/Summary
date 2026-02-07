
import os
import numpy as np
import torch
from Gaussian.data_generation import GaussianTask, generate_dataset, get_ground_truth
from Gaussian.smmd_torch import train_smmd_torch, refine_posterior, evaluate_posterior_torch

def test_refinement():
    print("Testing SMMD Refinement...")
    
    # 1. Setup
    task = GaussianTask(prior_type='uniform')
    
    # 2. Generate small data
    theta_train, x_train = generate_dataset(task=task, n_sims=100)
    
    # 3. Train Model (Briefly)
    result_dir = "test_results"
    os.makedirs(result_dir, exist_ok=True)
    
    # Mocking EPOCHS for speed
    import Gaussian.smmd_torch as smmd_module
    original_epochs = smmd_module.EPOCHS
    smmd_module.EPOCHS = 2
    
    model = train_smmd_torch(theta_train, x_train, result_dir)
    
    # Restore
    smmd_module.EPOCHS = original_epochs
    
    # 4. Get Ground Truth
    true_params, obs_2d, obs_3d = get_ground_truth(task=task)
    
    # 5. Run Refinement
    # Small parameters for test
    # Test new logic: 10 chains, 1 sample each, 10 burn-in
    n_chains = 10
    n_samples = 1
    posterior_samples = refine_posterior(
        model, obs_3d, task,
        n_chains=n_chains, 
        n_samples=n_samples, 
        burn_in=10, 
        thin=1, 
        nsims=5, 
        epsilon=None
    )
    
    print("Posterior Samples Shape:", posterior_samples.shape)
    assert posterior_samples.shape == (n_samples * n_chains, 5) # (n_samples * n_chains, d)
    
    print("Test passed!")

if __name__ == "__main__":
    test_refinement()
