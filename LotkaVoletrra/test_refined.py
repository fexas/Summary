import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from data_generation import LVTask
from run_experiment import run_single_experiment

def run_test(model_type):
    print(f"\n=== Testing Refined+ for {model_type} ===")
    
    # Setup Small Experiment
    dataset_size = 5000  # Increased for better training
    batch_size = 128
    epochs = 500 
    
    task = LVTask(t_max=30.0, dt=0.2)
    
    # 1. Generate Train Data
    print("Generating train data...")
    theta_train = task.sample_prior(dataset_size, "vague")
    x_train = task.simulator(theta_train)
    
    dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(theta_train).float())
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Ground Truth
    theta_true, x_obs = task.get_ground_truth()
    
    # 3. Config
    # Reduced MCMC settings for testing: 10 burn-in steps
    model_config = {"epochs": epochs, "mcmc_burn_in": 10}
    
    # 4. Run
    # Pass 10 as n_samples (which becomes n_chains in run_single_experiment)
    # This ensures we only run 10 chains * 10 steps = 100 simulations total (very fast)
    try:
        metrics = run_single_experiment(model_type, task, train_loader, theta_true, x_obs, 10, model_config)
        print(f"Success! Metrics: {metrics}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test SMMD
    run_test("smmd")
