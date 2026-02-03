"""
Test script for PyTorch SMMD implementation.
"""

import os
import numpy as np
import torch
from data_generation import generate_dataset, get_ground_truth
from smmd_torch import train_smmd_torch, evaluate_posterior_torch

def main():
    result_dir = "smmd_result_torch"
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Get Ground Truth
    theta_true, x_obs = get_ground_truth()
    x_obs = x_obs[np.newaxis, ...] # (1, n, 1)
    
    print(f"True Params: {theta_true}")
    
    # 2. Run with weak_informative prior
    prior_type = 'weak_informative'
    
    # Generate Data
    theta_train, x_train = generate_dataset(prior_type)
    
    # Train
    model = train_smmd_torch(theta_train, x_train, prior_type, result_dir)
    
    # Evaluate
    evaluate_posterior_torch(model, x_obs, theta_true, prior_type, result_dir)
    
    print(f"Results saved to {result_dir}")

if __name__ == "__main__":
    main()
