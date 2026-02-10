"""
Test Script for Transfer and Generalization Capability of SMMD/MMD Models.
Independent of run_experiment.py.
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Ensure we can import local modules
sys.path.append(os.getcwd())

from data_generation import LVTask
from models.smmd import SMMD_Model
from models.mmd import MMD_Model
from utilities import compute_metrics

# Configuration
CONFIG_PATH = "config.json"
RESULTS_DIR = "results/transfer_test"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: config.json not found. Using defaults.")
        return {}

CONFIG = load_config()
N_TIME_STEPS = CONFIG.get("n_time_steps", 151)
DT = CONFIG.get("dt", 0.2)
d = 4
d_x = 2

def load_model(model_type, round_id, model_path, summary_dim=10):
    """Load a trained model."""
    print(f"Loading {model_type} from {model_path}...")
    
    if model_type == "smmd":
        model = SMMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=N_TIME_STEPS)
    elif model_type == "mmd":
        model = MMD_Model(summary_dim=summary_dim, d=d, d_x=d_x, n=N_TIME_STEPS)
    else:
        raise ValueError(f"Unsupported model type for this test: {model_type}")
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None

def evaluate_on_test_set(model, task, test_size=100, n_samples_posterior=1000):
    """Evaluate model on a new test set (Generalization)."""
    print(f"Evaluating on test set (Size: {test_size})...")
    
    # Generate Test Set
    theta_test = task.sample_prior(test_size, "vague")
    x_test = task.simulator(theta_test)
    
    metrics_list = []
    
    for i in range(test_size):
        x_obs = x_test[i]
        theta_true = theta_test[i]
        
        # Sample Posterior
        samples = model.sample_posterior(x_obs, n_samples_posterior)
        
        # Compute Metrics
        m = compute_metrics(samples, theta_true)
        metrics_list.append(m)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{test_size} test samples")
            
    # Aggregate Metrics
    keys = metrics_list[0].keys()
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in keys}
    
    return avg_metrics, theta_test, metrics_list

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Parameters
    models_to_test = ["smmd", "mmd"] # Add others if needed
    round_ids = [1] # Which rounds to test
    test_size = 50 # Number of test samples
    
    # Task
    t_max = (N_TIME_STEPS - 1) * DT
    task = LVTask(t_max=t_max, dt=DT)
    
    results = []
    
    for model_type in models_to_test:
        for round_id in round_ids:
            print(f"\nTesting {model_type} - Round {round_id}")
            
            # 1. Test Initial Model
            initial_path = f"results/models/{model_type}/round_{round_id}/initial_state_dict.pt"
            model_initial = load_model(model_type, round_id, initial_path)
            
            if model_initial:
                metrics_init, _, _ = evaluate_on_test_set(model_initial, task, test_size=test_size)
                metrics_init["model"] = model_type
                metrics_init["round"] = round_id
                metrics_init["stage"] = "initial"
                results.append(metrics_init)
                print(f"Initial Model Bias L2: {metrics_init['bias_l2']:.4f}")
            
            # 2. Test Sequential Learning (Refined+) Model
            sl_path = f"results/models/{model_type}/round_{round_id}/sl_state_dict.pt"
            model_sl = load_model(model_type, round_id, sl_path)
            
            if model_sl:
                metrics_sl, _, _ = evaluate_on_test_set(model_sl, task, test_size=test_size)
                metrics_sl["model"] = model_type
                metrics_sl["round"] = round_id
                metrics_sl["stage"] = "sequential_learning"
                results.append(metrics_sl)
                print(f"SL Model Bias L2: {metrics_sl['bias_l2']:.4f}")
                
    # Save Results
    if results:
        df = pd.DataFrame(results)
        save_path = f"{RESULTS_DIR}/transfer_generalization_results.csv"
        df.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")
        print(df[["model", "stage", "bias_l2", "hdi_length", "coverage"]])
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
