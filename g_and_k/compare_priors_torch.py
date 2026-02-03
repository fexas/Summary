"""
Script to compare Posterior estimates under different Priors using PyTorch SMMD.
"""

import os
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_generation import generate_dataset, get_ground_truth, PRIOR_CONFIGS
from smmd_torch import train_smmd_torch, evaluate_posterior_torch, d

def main():
    result_dir = "smmd_result_torch_comparison"
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Get Ground Truth (Fixed)
    theta_true, x_obs = get_ground_truth()
    x_obs_expanded = x_obs[np.newaxis, ...] # (1, n, 1)
    
    print(f"True Params: {theta_true}")
    
    # 2. Loop over priors
    priors_to_test = ['vague', 'weak_informative', 'informative']
    
    # DataFrame to store all samples for plotting
    all_samples_df = pd.DataFrame()
    
    for prior_type in priors_to_test:
        print(f"\n==========================================")
        print(f"Running Experiment with {prior_type} Prior")
        print(f"==========================================")
        
        # A. Generate Data
        theta_train, x_train = generate_dataset(prior_type)
        
        # B. Train Model
        # We use a separate sub-folder for individual run artifacts if needed, 
        # or just save to the main dir with prefixes (which train_smmd_torch handles)
        model = train_smmd_torch(theta_train, x_train, prior_type, result_dir)
        
        # C. Get Posterior Samples
        # evaluate_posterior_torch returns numpy array (N, d)
        samples = evaluate_posterior_torch(model, x_obs_expanded, theta_true, prior_type, result_dir)
        
        # D. Add to DataFrame
        param_names = ['A', 'B', 'g', 'k']
        df_temp = pd.DataFrame(samples, columns=param_names)
        df_temp['Prior'] = prior_type
        
        all_samples_df = pd.concat([all_samples_df, df_temp], ignore_index=True)

    # 3. Plot Combined Results
    print("\nGenerating Combined Comparison Plot...")
    
    # Use Seaborn PairGrid/pairplot with hue
    # corner=True to show lower triangle
    g = sns.pairplot(all_samples_df, hue='Prior', kind='kde', corner=True, 
                     plot_kws={'fill': False, 'linewidth': 1.5},
                     diag_kws={'fill': True, 'linewidth': 1.5})
    
    # Add True Parameter Markers
    # pairplot creates a grid of axes. We need to iterate and add lines/stars.
    # g.axes is a 2D array of axes objects (some are None because of corner=True)
    
    param_names = ['A', 'B', 'g', 'k']
    
    # Iterate over the grid
    for i in range(d): # Row
        for j in range(i + 1): # Column (lower triangle)
            ax = g.axes[i, j]
            if ax is not None:
                if i == j:
                    # Diagonal: Add vertical line
                    ax.axvline(theta_true[i], color='red', linestyle='--', linewidth=2, label='Truth' if i==0 else "")
                else:
                    # Off-diagonal: Add star marker
                    ax.scatter(theta_true[j], theta_true[i], color='red', marker='*', s=150, zorder=10, label='Truth' if i==1 and j==0 else "")

    # Adjust legend? pairplot handles hue legend automatically.
    # We might want to add 'Truth' to legend, but it's tricky with pairplot.
    # Usually the red dashed line is self-explanatory or we add a custom legend entry.
    
    g.fig.suptitle("Posterior Comparison across Priors (PyTorch SMMD)", y=1.02)
    
    save_path = os.path.join(result_dir, "posterior_comparison_combined.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    main()
