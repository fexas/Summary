"""
Script to compare Posterior estimates under different Summary Statistics Dimensions (p).
Uses PyTorch SMMD with weak_informative prior.
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
    result_dir = "smmd_result_dims_comparison"
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Get Ground Truth (Fixed)
    theta_true, x_obs = get_ground_truth()
    x_obs_expanded = x_obs[np.newaxis, ...] # (1, n, 1)
    
    print(f"True Params: {theta_true}")
    
    # 2. Fixed Data Generation (Weak Informative Prior)
    prior_type = 'weak_informative'
    print(f"\nGenerating fixed Reference Table with {prior_type} prior...")
    theta_train, x_train = generate_dataset(prior_type)
    
    # 3. Loop over Summary Statistic Dimensions (p)
    p_values = [5, 10, 15, 20, 25]
    
    # DataFrame to store all samples for plotting
    all_samples_df = pd.DataFrame()
    
    for p_val in p_values:
        print(f"\n==========================================")
        print(f"Running Experiment with p = {p_val}")
        print(f"==========================================")
        
        # Train Model with specific p
        # We pass a dummy result_dir to avoid overwriting main plots inside train/eval functions if they share names
        # But train_smmd_torch appends names, so it's okay.
        model = train_smmd_torch(theta_train, x_train, prior_type, result_dir, summary_dim=p_val)
        
        # Get Posterior Samples
        # Note: evaluate_posterior_torch uses the model's T and G, which are correctly sized.
        # It generates a plot internally, which might get overwritten or saved with same name if prior_name is same.
        # To avoid confusion in saved files, we can pass a modified prior name for the plot saving part.
        unique_name = f"{prior_type}_p{p_val}"
        samples = evaluate_posterior_torch(model, x_obs_expanded, theta_true, unique_name, result_dir)
        
        # Add to DataFrame
        param_names = ['A', 'B', 'g', 'k']
        df_temp = pd.DataFrame(samples, columns=param_names)
        df_temp['Dimension'] = f"p={p_val}"
        
        all_samples_df = pd.concat([all_samples_df, df_temp], ignore_index=True)

    # 4. Plot Combined Results
    print("\nGenerating Combined Comparison Plot (Curse of Dimensionality Check)...")
    
    # Use Seaborn pairplot with hue
    g = sns.pairplot(all_samples_df, hue='Dimension', kind='kde', corner=True, 
                     plot_kws={'fill': False, 'linewidth': 1.5},
                     diag_kws={'fill': True, 'linewidth': 1.5},
                     palette="viridis") # distinct colors
    
    # Add True Parameter Markers
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

    g.fig.suptitle("Posterior Comparison across Summary Stats Dimensions (p)", y=1.02)
    
    save_path = os.path.join(result_dir, "posterior_dims_comparison.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    main()
