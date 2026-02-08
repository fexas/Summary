
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

# Local imports
from data_generation import GaussianTask, d, d_x, n
from load_models import load_bayesflow_model, load_torch_model
# from utilities import refine_posterior # Optional: if we want to test refinement

# ==========================================
# Hyperparameters
# ==========================================
LOAD_ROUND_ID = 1
MODELS_TO_TEST = ["bayesflow"]  # Options: "smmd", "mmd", "bayesflow"
N_SAMPLES = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = "results/transfer_test"

# ==========================================
# Helper Functions
# ==========================================

def inverse_stereo_proj(x_3d):
    """
    Invert stereographic projection: 3D -> 2D.
    x_3d: (batch, n, 3)
    """
    X = x_3d[..., 0]
    Y = x_3d[..., 1]
    Z = x_3d[..., 2]
    
    # Avoid division by zero
    denom = 1.0 - Z
    denom[np.abs(denom) < 1e-6] = 1e-6
    
    x_2d = X / denom
    y_2d = Y / denom
    
    return np.stack([x_2d, y_2d], axis=-1)

def exact_log_likelihood(theta, x_2d):
    """
    Compute exact log likelihood of 2D data under Gaussian Model.
    theta: (d,)
    x_2d: (n, 2)
    """
    m0, m1 = theta[0], theta[1]
    # theta[2] and theta[3] are squared to get variance in simulator?
    # Based on data_generation.py: xs[..., 0] = s0 * us[..., 0] + m0 where s0 = theta[:, 2:3] ** 2
    # So the standard deviation is theta[2]**2
    std0 = theta[2]**2
    std1 = theta[3]**2
    r_param = theta[4]
    corr = np.tanh(r_param)
    
    cov = np.array([
        [std0**2, std0*std1*corr],
        [std0*std1*corr, std1**2]
    ])
    
    mean = np.array([m0, m1])
    
    try:
        log_prob = multivariate_normal.logpdf(x_2d, mean=mean, cov=cov)
        return np.sum(log_prob)
    except np.linalg.LinAlgError:
        return -np.inf

def run_true_mcmc(x_obs, task, n_samples=2000, burn_in=500, thin=1):
    """
    Run MCMC to get True Posterior samples using Exact Likelihood.
    """
    print("Running MCMC for True Posterior...")
    x_2d = inverse_stereo_proj(x_obs)[0] # (n, 2)
    
    current_theta = task.sample_prior(1).flatten() # Start from prior
    
    samples = []
    accepted = 0
    
    # Adaptive proposal
    proposal_cov = np.eye(task.d) * 0.1
    
    for i in range(n_samples * thin + burn_in):
        # Propose new theta
        proposal = np.random.multivariate_normal(current_theta, proposal_cov)
        
        # Check prior support
        if np.any(np.abs(proposal) > 3.0): # Assuming prior is roughly uniform/Gaussian within bounds, simplified check
             # Actually prior is Uniform[-3, 3] usually in this task, or Gaussian(0,1). 
             # data_generation.py: return np.random.uniform(-3, 3, (batch_size, self.d))
             prior_ratio = 0 # Out of bounds
        else:
             prior_ratio = 1 # Uniform prior
        
        if prior_ratio == 0:
            accept_prob = 0
        else:
            log_lik_curr = exact_log_likelihood(current_theta, x_2d)
            log_lik_prop = exact_log_likelihood(proposal, x_2d)
            
            if log_lik_prop == -np.inf:
                accept_prob = 0
            else:
                accept_prob = np.exp(log_lik_prop - log_lik_curr)
        
        if np.random.rand() < accept_prob:
            current_theta = proposal
            if i >= burn_in:
                accepted += 1
        
        if i >= burn_in and (i - burn_in) % thin == 0:
            samples.append(current_theta)
            
    print(f"MCMC Finished. Acceptance Rate: {accepted / (n_samples * thin):.2f}")
    return np.array(samples)

def plot_comparison(true_samples, model_samples_dict, theta_true, save_path):
    """
    Plot marginals comparing True Posterior vs Model Posteriors.
    """
    n_params = true_samples.shape[1]
    cols = n_params
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    
    # Plot True Posterior
    df_true = np.array(true_samples)
    
    for i in range(cols):
        ax = axes[i]
        
        # True Posterior
        sns.kdeplot(df_true[:, i], ax=ax, fill=True, color='gray', alpha=0.3, label='True Posterior')
        
        # Model Posteriors
        colors = ['blue', 'green', 'orange', 'purple']
        for idx, (name, samples) in enumerate(model_samples_dict.items()):
            sns.kdeplot(samples[:, i], ax=ax, color=colors[idx % len(colors)], label=name)
            
        # True Parameter
        ax.axvline(theta_true[i], color='red', linestyle='--', label='True Param')
        
        ax.set_title(f"Theta {i+1}")
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to {save_path}")

# ==========================================
# Main Test Function
# ==========================================

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    print(f"=== Starting Transfer Test (Round {LOAD_ROUND_ID}) ===")
    
    # 1. Generate Data
    task = GaussianTask(n=n)
    theta_true, x_obs = task.get_ground_truth()
    
    # x_obs is (1, n, d_x)
    print(f"Ground Truth Theta: {theta_true}")
    
    # 2. Get True Posterior (MCMC)
    true_samples = run_true_mcmc(x_obs, task, n_samples=N_SAMPLES)
    
    model_results = {}
    
    # 3. Test Models
    for model_name in MODELS_TO_TEST:
        print(f"\n--- Testing {model_name.upper()} ---")
        try:
            # Load Model
            if model_name == "bayesflow":
                model = load_bayesflow_model(LOAD_ROUND_ID)
            else:
                model = load_torch_model(model_name, LOAD_ROUND_ID, device=DEVICE)
            
            # Inference
            print("Sampling...")
            samples = None
            
            if model_name == "bayesflow":
                # BayesFlow Sampling
                # Prepare conditions
                if isinstance(x_obs, torch.Tensor):
                     x_obs_cpu = x_obs.detach().cpu().numpy()
                else:
                     x_obs_cpu = np.asarray(x_obs)
                
                # Ensure batch dim
                if x_obs_cpu.ndim == 2:
                    x_obs_cpu = x_obs_cpu[np.newaxis, ...]
                    
                conditions = {"summary_variables": x_obs_cpu}
                
                # Sample
                out = model.sample(conditions=conditions, num_samples=N_SAMPLES)
                if isinstance(out, dict):
                    samples = out["inference_variables"]
                else:
                    samples = out
                    
                samples = samples.reshape(-1, d)
                
            else:
                # PyTorch Models (SMMD, MMD)
                # Ensure torch tensor
                if isinstance(x_obs, np.ndarray):
                    x_obs_torch = torch.from_numpy(x_obs).float().to(DEVICE)
                else:
                    x_obs_torch = x_obs.to(DEVICE)
                
                if x_obs_torch.ndim == 2:
                    x_obs_torch = x_obs_torch.unsqueeze(0)
                    
                samples = model.sample_posterior(x_obs_torch, N_SAMPLES)
                samples = samples.detach().cpu().numpy().reshape(-1, d)
            
            model_results[model_name] = samples
            print(f"Got {samples.shape[0]} samples from {model_name}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
    # 4. Plot Comparison
    if model_results:
        save_path = os.path.join(RESULT_DIR, f"comparison_round_{LOAD_ROUND_ID}.png")
        plot_comparison(true_samples, model_results, theta_true, save_path)
    else:
        print("No model results to plot.")

if __name__ == "__main__":
    main()
