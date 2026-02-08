
import os

# Set Keras Backend to Torch BEFORE importing keras/bayesflow
os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.stats import multivariate_normal

# Local imports
from data_generation import GaussianTask, d, d_x, n
from load_models import load_bayesflow_model, load_torch_model
from utilities import refine_posterior

# ==========================================
# Hyperparameters
# ==========================================
LOAD_ROUND_ID = 1
MODELS_TO_TEST = ["bayesflow"]  # Options: "smmd", "mmd", "bayesflow"
N_SAMPLES = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_DIR = "results/transfer_test"

# ==========================================
# Transfer Task Definition
# ==========================================

class TransferGaussianTask(GaussianTask):
    """
    Gaussian Task with Transfer Prior:
    theta_0, theta_2, theta_3, theta_4 ~ Uniform[-3, 3]
    theta_1 ~ N(theta_0^2, 0.1^2)
    """
    def __init__(self, n=n):
        super().__init__(n=n, prior_type='transfer')
        # Expand bounds for theta 1 to accommodate theta_0^2 (up to 9)
        # We keep other bounds at [-3, 3]
        self.lower = np.array([-3.0, -2.0, -3.0, -3.0, -3.0]) 
        self.upper = np.array([3.0, 15.0, 3.0, 3.0, 3.0])

    def sample_prior(self, batch_size):
        # 1. Sample Uniform parameters
        # Initialize with zeros
        theta = np.zeros((batch_size, self.d), dtype=np.float32)
        
        # Indices 0, 2, 3, 4 are Uniform[-3, 3]
        for i in [0, 2, 3, 4]:
            theta[:, i] = np.random.uniform(-3, 3, batch_size)
            
        # 2. Sample theta_1 | theta_0
        # theta_1 ~ N(theta_0^2, 0.1^2)
        mean_1 = theta[:, 0]**2
        std_1 = 0.1
        theta[:, 1] = np.random.normal(mean_1, std_1)
        
        return theta

    def log_prior(self, theta):
        theta = np.asarray(theta)
        is_batch = theta.ndim > 1
        if not is_batch:
            theta = theta[np.newaxis, :]
            
        log_probs = np.zeros(theta.shape[0], dtype=np.float32)
        
        # 1. Check bounds for Uniform parameters
        mask_uniform = np.ones(theta.shape[0], dtype=bool)
        for i in [0, 2, 3, 4]:
            mask_uniform &= (theta[:, i] >= -3.0) & (theta[:, i] <= 3.0)
            
        log_probs[~mask_uniform] = -np.inf
        
        # 2. Add log prob for theta_1 (Normal)
        # Only for valid uniform samples
        valid = mask_uniform
        if np.any(valid):
            t0 = theta[valid, 0]
            t1 = theta[valid, 1]
            
            # Log PDF of Normal(mu=t0^2, sigma=0.1)
            # log p = -0.5 * log(2*pi*sigma^2) - 0.5 * ((x - mu)/sigma)^2
            sigma = 0.1
            log_const = -0.5 * np.log(2 * np.pi * sigma**2)
            resid = (t1 - t0**2) / sigma
            
            log_probs[valid] += log_const - 0.5 * resid**2
            
        if not is_batch:
            return log_probs[0]
        return log_probs

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

def run_pymc_transfer(obs_xs_2d, n_draws=2000, n_tune=2000, chains=20):
    """
    Run PyMC with the Transfer Prior.
    obs_xs_2d: (n, 2)
    """
    print(f"Setting up PyMC Transfer model...")
    
    with pm.Model() as model:
        # 1. Priors
        # t0, t2, t3, t4 ~ U[-3, 3]
        t0 = pm.Uniform("t0", lower=-3.0, upper=3.0)
        t2 = pm.Uniform("t2", lower=-3.0, upper=3.0)
        t3 = pm.Uniform("t3", lower=-3.0, upper=3.0)
        t4 = pm.Uniform("t4", lower=-3.0, upper=3.0)
        
        # t1 ~ N(t0^2, 0.1)
        t1 = pm.Normal("t1", mu=t0**2, sigma=0.1)
        
        # Stack into a single tensor for easier handling if needed, 
        # but we use individual variables for likelihood construction
        
        # 2. Transformations for Likelihood
        # s0 = theta[2]**2, s1 = theta[3]**2
        s0_real = t2**2
        s1_real = t3**2
        
        # r = tanh(theta[4])
        rho = pm.math.tanh(t4)
        
        # 3. Construct Covariance Matrix
        # Sigma = [[s0_real**2, rho * s0_real * s1_real],
        #          [rho * s0_real * s1_real, s1_real**2]]
        
        cov_00 = s0_real**2
        cov_01 = rho * s0_real * s1_real
        cov_11 = s1_real**2
        
        cov = pt.stack([
            pt.stack([cov_00, cov_01]),
            pt.stack([cov_01, cov_11])
        ])
        
        mu = pt.stack([t0, t1])
        
        # 4. Likelihood
        obs = pm.MvNormal("obs", mu=mu, cov=cov, observed=obs_xs_2d)
        
        # 5. Sampling
        print("Starting PyMC sampling (Transfer)...")
        trace = pm.sample(draws=n_draws, tune=n_tune, chains=chains, progressbar=True)
        
        # 6. Extract samples
        # Extract individual variables and stack
        post = trace.posterior
        t0_s = post['t0'].values.flatten()
        t1_s = post['t1'].values.flatten()
        t2_s = post['t2'].values.flatten()
        t3_s = post['t3'].values.flatten()
        t4_s = post['t4'].values.flatten()
        
        flat_samples = np.stack([t0_s, t1_s, t2_s, t3_s, t4_s], axis=1)
        
        print(f"PyMC Finished. Samples shape: {flat_samples.shape}")
        return flat_samples

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
    
    # 1. Generate Data with Transfer Task
    task = TransferGaussianTask(n=n)
    
    # Get Ground Truth (Fixed)
    # theta_true_batch = task.sample_prior(1)
    # theta_true = theta_true_batch[0]
    
    # Use fixed theta as requested
    theta_true = np.array([1.0, 1.0, -1.0, -0.9, 0.6], dtype=np.float32)
    theta_true_batch = theta_true[np.newaxis, :]
    
    # Generate observation
    x_obs_3d = task.simulator(theta_true_batch)[0] # (n, 3)
    
    print(f"Ground Truth Theta: {theta_true}")
    
    # 2. Get True Posterior (PyMC with Transfer Prior)
    # Convert to 2D for PyMC
    x_obs_2d = inverse_stereo_proj(x_obs_3d[np.newaxis, ...])[0] # (n, 2)
    
    true_samples = run_pymc_transfer(x_obs_2d, n_draws=N_SAMPLES)
    
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
            
            # --- Amortized Inference ---
            print("Sampling Amortized...")
            samples_amortized = None
            
            if model_name == "bayesflow":
                # BayesFlow Sampling
                if isinstance(x_obs_3d, torch.Tensor):
                     x_obs_cpu = x_obs_3d.detach().cpu().numpy()
                else:
                     x_obs_cpu = np.asarray(x_obs_3d)
                
                if x_obs_cpu.ndim == 2:
                    x_obs_cpu = x_obs_cpu[np.newaxis, ...]
                    
                conditions = {"summary_variables": x_obs_cpu}
                
                out = model.sample(conditions=conditions, num_samples=N_SAMPLES)
                if isinstance(out, dict):
                    samples_amortized = out["inference_variables"]
                else:
                    samples_amortized = out
                    
                samples_amortized = samples_amortized.reshape(-1, d)
                
            else:
                # PyTorch Models
                if isinstance(x_obs_3d, np.ndarray):
                    x_obs_torch = torch.from_numpy(x_obs_3d).float().to(DEVICE)
                else:
                    x_obs_torch = x_obs_3d.to(DEVICE)
                
                if x_obs_torch.ndim == 2:
                    x_obs_torch = x_obs_torch.unsqueeze(0)
                    
                samples_amortized = model.sample_posterior(x_obs_torch, N_SAMPLES)
                samples_amortized = samples_amortized.detach().cpu().numpy().reshape(-1, d)
            
            model_results[f"{model_name}_amortized"] = samples_amortized
            
            # --- Refinement with Transfer Prior ---
            print(f"Refining {model_name} with Transfer Prior...")
            
            # Use the TransferGaussianTask instance 'task' which has the correct log_prior
            refined_samples = refine_posterior(
                model, 
                x_obs_3d, 
                task, # Passing TransferGaussianTask
                n_chains=1000,
                n_samples=1,
                burn_in=99,
                thin=1,
                nsims=50,
                device=str(DEVICE)
            )
            
            refined_samples_flat = refined_samples.reshape(-1, d)
            model_results[f"{model_name}_refined"] = refined_samples_flat
            
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
