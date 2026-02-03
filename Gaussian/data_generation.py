"""
This script handles data generation for a Toy Example in Simulation-Based Inference.
Refactored for simplicity and flexibility.
"""
import os
import time
import scipy.stats
import numpy as np
import tensorflow as tf
import bayesflow as bf
import math
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances

# Set matplotlib backend to Agg to avoid display issues
plt.switch_backend('Agg')

# ============================================================================
# 1. Global Hyperparameters & Configuration
# ============================================================================
# Data parameters
N = 12800  # Size of the training dataset
n = 50     # Number of samples per observation (sample size)
d = 5      # Dimension of parameter theta
d_x = 2    # Dimension of x (changed from 3 to 2 as stereo_proj is removed)
p = 10     # Dimension of summary statistics

# Create data storage folder
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Prior Configuration
# User can switch between 'vague', 'weak_informative', 'informative'
PRIOR_TYPE = 'weak_informative'

PRIOR_CONFIGS = {
    'vague': {
        'bounds_limit': 9.0,    # Uniform [-10, 10]
        'cond_noise_std': 1.0,   # Noise for theta[1] dependency
        'mcmc_step': 0.09,
    },
    'weak_informative': {
        'bounds_limit': 6.0,     # Uniform [-3, 3]
        'cond_noise_std': 0.5,   # Original setting
        'mcmc_step': 0.05,
    },
    'informative': {
        'bounds_limit': 3.0,     # Uniform [-1, 1]
        'cond_noise_std': 0.1,  
        'mcmc_step': 0.05,
    }
}

CURRENT_PRIOR = PRIOR_CONFIGS[PRIOR_TYPE]

# ============================================================================
# 2. Simulator & Prior Functions
# ============================================================================

def prior_generator(d=5):
    """
    Generate prior samples based on CURRENT_PRIOR configuration.
    """
    limit = CURRENT_PRIOR['bounds_limit']
    noise_std = CURRENT_PRIOR['cond_noise_std']
    
    # Uniform sampling for all dimensions first
    lower = np.array([-limit] * d, dtype=float)
    upper = np.array([+limit] * d, dtype=float)
    
    u = np.random.rand(d)
    theta = (upper - lower) * u + lower
    
    # Apply specific dependency structure for this Toy Example
    # theta[1] depends on theta[0]
    a = np.random.randn(1)
    theta[1] = theta[0] ** 2 + a[0] * noise_std
    
    return theta

def unpack_params(ps):
    """
    Unpack parameters ps into m0, m1, s0, s1, r.
    ps shape: (batch_size, 5)
    """
    # Ensure ps is at least 2D
    if ps.ndim == 1:
        ps = ps[np.newaxis, :]
        
    m0 = ps[:, [0]]
    m1 = ps[:, [1]]
    s0 = ps[:, [2]] ** 2
    s1 = ps[:, [3]] ** 2
    r = np.tanh(ps[:, [4]])
    
    return m0, m1, s0, s1, r

def simulator(ps, n_samples=n, rng=np.random):
    """
    Simulate data given parameters ps.
    Returns data of shape (batch_size, n_samples, 2).
    """
    ps = np.asarray(ps, dtype=float)
    if ps.ndim == 1:
        ps = ps[np.newaxis, :]
    
    n_sims = ps.shape[0]
    m0, m1, s0, s1, r = unpack_params(ps)
    
    # Generate standard normal noise
    us = rng.randn(n_sims, n_samples, 2)
    xs = np.empty_like(us)
    
    # Apply transformation
    xs[:, :, 0] = s0 * us[:, :, 0] + m0
    xs[:, :, 1] = s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r**2) * us[:, :, 1]) + m1
    
    # If input was 1D, return 2D array (n_samples, 2)
    if n_sims == 1:
        return xs[0]
        
    return xs

def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed data.
    """
    # Fixed ground truth parameters
    est_ps = np.array([1, 1, -1.0, -0.9, 0.6])
    
    rng = np.random.RandomState()
    # Generate one set of observation
    obs_data = simulator(est_ps, n_samples=n, rng=rng)
    
    return est_ps, obs_data

# ============================================================================
# 3. MCMC Components
# ============================================================================

def log_posterior(theta, obs_xs):
    """
    Calculate log posterior for parameter theta given observed data obs_xs.
    Uses CURRENT_PRIOR for boundaries and prior probability.
    """
    limit = CURRENT_PRIOR['bounds_limit']
    noise_std = CURRENT_PRIOR['cond_noise_std']
    
    # Check bounds
    if (abs(theta[0]) >= limit or 
        abs(theta[2]) >= limit or 
        abs(theta[3]) >= limit or 
        abs(theta[4]) >= limit):
        return -np.inf
    
    # Likelihood Calculation
    u = [theta[0], theta[1]]
    s1 = theta[2] ** 2
    s2 = theta[3] ** 2
    rho = math.tanh(theta[4])

    Sigma = [
        [s1**2, rho * s1 * s2],
        [rho * s1 * s2, s2**2],
    ]

    try:
        IS = np.linalg.inv(Sigma)
        det_Sigma = np.linalg.det(Sigma)
        if det_Sigma <= 0:
            return -np.inf
            
        # obs_xs shape is (n, 2)
        diff = obs_xs - u
        # vectorized quadratic form: sum((x-u)^T * InvSigma * (x-u))
        quad_form = np.sum([np.dot(np.dot(d, IS), d.T) for d in diff])
        
        log_likelihood = -0.5 * quad_form - 0.5 * n * np.log(det_Sigma)
        
        # Prior Calculation (matching the generative process)
        # theta[1] ~ N(theta[0]^2, noise_std^2)
        log_prior = -((theta[1] - theta[0] ** 2) ** 2) / (2 * noise_std**2)
        
        return log_likelihood + log_prior
        
    except np.linalg.LinAlgError:
        return -np.inf

def log_posterior_array(theta, obs_xs):
    """
    Calculate log posterior for an array of parameters.
    """
    log_posteriors = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        log_posteriors[i] = log_posterior(theta[i], obs_xs)
    return log_posteriors

def generate_initial_proposal_mcmc(N_proposal):
    """
    Generate initial proposal samples for MCMC from the prior.
    """
    # Use BayesFlow's Prior wrapper for consistency or just call prior_generator
    # Here we just generate directly since we simplified
    samples = np.array([prior_generator(d) for _ in range(N_proposal)])
    return samples

def mcmc(obs_xs, N_proposal, burn_in_steps, step_size=None):
    """
    Run N_proposal MCMC chains simultaneously.
    """
    Theta_seq = []
    accp = 0
    
    # Use provided step_size or default from configuration
    if step_size is None:
        h = CURRENT_PRIOR['mcmc_step']
    else:
        h = step_size
        
    print(f"MCMC Step Size (h): {h}")

    Theta_proposal = generate_initial_proposal_mcmc(N_proposal)
    log_posterior_0 = log_posterior_array(Theta_proposal, obs_xs)

    for mcmc_step in tqdm(range(burn_in_steps + 1), desc="MCMC Sampling"):
        Theta_new_proposal = np.random.normal(
            loc=Theta_proposal, scale=h, size=(N_proposal, d)
        )
        log_posterior_1 = log_posterior_array(Theta_new_proposal, obs_xs)
        log_ratio = log_posterior_1 - log_posterior_0
        u = np.log(np.random.uniform(size=N_proposal))
        accept = u <= log_ratio

        Theta_proposal[accept] = Theta_new_proposal[accept]
        log_posterior_0[accept] = log_posterior_1[accept]
        accp += np.sum(accept)

        Theta_seq.append(Theta_proposal.copy())

    # Return the last step samples (or concatenated if needed, but original returned one slice)
    # Original: tf.concat(Theta_seq[burn_in_steps: burn_in_steps + 1], axis=0) which is just the last one
    Theta_mcmc = tf.constant(Theta_seq[-1], dtype=tf.float32)
    
    accp_rate = accp / (N_proposal * (burn_in_steps + 1))
    print(f"Acceptance rate: {accp_rate:.4f}")

    return Theta_mcmc, accp

def plot_posterior(ps_samples, true_params, iteration):
    """
    Plot the posterior density (pairplot) and mark the true parameters.
    """
    param_names = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$", r"$\theta_5$"]
    
    # Convert samples to DataFrame for Seaborn
    df = pd.DataFrame(ps_samples, columns=param_names)
    
    # Create PairGrid
    g = sns.PairGrid(df, diag_sharey=False, corner=True)
    g.map_lower(sns.kdeplot, fill=True, levels=5, cmap="Blues")
    g.map_diag(sns.histplot, kde=True, color="blue", alpha=0.3)
    
    # Overlay True Parameters
    # Loop through axes to plot the red star
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                # Diagonal: draw vertical line
                g.diag_axes[i].axvline(x=true_params[i], color='red', linestyle='--', linewidth=2, label='True')
            else:
                # Off-diagonal: draw point
                ax = g.axes[i, j]
                ax.scatter(true_params[j], true_params[i], color='red', marker='*', s=100, zorder=10, label='True' if (i==1 and j==0) else "")
    
    # Add Legend (only once)
    handles = [plt.Line2D([0], [0], color='red', linestyle='--', label='True Param')]
    g.fig.legend(handles=handles, loc='upper right')
    
    g.fig.suptitle(f"Posterior Samples - Iteration {iteration}", y=1.02)
    
    save_path = os.path.join(data_folder, f"posterior_plot_{iteration}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Posterior plot saved to {save_path}")

# ============================================================================
# 4. Main Execution Loop
# ============================================================================

if __name__ == "__main__":
    # MCMC parameters (Reduced for testing/verification, increase for production)
    # Recommended: N_proposal=5000, burn_in_steps=7500
    N_proposal = 5000
    burn_in_steps = 1000
    
    # Verify prior config
    print(f"Using Prior Configuration: {PRIOR_TYPE}")
    print(CURRENT_PRIOR)

    # Number of iterations (Reduced for testing/verification)
    n_iterations = 1 

    for it in range(n_iterations):
        iter_start_time = time.time()
        print(f"--- Starting Iteration {it} ---")
        
        # 1. Generate NEW obs_xs for this iteration
        true_ps_val, obs_xs_it = get_ground_truth()
        # obs_xs_it is already (n, 2)
        
        np.save(os.path.join(data_folder, f"obs_xs_{it}.npy"), obs_xs_it)
        print(f"Generated and saved obs_xs_{it}.npy")

        # 2. Immediately run MCMC for this obs_xs
        print(f"Running MCMC for iteration {it}...")
        Theta_mcmc_tensor, accp = mcmc(obs_xs_it, N_proposal, burn_in_steps)
        ps_it = Theta_mcmc_tensor.numpy()
        np.save(os.path.join(data_folder, f"ps_{it}.npy"), ps_it)
        print(f"Saved posterior samples to ps_{it}.npy")

        # 2.5 Plot Posterior
        plot_posterior(ps_it, true_ps_val, it)

        # 3. Calculate h_mmd based on CURRENT ps_it
        ps_quantile = ps_it.copy()
        ps_quantile[:, 3] = np.abs(ps_quantile[:, 3])
        ps_quantile[:, 2] = np.abs(ps_quantile[:, 2])
        Diff = pairwise_distances(ps_quantile, metric="euclidean")
        diff = Diff[np.triu_indices(ps_it.shape[0], 1)]
        h_mmd_it = np.median(diff)
        print(f"Iteration {it} h_mmd: {h_mmd_it}")
        np.save(os.path.join(data_folder, f"h_mmd_{it}.npy"), h_mmd_it)

        # 4. Generate training set for this iteration
        # Generate Prior Draws
        Theta = np.array([prior_generator(d) for _ in range(N)])
        
        # Simulate Data (Sim preserved shape logic is now default in simulator)
        X = simulator(Theta, n_samples=n)
        
        # NO STEREO PROJECTION
        # X shape is (N, n, 2)
        
        # Flatten X for training: (N, n*2)
        XS = X.reshape(N, n * 2)
        x_train = np.concatenate((Theta, XS), axis=1)

        # Removed MMD weight calculation and select_index as requested
        # x_train_nn is now identical to x_train (Theta + XS)
        x_train_nn = x_train.copy()
        
        # Casting
        Theta = Theta.astype("float32")
        X = X.astype("float32")
        x_train = x_train.astype("float32")
        x_train_nn = x_train_nn.astype("float32")
        
        # Save dictionary for BayesFlow (optional, keeping for compatibility)
        keys = [
            "prior_non_batchable_context",
            "prior_batchable_context",
            "prior_draws",
            "sim_non_batchable_context",
            "sim_batchable_context",
            "sim_data",
        ]
        x_train_bf = dict.fromkeys(keys)
        x_train_bf["prior_draws"] = Theta
        x_train_bf["sim_data"] = X

        # Save Files
        np.save(os.path.join(data_folder, f"x_train_{it}.npy"), x_train)
        np.save(os.path.join(data_folder, f"x_train_nn_{it}.npy"), x_train_nn)
        np.save(os.path.join(data_folder, f"X_{it}.npy"), X)
        np.save(os.path.join(data_folder, f"Theta_{it}.npy"), Theta)
        
        file_path = os.path.join(data_folder, f"x_train_bf_{it}.pkl")
        with open(file_path, "wb") as pickle_file:
            pickle.dump(x_train_bf, pickle_file)
        
        iter_end_time = time.time()
        iter_duration = iter_end_time - iter_start_time
        print(f"Iteration {it} data generation complete. Duration: {iter_duration:.2f} seconds ({iter_duration/60:.2f} minutes).\n")
