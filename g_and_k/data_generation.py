"""
Data generation for g-and-k distribution.
This script handles the simulator and prior configuration.
No MCMC is included as the likelihood is intractable.
"""

import numpy as np
import scipy.stats
import os
import pickle

# ============================================================================
# 1. Configuration
# ============================================================================

# Data parameters
N = 10000    # Size of the training dataset (Reference Table size)
n = 100      # Sample size per observation
d = 4        # Parameters: A, B, g, k
d_x = 1      # Univariate data (g-and-k produces scalar samples)

c = 0.8      # Fixed constant for g-and-k

# Prior Configurations
# Parameters order: A, B, g, k
# Constraints: B > 0, k > -0.5
PRIOR_CONFIGS = {
    'vague': {
        'A': [-10.0, 10.0],
        'B': [0.1, 10.0],
        'g': [-5.0, 5.0],
        'k': [-0.4, 5.0]
    },
    'weak_informative': {
        'A': [0.0, 6.0],
        'B': [0.5, 3.0],
        'g': [0.0, 4.0],
        'k': [0.0, 2.0]
    },
    'informative': {
        'A': [2.5, 3.5],      # True is 3.0
        'B': [0.8, 1.2],      # True is 1.0
        'g': [1.5, 2.5],      # True is 2.0
        'k': [0.3, 0.7]       # True is 0.5
    }
}

# Default Ground Truth Parameters
TRUE_PARAMS = np.array([3.0, 1.0, 2.0, 0.5]) # A, B, g, k

# ============================================================================
# 2. Simulator & Prior
# ============================================================================

def prior_generator(prior_type='weak_informative'):
    """
    Generate a single parameter set theta = [A, B, g, k] from the specified prior.
    """
    config = PRIOR_CONFIGS[prior_type]
    
    A = np.random.uniform(config['A'][0], config['A'][1])
    B = np.random.uniform(config['B'][0], config['B'][1])
    g = np.random.uniform(config['g'][0], config['g'][1])
    k = np.random.uniform(config['k'][0], config['k'][1])
    
    return np.array([A, B, g, k])

def simulator(theta, n_samples=n):
    """
    Simulate data from g-and-k distribution.
    theta: (batch_size, 4) or (4,)
    Returns: (batch_size, n_samples, 1)
    """
    theta = np.asarray(theta)
    if theta.ndim == 1:
        theta = theta[np.newaxis, :]
    
    batch_size = theta.shape[0]
    
    A = theta[:, 0:1]
    B = theta[:, 1:2]
    g = theta[:, 2:3]
    k = theta[:, 3:4]
    
    # Generate standard normal z
    z = np.random.normal(0, 1, size=(batch_size, n_samples))
    
    # g-and-k formula
    # Q(z) = A + B * (1 + c * tanh(g*z/2)) * z * (1 + z^2)^k
    
    term1 = 1 + c * np.tanh(g * z / 2.0)
    term2 = z
    term3 = (1 + z**2)**k
    
    x = A + B * term1 * term2 * term3
    
    # Expand dims to match (batch, n, d_x) where d_x=1
    x = x[:, :, np.newaxis]
    
    return x

def generate_dataset(prior_type, n_sims=N, n_obs=n):
    """
    Generates the Reference Table (Theta, X) for training.
    """
    print(f"Generating dataset with {prior_type} prior...")
    
    # 1. Sample Theta from Prior
    thetas = []
    for _ in range(n_sims):
        thetas.append(prior_generator(prior_type))
    thetas = np.array(thetas).astype(np.float32) # (N, 4)
    
    # 2. Simulate X
    xs = simulator(thetas, n_samples=n_obs).astype(np.float32) # (N, n, 1)
    
    return thetas, xs

def get_ground_truth():
    """
    Returns ground truth parameters and one observation.
    """
    theta_true = TRUE_PARAMS
    x_obs = simulator(theta_true, n_samples=n) # (1, n, 1)
    return theta_true, x_obs[0] # Return (4,), (n, 1)

if __name__ == "__main__":
    # Test the generator
    t, x = generate_dataset('weak_informative', n_sims=10)
    print("Theta shape:", t.shape)
    print("X shape:", x.shape)
    print("Sample X:", x[0, :5, 0])
