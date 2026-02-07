
import pymc as pm
import pytensor.tensor as pt
import numpy as np
import arviz as az

def run_pymc(obs_xs, n_draws=2000, n_tune=1000, chains=4, target_accept=0.9):
    """
    Run MCMC using PyMC for the Gaussian Task.
    
    Parameters:
    - obs_xs: Observed data (n_samples, 2)
    - n_draws: Number of samples to draw
    - n_tune: Number of tuning steps
    - chains: Number of chains
    - target_accept: Target acceptance rate for NUTS
    
    Returns:
    - flat_samples: (n_draws * chains, 5) numpy array of posterior samples
    """
    print(f"Setting up PyMC model for data shape {obs_xs.shape}...")
    
    with pm.Model() as model:
        # 1. Priors (Uniform [-3, 3] for all parameters as per GaussianTask default)
        # Parameters: m0, m1, s0, s1, r
        theta = pm.Uniform("theta", lower=-3.0, upper=3.0, shape=5)
        
        m0 = theta[0]
        m1 = theta[1]
        s0 = theta[2]
        s1 = theta[3]
        r = theta[4]
        
        # 2. Transformations
        # s0, s1 are squared in the simulator/likelihood logic
        s0_real = s0**2
        s1_real = s1**2
        
        # r is passed through tanh
        rho = pm.math.tanh(r)
        
        # 3. Construct Covariance Matrix
        # Sigma = [[s0_real**2, rho * s0_real * s1_real],
        #          [rho * s0_real * s1_real, s1_real**2]]
        
        # Construct full covariance matrix manually
        cov_00 = s0_real**2
        cov_01 = rho * s0_real * s1_real
        cov_11 = s1_real**2
        
        # Stack to form matrix (2, 2)
        # Note: PyMC/PyTensor handling of matrices can be tricky.
        # Alternatively, we can specify mu and cov directly to MvNormal
        
        cov = pt.stack([
            pt.stack([cov_00, cov_01]),
            pt.stack([cov_01, cov_11])
        ])
        
        mu = pt.stack([m0, m1])
        
        # 4. Likelihood
        obs = pm.MvNormal("obs", mu=mu, cov=cov, observed=obs_xs)
        
        # 5. Sampling
        print("Starting PyMC sampling...")
        trace = pm.sample(draws=n_draws, tune=n_tune, chains=chains, target_accept=target_accept, progressbar=True)
        
        # 6. Extract samples
        # We need the 'theta' values
        # trace.posterior['theta'] is (chains, draws, 5)
        posterior = trace.posterior['theta'].values
        
        # Flatten: (chains * draws, 5)
        flat_samples = posterior.reshape(-1, 5)
        
        # Diagnostics
        print("\n=== PyMC Diagnostics ===")
        print(az.summary(trace, var_names=["theta"]))
        
        return flat_samples

if __name__ == "__main__":
    # Test
    # Generate dummy data
    from data_generation import GaussianTask, get_ground_truth
    task = GaussianTask()
    _, obs_2d, _ = get_ground_truth(task)
    
    samples = run_pymc(obs_2d, n_draws=100, n_tune=50, chains=2)
    print("Samples shape:", samples.shape)
