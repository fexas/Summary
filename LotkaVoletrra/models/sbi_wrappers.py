import torch
import numpy as np
from sbi.inference import NPE_A, NPE_B, SNPE
from sbi.utils import BoxUniform

def calculate_summary_statistics(x):
    """
    Calculate summary statistics for Lotka-Volterra time series as described in
    ExperimentDescription.md.
    
    Data x is (batch_size, 151, 2) corresponding to 30 time units with dt=0.2.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        
    # Handle single observation case (T, D) -> (1, T, D)
    if x.ndim == 2:
        x = x.unsqueeze(0)
        
    # Ensure x is 3D (batch, steps, dims)
    if x.ndim != 3:
        raise ValueError(f"Expected input shape (batch, steps, dims), got {x.shape}")

    batch_size, n_steps, n_dims = x.shape
    # n_dims should be 2 (Prey, Predator)
    
    # 1. Means (batch, 2)
    means = x.mean(dim=1)
    
    # 2. Log Variance (batch, 2)
    # Add small epsilon to avoid log(0)
    variances = x.var(dim=1, unbiased=False) # Use biased or unbiased? usually unbiased.
    log_vars = torch.log(variances + 1e-8)
    
    # Pre-calculate centered data for correlations
    x_centered = x - means.unsqueeze(1)
    
    # Helper to calc autocorrelation at lag k
    def calc_autocorr(k):
        # Numerator: sum_{t=0}^{T-k-1} (x_t - mu)(x_{t+k} - mu)
        # Denominator: sum_{t=0}^{T-1} (x_t - mu)^2  (which is var * T)
        
        # Slice views
        x_t = x_centered[:, :-k, :]
        x_tk = x_centered[:, k:, :]
        
        # Covariance at lag k
        cov_k = (x_t * x_tk).sum(dim=1)
        
        # Variance * n_steps (denominator)
        var_n = (x_centered ** 2).sum(dim=1)
        
        # Autocorrelation
        # Avoid division by zero
        rho_k = cov_k / (var_n + 1e-8)
        return rho_k

    # 3. Autocorrelation at lag 1 (dt=0.2) -> (batch, 2)
    ac_lag1 = calc_autocorr(1)
    
    # 4. Autocorrelation at lag 2 (dt=0.4) -> (batch, 2)
    ac_lag2 = calc_autocorr(2)
    
    # 5. Cross-correlation (batch, 1)
    # Pearson correlation between series 0 and series 1
    # cov(x0, x1) / (std(x0) * std(x1))
    
    x0_c = x_centered[:, :, 0]
    x1_c = x_centered[:, :, 1]
    
    cov_01 = (x0_c * x1_c).sum(dim=1)
    var_0 = (x0_c ** 2).sum(dim=1)
    var_1 = (x1_c ** 2).sum(dim=1)
    
    cross_corr = cov_01 / (torch.sqrt(var_0 * var_1) + 1e-8)
    cross_corr = cross_corr.unsqueeze(1) # (batch, 1)
    
    # Concatenate all features
    # Order: Mean(2), LogVar(2), AC1(2), AC2(2), CrossCorr(1) -> Total 9
    summary_stats = torch.cat([
        means, 
        log_vars, 
        ac_lag1, 
        ac_lag2, 
        cross_corr
    ], dim=1)
    
    return summary_stats

def run_sbi_model(model_type, train_loader, x_obs, theta_true, task, device="cpu", num_rounds=1, sims_per_round=1000, max_epochs=1000, **kwargs):
    """
    Run SBI training and inference using SNPE-A logic.
    
    model_type: 'snpe_a', 'snpe_b', or 'npe'
    """
    print(f"=== Running {model_type.upper()} (Rounds={num_rounds}, Sims/Round={sims_per_round}, MaxEpochs={max_epochs}) ===")
    
    # 0. Prepare Observation (x_obs)
    if isinstance(x_obs, np.ndarray):
        x_obs = torch.from_numpy(x_obs).float()
    
    # Handle dimensions for summary stats calculation
    if x_obs.ndim == 2: # (steps, 2)
        x_obs_input = x_obs.unsqueeze(0)
    elif x_obs.ndim == 3: # (1, steps, 2)
        x_obs_input = x_obs
    else:
        raise ValueError(f"Invalid x_obs shape: {x_obs.shape}")

    x_obs_stats = calculate_summary_statistics(x_obs_input)
    print(f"Observation Summary Stats: {x_obs_stats}")

    # 1. Define Prior
    # Use task bounds for prior (BoxUniform)
    low = torch.tensor(task.lower, dtype=torch.float32)
    high = torch.tensor(task.upper, dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)
    
    # 2. Instantiate Inference Object
    # Force CPU to avoid MPS autograd issues if needed, but SBI usually handles it if configured.
    # We will respect the device passed, but fallback to cpu for safety if needed.
    device_sbi = "cpu" 
    
    if model_type.lower() == 'snpe_a' or model_type.lower() == 'npe':
        # Default to NPE_A as requested for 'npe' or 'snpe_a'
        inference = NPE_A(prior=prior, device=device_sbi)
    elif model_type.lower() == 'snpe_b':
        inference = NPE_B(prior=prior, device=device_sbi)
    elif model_type.lower() == 'snpe_c':
        inference = SNPE(prior=prior, device=device_sbi)
    else:
        # Fallback to NPE_A
        inference = NPE_A(prior=prior, device=device_sbi)

    # 3. Sequential Inference Loop (SNPE-A Style)
    proposal = prior
    
    for r in range(num_rounds):
        print(f"--- SBI Round {r+1}/{num_rounds} ---")
        
        # A. Simulate Data
        # Sample theta from proposal
        theta = proposal.sample((sims_per_round,))
        
        # Simulator (wrapper to handle numpy/torch and stats)
        theta_np = theta.cpu().numpy()
        x_sim_np = task.simulator(theta_np) # Returns (N, T, D)
        x_sim_torch = torch.from_numpy(x_sim_np).float()
        
        # Calculate summary stats
        x = calculate_summary_statistics(x_sim_torch)
        
        # Ensure data is on correct device
        theta = theta.to(device_sbi)
        x = x.to(device_sbi)
        
        # B. Train
        # NPE-A trains a Gaussian density estimator in all but the last round.
        # In the last round, it trains a mixture of Gaussians.
        final_round = (r == num_rounds - 1)
        
        with torch.enable_grad():
            # append_simulations returns the inference object, then we call train
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(
                max_num_epochs=max_epochs, 
                final_round=final_round
            )
            
        # C. Build Posterior & Update Proposal
        posterior = inference.build_posterior(density_estimator).set_default_x(x_obs_stats)
        proposal = posterior
        
    # 4. Final Sampling
    print("Sampling from final posterior...")
    
    n_samples = 1000 # Final posterior samples count
    samples = posterior.sample((n_samples,))
    
    return samples.detach().cpu().numpy()
