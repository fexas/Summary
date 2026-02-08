import torch
import numpy as np
import time

# =============================================================================
# Shared Helpers
# =============================================================================

def run_mcmc_refinement(current_theta, x_obs_stats, task, 
                       log_prior_fn, likelihood_fn, 
                       n_chains, n_samples, burn_in, thin, 
                       proposal_std=0.5):
    """
    Generic MCMC refinement loop.
    
    Args:
        current_theta: (n_chains, d) Initial samples.
        x_obs_stats: (1, p) Target summary statistics.
        task: GaussianTask instance.
        log_prior_fn: Function (theta) -> log_prob
        likelihood_fn: Function (theta, x_obs_stats) -> likelihood (numpy array)
        ...
    """
    # Clip to bounds initially
    current_theta = np.clip(current_theta, task.lower, task.upper)
    
    # Dimensional-wise Proposal Step Size
    std_per_dim = np.std(current_theta, axis=0)
    std_per_dim = np.maximum(std_per_dim, 1e-6)
    proposal_scale = std_per_dim * proposal_std
    print(f"Proposal Scale (per dim): {proposal_scale}")
    
    # Initial Log Probabilities
    current_log_prior = log_prior_fn(current_theta)
    current_likelihood = likelihood_fn(current_theta, x_obs_stats)
    current_prob = np.exp(current_log_prior) * current_likelihood
    
    samples = []
    total_accepted = 0
    start_time = time.time()
    
    total_sampling_steps = n_samples * thin
    total_steps = burn_in + total_sampling_steps
    
    print(f"Total MCMC steps: {total_steps} (Burn-in: {burn_in}, Sampling: {total_sampling_steps})")
    
    for step in range(1, total_steps + 1):
        # Propose
        proposal_noise = np.random.randn(n_chains, task.d) * proposal_scale
        proposed_theta = current_theta + proposal_noise
        
        # Proposed Log Prob
        proposed_log_prior = log_prior_fn(proposed_theta)
        proposed_likelihood = likelihood_fn(proposed_theta, x_obs_stats)
        proposed_prob = np.exp(proposed_log_prior) * proposed_likelihood
        
        # Acceptance Probability
        ratio = np.divide(proposed_prob, current_prob, out=np.zeros_like(current_prob), where=current_prob!=0)
        accept_prob = np.minimum(1.0, ratio)
        
        # Random Uniform
        u = np.random.rand(n_chains)
        accept_mask = u < accept_prob
        
        # Update
        current_theta[accept_mask] = proposed_theta[accept_mask]
        current_prob[accept_mask] = proposed_prob[accept_mask]
        
        if step > burn_in:
            total_accepted += np.sum(accept_mask)
            if (step - burn_in) % thin == 0:
                samples.append(current_theta.copy())
                
        if step % 10 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            print(f"Step {step}/{total_steps} | Accepted: {np.mean(accept_mask):.2%} | Time: {elapsed:.2f}s")
            
    if not samples:
        samples.append(current_theta.copy())
        
    posterior_samples = np.vstack(samples)
    acceptance_rate = total_accepted / (n_chains * total_sampling_steps) if total_sampling_steps > 0 else 0
    print(f"Refinement Complete. Acceptance Rate: {acceptance_rate:.2%}")
    
    return posterior_samples

def compute_bandwidth_core(theta0_np, x_obs_tensor, task, stats_fn, n_samples, quantile_level, device):
    """
    Core bandwidth computation logic.
    """
    # Simulate Data from Theta0
    xn_0 = task.simulator(theta0_np, n_samples=task.n) 
    xn_0_tensor = torch.from_numpy(xn_0).float().to(device)
    
    with torch.no_grad():
        TT = stats_fn(xn_0_tensor)
        T_target = stats_fn(x_obs_tensor)
        
        diff = T_target - TT
        dist_sq = torch.sum(diff**2, dim=1)
        dist = torch.sqrt(dist_sq)
        
    dist_np = dist.cpu().numpy()
    quan1 = np.quantile(dist_np, quantile_level)
    print(f"Computed bandwidth (epsilon): {quan1}")
    return quan1

def approximate_likelihood_core(theta, x_obs_stats, task, stats_fn, nsims, epsilon, device):
    """
    Core approximate likelihood computation.
    """
    batch_size = theta.shape[0]
    theta_exp = np.repeat(theta, nsims, axis=0)
    sim_data = task.simulator(theta_exp, n_samples=task.n)
    sim_data_tensor = torch.from_numpy(sim_data).float().to(device)
    
    with torch.no_grad():
        sim_stats = stats_fn(sim_data_tensor)
        
    sim_stats = sim_stats.view(batch_size, nsims, -1)
    target = x_obs_stats.unsqueeze(0)
    
    diff = sim_stats - target
    dist_sq = torch.sum(diff**2, dim=-1)
    
    kernel = torch.exp(-dist_sq / (2 * epsilon**2))
    likelihood = torch.mean(kernel, dim=1)
    
    return likelihood.cpu().numpy()

# =============================================================================
# SMMD/MMD Implementation
# =============================================================================

def compute_bandwidth_smmd(model, x_obs, task, n_samples=5000, quantile_level=0.005, device="cpu"):
    print("Computing bandwidth for SMMD refinement...")
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device)
    x_obs_batch = x_obs_tensor.expand(n_samples, -1, -1)
    
    with torch.no_grad():
        stats = model.T(x_obs_batch)
        Z = torch.randn(n_samples, 1, task.d, device=device)
        Theta0 = model.G(Z, stats).squeeze(1)
    Theta0_np = Theta0.cpu().numpy()
    
    return compute_bandwidth_core(Theta0_np, x_obs_tensor, task, model.T, n_samples, quantile_level, device)

def refine_posterior_smmd(model, x_obs, task, 
                         n_chains=1000, n_samples=1, burn_in=99, 
                         thin=1, nsims=50, epsilon=None, proposal_std=0.5, device="cpu"):
    print(f"Starting SMMD MCMC Refinement...")
    
    if epsilon is None:
        epsilon = compute_bandwidth_smmd(model, x_obs, task, device=device)
        
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device)
    
    # 1. Stats and Initial Samples
    with torch.no_grad():
        x_obs_stats = model.T(x_obs_tensor)
        # Initial samples
        Z = torch.randn(n_chains, 1, task.d, device=device)
        current_theta = model.G(Z, x_obs_stats.expand(n_chains, -1)).squeeze(1).cpu().numpy()
        
    # 2. Define Likelihood Function
    def likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(theta, target_stats, task, model.T, nsims, epsilon, device)
        
    # 3. Run MCMC
    return run_mcmc_refinement(current_theta, x_obs_stats, task, 
                              task.log_prior, likelihood_fn, 
                              n_chains, n_samples, burn_in, thin, proposal_std)

# =============================================================================
# BayesFlow Implementation
# =============================================================================

def compute_bandwidth_bayesflow(model, x_obs, task, n_samples=5000, quantile_level=0.005, device="cpu"):
    print("Computing bandwidth for BayesFlow refinement...")
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device)
    
    # BayesFlow adapter needs CPU/Numpy
    x_obs_cpu = x_obs_tensor.cpu().numpy()
    
    # Sample from BayesFlow model
    # model.sample(conditions=...) returns (n_samples, batch, d) or (batch, n_samples, d)
    
    # Update for Keras 3 with explicit adapter (requires dict)
    conditions_dict = {"summary_variables": x_obs_cpu}
    Theta0_np = model.sample(conditions=conditions_dict, num_samples=n_samples)
    
    # Handle Dict output (BayesFlow/Keras3)
    if isinstance(Theta0_np, dict):
        Theta0_np = Theta0_np["inference_variables"]
    
    # Theta0_np shape: (1, n_samples, d) (since batch_size=1)
    # Ensure numpy array
    if isinstance(Theta0_np, torch.Tensor):
        Theta0_np = Theta0_np.detach().cpu().numpy()
    elif not isinstance(Theta0_np, np.ndarray):
         Theta0_np = np.asarray(Theta0_np)
         
    Theta0_np = Theta0_np.reshape(n_samples, -1)
        
    return compute_bandwidth_core(Theta0_np, x_obs_tensor, task, model.summary_network, n_samples, quantile_level, device)

def refine_posterior_bayesflow(model, x_obs, task, 
                              n_chains=1000, n_samples=1, burn_in=99, 
                              thin=1, nsims=50, epsilon=None, proposal_std=0.5, device="cpu"):
    print(f"Starting BayesFlow MCMC Refinement...")
    
    if epsilon is None:
        epsilon = compute_bandwidth_bayesflow(model, x_obs, task, device=device)
        
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device)
    
    # 1. Stats and Initial Samples
    with torch.no_grad():
        x_obs_stats = model.summary_network(x_obs_tensor)
        
        # BayesFlow adapter needs CPU/Numpy
        x_obs_cpu = x_obs_tensor.cpu().numpy()
        conditions_dict = {"summary_variables": x_obs_cpu}
        current_theta = model.sample(conditions=conditions_dict, num_samples=n_chains)
        
        # Handle Dict output (BayesFlow/Keras3)
        if isinstance(current_theta, dict):
            current_theta = current_theta["inference_variables"]
            
        # Ensure numpy array
        if isinstance(current_theta, torch.Tensor):
            current_theta = current_theta.detach().cpu().numpy()
        elif not isinstance(current_theta, np.ndarray):
             current_theta = np.asarray(current_theta)
             
        current_theta = current_theta.reshape(n_chains, -1)
            
    # 2. Define Likelihood Function
    def likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(theta, target_stats, task, model.summary_network, nsims, epsilon, device)
        
    # 3. Run MCMC
    return run_mcmc_refinement(current_theta, x_obs_stats, task, 
                              task.log_prior, likelihood_fn, 
                              n_chains, n_samples, burn_in, thin, proposal_std)

# Backwards compatibility / Dispatcher (optional)
def refine_posterior(model, *args, **kwargs):
    if hasattr(model, "summary_network"):
        return refine_posterior_bayesflow(model, *args, **kwargs)
    else:
        return refine_posterior_smmd(model, *args, **kwargs)

def compute_bandwidth_torch(model, *args, **kwargs):
    if hasattr(model, "summary_network"):
        return compute_bandwidth_bayesflow(model, *args, **kwargs)
    else:
        return compute_bandwidth_smmd(model, *args, **kwargs)
