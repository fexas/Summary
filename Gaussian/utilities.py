
import torch
import numpy as np
import time

def compute_bandwidth_torch(model, x_obs, task, n_samples=5000, quantile_level=0.005, device="cpu"):
    """
    Compute bandwidth for likelihood estimation using quantiles of distance.
    """
    print("Computing bandwidth for refinement...")
    # model.eval() # Assumed to be in eval mode or handled by wrapper
    
    # 1. Generate Theta0 from learned model
    # x_obs: (n, d_x)
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device) # (1, n, d_x)
    
    # Replicate x_obs for batch generation
    x_obs_batch = x_obs_tensor.expand(n_samples, -1, -1) # (N0, n, d_x)
    
    # Use model.sample_posterior to be generic
    # If model doesn't have sample_posterior, try legacy G(z, stats) approach
    if hasattr(model, "sample_posterior"):
        # Expecting sample_posterior to handle stats computation internally if needed
        # Or we pass stats? SMMD needs stats. BayesFlow needs conditions.
        # Let's assume wrapper handles (x_obs)
        # Wait, x_obs_batch is (N0, n, d_x).
        # Wrapper should handle this.
        # But wait, SMMD sample_posterior might need Z?
        # Let's define a standard interface: model.sample_posterior(x_obs, n_samples)
        Theta0_np = model.sample_posterior(x_obs, n_samples) # Returns numpy (n_samples, d)
    else:
        # Legacy SMMD support (if not wrapped)
        with torch.no_grad():
            stats = model.T(x_obs_batch) # (N0, p)
            Z = torch.randn(n_samples, 1, task.d, device=device) # Assuming task.d exists
            Theta0 = model.G(Z, stats).squeeze(1) # (N0, d)
        Theta0_np = Theta0.cpu().numpy()
    
    # 2. Simulate Data from Theta0
    # Use GaussianTask simulator (batch)
    # Output: (N0, n, d_x)
    xn_0 = task.simulator(Theta0_np, n_samples=task.n) 
    xn_0_tensor = torch.from_numpy(xn_0).float().to(device)
    
    # 3. Compute Stats and Distance
    with torch.no_grad():
        # Generic model.compute_stats or model.T
        if hasattr(model, "compute_stats"):
             TT = model.compute_stats(xn_0_tensor)
             T_target = model.compute_stats(x_obs_tensor)
        else:
             TT = model.T(xn_0_tensor) # (N0, p)
             T_target = model.T(x_obs_tensor) # (1, p)
        
        # Distance (Euclidean)
        diff = T_target - TT
        dist_sq = torch.sum(diff**2, dim=1) # (N0,)
        dist = torch.sqrt(dist_sq)
        
    dist_np = dist.cpu().numpy()
    
    # 4. Quantile
    quan1 = np.quantile(dist_np, quantile_level)
    
    print(f"Computed bandwidth (epsilon): {quan1}")
    return quan1

def approximate_likelihood_torch(model, theta, x_obs_stats, task, nsims, epsilon, device="cpu"):
    """
    Compute KDE-based likelihood for a batch of parameters.
    theta: (batch, d) numpy
    x_obs_stats: (1, p) torch tensor
    Returns: (batch,) numpy
    """
    batch_size = theta.shape[0]
    
    # 1. Simulate: (batch * nsims, n, d_x)
    theta_exp = np.repeat(theta, nsims, axis=0)
    
    sim_data = task.simulator(theta_exp, n_samples=task.n)
    sim_data_tensor = torch.from_numpy(sim_data).float().to(device)
    
    # 2. Compute Stats: (B*nsims, p)
    with torch.no_grad():
        if hasattr(model, "compute_stats"):
            sim_stats = model.compute_stats(sim_data_tensor)
        else:
            sim_stats = model.T(sim_data_tensor)
        
    # 3. Reshape and Compute Distance
    # (B, nsims, p)
    sim_stats = sim_stats.view(batch_size, nsims, -1)
    
    # x_obs_stats: (1, p) -> (1, 1, p)
    target = x_obs_stats.unsqueeze(0)
    
    # Dist sq: (B, nsims)
    diff = sim_stats - target
    dist_sq = torch.sum(diff**2, dim=-1)
    
    # 4. KDE
    # kernel = exp(-dist_sq / (2 * epsilon^2))
    kernel = torch.exp(-dist_sq / (2 * epsilon**2))
    likelihood = torch.mean(kernel, dim=1) # (B,)
    
    return likelihood.cpu().numpy()

def refine_posterior(model, x_obs, task, 
                     n_chains=1000, n_samples=1, burn_in=99, 
                     thin=1, nsims=50, epsilon=None, proposal_std=0.5, device="cpu"):
    """
    Run Parallel MCMC Refinement.
    """
    print(f"Starting MCMC Refinement (chains={n_chains}, burn_in={burn_in}, samples={n_samples})...")
    # model.eval()
    
    # 1. Compute Bandwidth if not provided
    if epsilon is None:
        epsilon = compute_bandwidth_torch(model, x_obs, task, device=device)
        
    # 2. Initialize Chains from Model Posterior
    x_obs_tensor = torch.from_numpy(x_obs[np.newaxis, ...]).float().to(device)
    
    # Compute stats for x_obs once
    with torch.no_grad():
        if hasattr(model, "compute_stats"):
            x_obs_stats = model.compute_stats(x_obs_tensor)
        else:
            x_obs_stats = model.T(x_obs_tensor)
    
    # Sample Initial Points
    if hasattr(model, "sample_posterior"):
        # Should return (n_chains, d)
        current_theta = model.sample_posterior(x_obs, n_chains)
    else:
        with torch.no_grad():
            Z = torch.randn(n_chains, 1, task.d, device=device)
            current_theta = model.G(Z, x_obs_stats.expand(n_chains, -1)).squeeze(1).cpu().numpy()
            
    # Clip to bounds initially
    current_theta = np.clip(current_theta, task.lower, task.upper)
    
    # 3. Dimensional-wise Proposal Step Size
    # Calculate std per dimension from initial samples
    std_per_dim = np.std(current_theta, axis=0)
    # Avoid zero std
    std_per_dim = np.maximum(std_per_dim, 1e-6)
    
    # Use proposal_std as a scaling factor
    proposal_scale = std_per_dim * proposal_std
    print(f"Proposal Scale (per dim): {proposal_scale}")
    
    # 4. Initial Log Probabilities
    # Prior
    current_log_prior = task.log_prior(current_theta) # (chains,)
    
    # Likelihood
    current_likelihood = approximate_likelihood_torch(model, current_theta, x_obs_stats, task, nsims, epsilon, device=device)
    
    # Current Ratio (Prior * Likelihood)
    current_prob = np.exp(current_log_prior) * current_likelihood
    
    # Storage
    samples = []
    total_accepted = 0
    
    start_time = time.time()
    
    # MCMC Loop
    total_sampling_steps = n_samples * thin
    total_steps = burn_in + total_sampling_steps
    
    print(f"Total MCMC steps: {total_steps} (Burn-in: {burn_in}, Sampling: {total_sampling_steps})")
    
    for step in range(1, total_steps + 1):
        # Propose
        proposal_noise = np.random.randn(n_chains, task.d) * proposal_scale
        proposed_theta = current_theta + proposal_noise
        
        # Proposed Log Prob
        proposed_log_prior = task.log_prior(proposed_theta)
        
        proposed_likelihood = approximate_likelihood_torch(model, proposed_theta, x_obs_stats, task, nsims, epsilon, device=device)
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
            
            # Check if we should collect sample
            sampling_step = step - burn_in
            if sampling_step % thin == 0:
                samples.append(current_theta.copy())
                
        if step % 10 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            print(f"Step {step}/{total_steps} | Accepted: {np.mean(accept_mask):.2%} | Time: {elapsed:.2f}s")
            
    # Collect Samples
    if not samples:
        samples.append(current_theta.copy())
        
    posterior_samples = np.vstack(samples)
    
    acceptance_rate = total_accepted / (n_chains * total_sampling_steps) if total_sampling_steps > 0 else 0
    print(f"Refinement Complete. Acceptance Rate: {acceptance_rate:.2%}")
    
    return posterior_samples
