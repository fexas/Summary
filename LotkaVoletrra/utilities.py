import torch
import numpy as np
import time
from sklearn.neighbors import KernelDensity

# =============================================================================
# Shared Helpers
# =============================================================================

def fit_kde_and_evaluate(samples, evaluate_points):
    """
    Fit a KDE to samples and evaluate log density at evaluate_points.
    """
    # Use cross-validation or heuristic for bandwidth?
    # Simple heuristic: Scott's Rule
    n, d = samples.shape
    bandwidth = n ** (-1.0 / (d + 4))
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(samples)
    log_density = kde.score_samples(evaluate_points)
    return log_density

def run_mcmc_refinement(current_theta, x_obs_stats, task, 
                       log_prior_fn, likelihood_fn, 
                       n_chains, n_samples, burn_in, thin, 
                       proposal_std=0.5):
    """
    Generic MCMC refinement loop.
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
    # Note: task.simulator in LV returns (batch, n_obs, 2)
    xn_0 = task.simulator(theta0_np, n_samples=task.n_obs) 
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

def approximate_likelihood_core(theta, x_obs_stats, task, stats_fn, epsilon, device):
    """
    Core approximate likelihood computation.
    Optimized for nsims=1 and minimal overhead.
    """
    # 1. Simulate (Numpy) - forced by task.simulator
    # theta: (batch, d)
    sim_data = task.simulator(theta, n_samples=task.n_obs)
    
    # 2. To Tensor
    sim_data_tensor = torch.from_numpy(sim_data).float().to(device)
    
    with torch.no_grad():
        # 3. Stats (Tensor)
        sim_stats = stats_fn(sim_data_tensor)
        
        # 4. Distance (Tensor)
        # x_obs_stats should be (1, dim) or (dim,)
        # sim_stats is (batch, dim)
        diff = sim_stats - x_obs_stats
        dist_sq = torch.sum(diff**2, dim=-1)
        
        # 5. Kernel (Gaussian)
        # likelihood = exp(-dist^2 / (2*eps^2))
        likelihood = torch.exp(-dist_sq / (2 * epsilon**2))
    
    return likelihood.cpu().numpy()

# =============================================================================
# SMMD/MMD Implementation
# =============================================================================

def compute_bandwidth_smmd(model, x_obs, task, n_samples=5000, quantile_level=0.005, device=None):
    if device is None:
        # Try to infer device from model parameters
        try:
            device = next(model.parameters()).device
        except:
            device = torch.device("cpu")
        
    print(f"Computing bandwidth for SMMD refinement (Device: {device})...")
    
    # Ensure x_obs is float32 and on correct device
    # Handle x_obs if it's already a tensor or numpy
    if isinstance(x_obs, torch.Tensor):
        x_obs_tensor = x_obs.float().to(device)
    else:
        x_obs_tensor = torch.from_numpy(x_obs).float().to(device)
        
    if x_obs_tensor.ndim == 2:
        x_obs_tensor = x_obs_tensor.unsqueeze(0)
        
    x_obs_batch = x_obs_tensor.expand(n_samples, -1, -1)
    
    with torch.no_grad():
        stats = model.T(x_obs_batch)
        Z = torch.randn(n_samples, 1, task.d, device=device)
        Theta0 = model.G(Z, stats).squeeze(1)
    
    Theta0_np = Theta0.cpu().numpy()
    
    return compute_bandwidth_core(Theta0_np, x_obs_tensor, task, model.T, n_samples, quantile_level, device)

def refine_posterior_smmd(model, x_obs, task, 
                         n_chains=1000, n_samples=1, burn_in=99, 
                         thin=1, epsilon=None, proposal_std=0.5, device=None, theta_init=None):
    if device is None:
        try:
            device = next(model.parameters()).device
        except:
            device = torch.device("cpu")
            
    print(f"Starting SMMD MCMC Refinement (Device: {device})...")
    
    if epsilon is None:
        epsilon = compute_bandwidth_smmd(model, x_obs, task, device=device)
        
    if isinstance(x_obs, torch.Tensor):
        x_obs_tensor = x_obs.float().to(device)
    else:
        x_obs_tensor = torch.from_numpy(x_obs).float().to(device)
        
    if x_obs_tensor.ndim == 2:
        x_obs_tensor = x_obs_tensor.unsqueeze(0)
    
    # 1. Stats and Initial Samples
    with torch.no_grad():
        x_obs_stats = model.T(x_obs_tensor)
        # Initial samples
        if theta_init is not None:
            print(f"Using provided initial points for MCMC (shape: {theta_init.shape})")
            current_theta = theta_init
            # Ensure shape matches n_chains if possible, or just use what is given
            if current_theta.shape[0] != n_chains:
                print(f"Warning: theta_init shape {current_theta.shape} != n_chains {n_chains}. Adjusting n_chains.")
                n_chains = current_theta.shape[0]
        else:
            Z = torch.randn(n_chains, 1, task.d, device=device)
            current_theta = model.G(Z, x_obs_stats.expand(n_chains, -1)).squeeze(1).cpu().numpy()
        
    # 2. Define Likelihood Function
    def likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(theta, target_stats, task, model.T, epsilon, device)
        
    # 3. Run MCMC
    return run_mcmc_refinement(current_theta, x_obs_stats, task, 
                              task.log_prior, likelihood_fn, 
                              n_chains, n_samples, burn_in, thin, proposal_std)

# =============================================================================
# BayesFlow Implementation
# =============================================================================

def refine_posterior_bayesflow(model, x_obs, task, 
                             n_chains=1000, n_samples=1, burn_in=99, 
                             thin=1, epsilon=None, proposal_std=0.5, device=None, theta_init=None):
    if device is None:
        # Infer device from model (Keras model usually manages its own device, but we need it for tensors)
        device = "cpu"
        print("Warning: Device not provided for BayesFlow refinement. Defaulting to CPU.")

    print("Starting BayesFlow MCMC Refinement...")
    
    # 1. Bandwidth
    if epsilon is None:
        epsilon = compute_bandwidth_bayesflow(model, x_obs, task, device=device)
        
    # 2. Initial Points
    x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
    if x_obs_cpu.ndim == 2: x_obs_cpu = x_obs_cpu[np.newaxis, ...]

    if theta_init is not None:
        print(f"Using provided initial points for MCMC (shape: {theta_init.shape})")
        current_theta = theta_init
        if current_theta.shape[0] != n_chains:
             print(f"Warning: theta_init shape {current_theta.shape} != n_chains {n_chains}. Adjusting n_chains.")
             n_chains = current_theta.shape[0]
    else:
        # Sample from BayesFlow model
        # Replicate x_obs for batch sampling
        x_obs_rep = np.tile(x_obs_cpu, (n_chains, 1, 1))
        
        post = model.sample(conditions={"summary_variables": x_obs_rep}, num_samples=1)
        if isinstance(post, dict): post = post["inference_variables"]
        current_theta = post.reshape(n_chains, -1)
        if isinstance(current_theta, torch.Tensor):
            current_theta = current_theta.cpu().numpy()
        
    # 3. MCMC
    x_obs_tensor = torch.from_numpy(x_obs_cpu).float().to(device)
    # BayesFlow Summary Network
    # We need to wrap it to accept torch tensors and return torch tensors
    # Keras models usually expect numpy or TF tensors.
    # But approximate_likelihood_core expects a callable that takes torch tensors.
    
    def stats_fn_torch(x_tensor):
        # x_tensor: (batch, n_obs, d_x)
        x_np = x_tensor.cpu().numpy()
        # model.summary_network expects (batch, n_obs, d_x)
        # returns (batch, summary_dim)
        stats = model.summary_network(x_np.astype(np.float32))
        
        if hasattr(stats, "cpu"): stats = stats.cpu()
        if hasattr(stats, "numpy"): stats = stats.numpy()
        
        return torch.from_numpy(stats).float().to(device)

    # Pre-compute x_obs_stats
    # model.summary_network might return a tensor or numpy
    # Ensure float32 for model input
    x_obs_cpu = x_obs_cpu.astype(np.float32)
    x_obs_stats_tf = model.summary_network(x_obs_cpu)
    
    if hasattr(x_obs_stats_tf, "cpu"): x_obs_stats_tf = x_obs_stats_tf.cpu()
    if hasattr(x_obs_stats_tf, "numpy"): x_obs_stats_tf = x_obs_stats_tf.numpy()
    
    x_obs_stats = torch.from_numpy(x_obs_stats_tf).float().to(device)
    
    def likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(theta, target_stats, task, stats_fn_torch, epsilon, device)
        
    return run_mcmc_refinement(current_theta, x_obs_stats, task,
                              task.log_prior, likelihood_fn,
                              n_chains, n_samples, burn_in, thin, proposal_std)

def compute_bandwidth_bayesflow(model, x_obs, task, n_samples=5000, quantile_level=0.005, device="cpu"):
    print(f"Computing bandwidth for BayesFlow refinement...")
    
    # Generate simulations from Prior
    theta_prior = task.sample_prior(n_samples, "vague")
    x_sim = task.simulator(theta_prior) # (n, n_obs, d_x)
    
    # Get stats
    # Ensure x_sim is float32 (simulator might return int for population counts)
    x_sim_float = x_sim.astype(np.float32)
    stats_sim = model.summary_network(x_sim_float)
    if hasattr(stats_sim, "cpu"): stats_sim = stats_sim.cpu()
    if hasattr(stats_sim, "numpy"): stats_sim = stats_sim.numpy()
    
    # Get obs stats
    x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
    if x_obs_cpu.ndim == 2: x_obs_cpu = x_obs_cpu[np.newaxis, ...]
    
    # Ensure float32
    x_obs_cpu = x_obs_cpu.astype(np.float32)
    stats_obs = model.summary_network(x_obs_cpu)
    if hasattr(stats_obs, "cpu"): stats_obs = stats_obs.cpu()
    if hasattr(stats_obs, "numpy"): stats_obs = stats_obs.numpy()
    
    # Distances
    diff = stats_sim - stats_obs
    dist = np.linalg.norm(diff, axis=1)
    
    epsilon = np.quantile(dist, quantile_level)
    print(f"Computed bandwidth (epsilon): {epsilon}")
    return epsilon

# Backwards compatibility / Dispatcher
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

# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(samples, theta_true):
    """
    Compute Bias, 95% HDI, and Coverage.
    
    Args:
        samples: (n_samples, d) approximate posterior samples
        theta_true: (d,) ground truth parameters
    
    Returns:
        dict with keys:
            'bias': (d,) L2 distance between mean and true
            'hdi_lower': (d,) 2.5% quantile
            'hdi_upper': (d,) 97.5% quantile
            'hdi_length': (d,) upper - lower
            'coverage': (d,) 1 if true in HDI, 0 otherwise (as float)
    """
    # 1. Bias (Mean - True)
    # L2 distance between mean and true parameters.
    
    mean_est = np.mean(samples, axis=0)
    bias_vec = mean_est - theta_true
    bias_l2 = np.linalg.norm(bias_vec) # Scalar L2 distance
    
    # 2. 95% HDI (Credible Interval)
    # Using quantiles [0.025, 0.975]
    lower = np.quantile(samples, 0.025, axis=0)
    upper = np.quantile(samples, 0.975, axis=0)
    length = upper - lower
    
    # 3. Coverage
    # Check if theta_true is within [lower, upper]
    covered = (theta_true >= lower) & (theta_true <= upper)
    
    return {
        "bias_l2": bias_l2,
        "bias_vec": bias_vec,
        "hdi_lower": lower,
        "hdi_upper": upper,
        "hdi_length": length,
        "coverage": covered.astype(float)
    }
