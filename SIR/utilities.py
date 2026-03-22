import torch
import numpy as np
import time
from sklearn.neighbors import KernelDensity


def fit_kde_and_evaluate(samples, evaluate_points):
    n, d = samples.shape
    bandwidth = n ** (-1.0 / (d + 4))
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(samples)
    log_density = kde.score_samples(evaluate_points)
    return log_density


def run_mcmc_refinement(
    current_theta,
    x_obs_stats,
    task,
    log_prior_fn,
    likelihood_fn,
    n_chains,
    n_samples,
    burn_in,
    thin,
    proposal_std=0.5,
):
    current_theta = np.clip(current_theta, task.lower, task.upper)
    std_per_dim = np.std(current_theta, axis=0)
    std_per_dim = np.maximum(std_per_dim, 1e-6)
    proposal_scale = std_per_dim * proposal_std

    current_log_prior = log_prior_fn(current_theta)
    
    # Initialize current log_prob with -inf
    current_log_prob = np.full(n_chains, -np.inf)
    
    # Check for valid priors
    valid_mask_curr = np.isfinite(current_log_prior)
    
    if np.any(valid_mask_curr):
        theta_valid_curr = current_theta[valid_mask_curr]
        lik_valid_curr = likelihood_fn(theta_valid_curr, x_obs_stats)
        log_lik_valid_curr = np.log(lik_valid_curr + 1e-300)
        current_log_prob[valid_mask_curr] = current_log_prior[valid_mask_curr] + log_lik_valid_curr
    else:
        # If all initial points are invalid, warn but don't crash
        print("[Refine] WARNING: All initial theta points have zero prior probability!")

    print(
        f"[Refine] Starting ABC-MCMC with n_chains={n_chains}, "
        f"burn_in={burn_in}, n_samples={n_samples}, thin={thin}"
    )
    print(f"[Refine] Proposal std per dim: {proposal_scale}")
    samples = []
    total_accepted = 0
    total_sampling_steps = n_samples * thin
    total_steps = burn_in + total_sampling_steps

    t_start = time.time()
    for step in range(1, total_steps + 1):
        proposal_noise = np.random.randn(n_chains, task.d) * proposal_scale
        proposed_theta = current_theta + proposal_noise
        proposed_log_prior = log_prior_fn(proposed_theta)
        
        # Initialize log_prob with -inf
        proposed_log_prob = np.full(n_chains, -np.inf)
        
        # Check for valid priors (finite log prior)
        valid_mask = np.isfinite(proposed_log_prior)
        
        # Compute likelihood only for valid proposals to avoid simulator crashes
        if np.any(valid_mask):
            # Extract valid parameters
            theta_valid = proposed_theta[valid_mask]
            
            # Compute likelihood
            lik_valid = likelihood_fn(theta_valid, x_obs_stats)
            log_lik_valid = np.log(lik_valid + 1e-300)
            
            # Update log_prob for valid entries
            proposed_log_prob[valid_mask] = proposed_log_prior[valid_mask] + log_lik_valid
            
        log_ratio = proposed_log_prob - current_log_prob
        log_u = np.log(np.random.rand(n_chains))
        accept_mask = log_u < log_ratio
        current_theta[accept_mask] = proposed_theta[accept_mask]
        current_log_prob[accept_mask] = proposed_log_prob[accept_mask]
        if step > burn_in:
            total_accepted += np.sum(accept_mask)
            if (step - burn_in) % thin == 0:
                samples.append(current_theta.copy())
        if step == burn_in:
            print(f"[Refine] Finished burn-in at step={step}")
        if step % max(1, total_steps // 5) == 0 or step == total_steps:
            elapsed = time.time() - t_start
            if step > burn_in:
                eff_steps = step - burn_in
                eff_total = n_chains * eff_steps
                acc_rate_now = total_accepted / eff_total if eff_total > 0 else 0.0
            else:
                acc_rate_now = 0.0
            print(
                f"[Refine] Step {step}/{total_steps}, "
                f"accepted so far rate≈{acc_rate_now:.3f}, elapsed={elapsed:.1f}s"
            )

    if not samples:
        samples.append(current_theta.copy())

    posterior_samples = np.vstack(samples)
    acceptance_rate = (
        total_accepted / (n_chains * total_sampling_steps)
        if total_sampling_steps > 0
        else 0
    )
    print(
        f"[Refine] Finished ABC-MCMC: total_steps={total_steps}, "
        f"acceptance_rate={acceptance_rate:.3f}, "
        f"num_samples={posterior_samples.shape[0]}"
    )
    return posterior_samples


def compute_bandwidth_core(
    theta0_np, x_obs_tensor, task, stats_fn, n_samples, quantile_level, device
):
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
    return quan1


def approximate_likelihood_core(theta, x_obs_stats, task, stats_fn, epsilon, device):
    sim_data = task.simulator(theta, n_samples=task.n_obs)
    sim_data_tensor = torch.from_numpy(sim_data).float().to(device)

    with torch.no_grad():
        sim_stats = stats_fn(sim_data_tensor)
        diff = sim_stats - x_obs_stats
        dist_sq = torch.sum(diff**2, dim=-1)
        likelihood = torch.exp(-dist_sq / (2 * epsilon**2))

    return likelihood.cpu().numpy()


def compute_metrics(theta_samples, theta_true, ci=0.95):
    if isinstance(theta_samples, torch.Tensor):
        theta_samples = theta_samples.detach().cpu().numpy()
    theta_true = np.asarray(theta_true, dtype=np.float32)
    if theta_true.ndim == 1:
        theta_true = theta_true[np.newaxis, :]
    theta_true = theta_true[0]

    mean_est = np.mean(theta_samples, axis=0)
    bias_l2 = np.linalg.norm(mean_est - theta_true)

    lower_q = (1.0 - ci) / 2.0
    upper_q = 1.0 - lower_q
    hdi_lower = np.quantile(theta_samples, lower_q, axis=0)
    hdi_upper = np.quantile(theta_samples, upper_q, axis=0)
    hdi_length = hdi_upper - hdi_lower

    coverage = (theta_true >= hdi_lower) & (theta_true <= hdi_upper)
    coverage = coverage.astype(np.float32)

    return {
        "bias_l2": bias_l2,
        "hdi_length": hdi_length,
        "coverage": coverage,
    }


def refine_posterior_smmd(
    model,
    x_obs,
    task,
    n_chains=1000,
    n_samples=1,
    burn_in=99,
    thin=1,
    epsilon=None,
    proposal_std=0.2,
    device=None,
    stats_fn=None,
    bandwidth_n_samples=5000,
):
    print("[Refine] Using SMMD/MMD refine_posterior_smmd")
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

    if isinstance(x_obs, torch.Tensor):
        x_obs_tensor = x_obs.float().to(device)
    else:
        x_obs_tensor = torch.from_numpy(x_obs).float().to(device)

    if x_obs_tensor.ndim == 2:
        x_obs_tensor = x_obs_tensor.unsqueeze(0)

    if stats_fn is None:
        stats_fn = model.T

    if epsilon is None:
        print(
            f"[Refine] Estimating epsilon with {bandwidth_n_samples} prior samples "
            f"for d={task.d}"
        )
        theta0_np = np.random.uniform(
            low=task.lower, high=task.upper, size=(bandwidth_n_samples, task.d)
        ).astype(np.float32)
        epsilon = compute_bandwidth_core(
            theta0_np,
            x_obs_tensor,
            task,
            stats_fn,
            bandwidth_n_samples,
            0.02,
            device,
        )
        print(f"[Refine] Estimated epsilon={epsilon:.4f}")

    with torch.no_grad():
        x_obs_stats = stats_fn(x_obs_tensor)
        z = torch.randn(n_chains, 1, task.d, device=device)
        current_theta = (
            model.G(z, x_obs_stats.expand(n_chains, -1)).squeeze(1).cpu().numpy()
        )
    print(
        f"[Refine] Initial theta for MCMC has shape={current_theta.shape}, "
        f"std_per_dim={np.std(current_theta, axis=0)}"
    )

    def likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(theta, target_stats, task, stats_fn, epsilon, device)

    samples = run_mcmc_refinement(
        current_theta,
        x_obs_stats,
        task,
        task.log_prior,
        likelihood_fn,
        n_chains,
        n_samples,
        burn_in,
        thin,
        proposal_std,
    )
    print(f"[Refine] Returned {samples.shape[0]} refined samples from SMMD/MMD")
    return samples


def refine_posterior_bayesflow(
    model,
    x_obs,
    task,
    n_chains=1000,
    n_samples=1,
    burn_in=99,
    thin=1,
    epsilon=None,
    proposal_std=0.2,
    device=None,
    theta_init=None,
    bandwidth_n_samples=5000,
):
    print("[Refine] Using BayesFlow refine_posterior_bayesflow")
    if device is None:
        device = "cpu"

    if epsilon is None:
        print(
            f"[Refine] Estimating epsilon for BayesFlow with "
            f"{bandwidth_n_samples} prior samples"
        )
        epsilon = compute_bandwidth_bayesflow(
            model, x_obs, task, n_samples=bandwidth_n_samples, device=device
        )
        print(f"[Refine] Estimated epsilon={epsilon:.4f}")

    x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
    if x_obs_cpu.ndim == 2:
        x_obs_cpu = x_obs_cpu[np.newaxis, ...]

    if theta_init is not None:
        current_theta = theta_init
        if current_theta.shape[0] != n_chains:
            n_chains = current_theta.shape[0]
    else:
        x_obs_rep = np.tile(x_obs_cpu, (n_chains, 1, 1))
        post = model.sample(conditions={"summary_variables": x_obs_rep}, num_samples=1)
        if isinstance(post, dict):
            post = post["inference_variables"]
        if hasattr(post, "numpy"):
            post = post.numpy()
        current_theta = np.asarray(post).reshape(n_chains, -1)
        if isinstance(current_theta, torch.Tensor):
            current_theta = current_theta.cpu().numpy()
    print(
        f"[Refine] Initial theta for BayesFlow MCMC has shape={current_theta.shape}, "
        f"std_per_dim={np.std(current_theta, axis=0)}"
    )

    x_obs_tensor = torch.from_numpy(x_obs_cpu).float().to(device)

    def stats_fn_torch(x_tensor):
        x_np = x_tensor.cpu().numpy().astype(np.float32)
        stats = model.summary_network(x_np)
        if hasattr(stats, "cpu"):
            stats = stats.cpu()
        if hasattr(stats, "numpy"):
            stats = stats.numpy()
        return torch.from_numpy(stats).float().to(device)

    x_obs_cpu = x_obs_cpu.astype(np.float32)
    x_obs_stats_tf = model.summary_network(x_obs_cpu)
    if hasattr(x_obs_stats_tf, "cpu"):
        x_obs_stats_tf = x_obs_stats_tf.cpu()
    if hasattr(x_obs_stats_tf, "numpy"):
        x_obs_stats_tf = x_obs_stats_tf.numpy()
    x_obs_stats = torch.from_numpy(x_obs_stats_tf).float().to(device)

    def likelihood_fn(theta, target_stats):
        return approximate_likelihood_core(
            theta, target_stats, task, stats_fn_torch, epsilon, device
        )

    samples = run_mcmc_refinement(
        current_theta,
        x_obs_stats,
        task,
        task.log_prior,
        likelihood_fn,
        n_chains,
        n_samples,
        burn_in,
        thin,
        proposal_std,
    )
    print(f"[Refine] Returned {samples.shape[0]} refined samples from BayesFlow")
    return samples


def compute_bandwidth_bayesflow(
    model, x_obs, task, n_samples=5000, quantile_level=0.02, device="cpu"
):
    theta_prior = task.sample_prior(n_samples)
    x_sim = task.simulator(theta_prior)
    x_sim_float = x_sim.astype(np.float32)
    stats_sim = model.summary_network(x_sim_float)
    if hasattr(stats_sim, "cpu"):
        stats_sim = stats_sim.cpu()
    if hasattr(stats_sim, "numpy"):
        stats_sim = stats_sim.numpy()

    x_obs_cpu = x_obs if isinstance(x_obs, np.ndarray) else x_obs.cpu().numpy()
    if x_obs_cpu.ndim == 2:
        x_obs_cpu = x_obs_cpu[np.newaxis, ...]
    x_obs_cpu = x_obs_cpu.astype(np.float32)
    stats_obs = model.summary_network(x_obs_cpu)
    if hasattr(stats_obs, "cpu"):
        stats_obs = stats_obs.cpu()
    if hasattr(stats_obs, "numpy"):
        stats_obs = stats_obs.numpy()

    diff = stats_sim - stats_obs
    dist = np.linalg.norm(diff, axis=1)
    epsilon = np.quantile(dist, quantile_level)
    return epsilon


def refine_posterior(model, *args, **kwargs):
    if hasattr(model, "summary_network"):
        return refine_posterior_bayesflow(model, *args, **kwargs)
    return refine_posterior_smmd(model, *args, **kwargs)


def compute_bandwidth_torch(model, *args, **kwargs):
    if hasattr(model, "summary_network"):
        return compute_bandwidth_bayesflow(model, *args, **kwargs)
    return compute_bandwidth_core(model, *args, **kwargs)
