import torch
import numpy as np
from sbi.inference import NPE_A, NPE_B, SNPE
from sbi.utils import BoxUniform

def calculate_summary_statistics(x):
    """
    For Gaussian task, we use the flattened raw data as summary statistics,
    following the reference implementation in snl-master.
    
    Input x: (batch_size, n_samples, d_x)
    Output: (batch_size, n_samples * d_x)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        
    # Handle single observation case (n_samples, d_x) -> (1, n_samples, d_x)
    if x.ndim == 2:
        x = x.unsqueeze(0)
        
    batch_size = x.shape[0]
    # Flatten the last two dimensions
    return x.reshape(batch_size, -1)

def run_sbi_model(model_type, train_loader, x_obs, theta_true, task, device="cpu", num_rounds=1, sims_per_round=1000, max_epochs=1000, **kwargs):
    """
    Run SBI training and inference using SNPE-A logic (or others).
    
    Args:
        model_type: 'snpe_a', 'snpe_b', or 'npe'
        train_loader: Ignored for SBI as it generates its own simulations on the fly (sequential) 
                      or we could use it for offline, but SNPE usually does sequential. 
                      Here we assume sequential generation using task.simulator.
        x_obs: Observation (n_samples, d_x) or (1, n_samples, d_x)
        theta_true: True parameters (for reference, not used in training)
        task: GaussianTask instance
        device: 'cpu' or 'cuda' (MPS support depends on SBI/PyTorch version)
        num_rounds: Number of SNPE rounds
        sims_per_round: Simulations per round
        max_epochs: Max training epochs per round
    """
    print(f"=== Running {model_type.upper()} (Rounds={num_rounds}, Sims/Round={sims_per_round}, MaxEpochs={max_epochs}) ===")
    
    # 0. Prepare Observation (x_obs)
    if isinstance(x_obs, np.ndarray):
        x_obs = torch.from_numpy(x_obs).float()
    
    # Ensure x_obs is 3D (1, n, d) for consistency before flattening
    if x_obs.ndim == 2:
        x_obs_input = x_obs.unsqueeze(0)
    elif x_obs.ndim == 3:
        x_obs_input = x_obs
    else:
        # If it's already flattened? Assume not based on previous code flow
        raise ValueError(f"Expected x_obs shape (n, d) or (1, n, d), got {x_obs.shape}")

    # Calculate 'summary stats' (flattened data)
    x_obs_stats = calculate_summary_statistics(x_obs_input)
    print(f"Observation Summary Stats Shape: {x_obs_stats.shape}")

    # 1. Define Prior
    # Use task bounds for prior (BoxUniform)
    low = torch.tensor(task.lower, dtype=torch.float32)
    high = torch.tensor(task.upper, dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)
    
    # 2. Instantiate Inference Object
    # SBI on MPS can be tricky, fallback to CPU if needed, but let's try passing device.
    # If device is 'mps', check if sbi supports it. Safest is CPU for SBI often.
    # User instructions didn't specify, but existing code uses DEVICE.
    # Let's default to cpu for sbi to avoid potential "NotImplemented" for some ops.
    device_sbi = "cpu" 
    
    if model_type.lower() == 'snpe_a' or model_type.lower() == 'npe':
        inference = NPE_A(prior=prior, device=device_sbi)
    elif model_type.lower() == 'snpe_b':
        inference = NPE_B(prior=prior, device=device_sbi)
    elif model_type.lower() == 'snpe_c':
        inference = SNPE(prior=prior, device=device_sbi)
    else:
        # Fallback to NPE_A
        inference = NPE_A(prior=prior, device=device_sbi)

    # 3. Sequential Inference Loop
    proposal = prior
    
    for r in range(num_rounds):
        print(f"--- SBI Round {r+1}/{num_rounds} ---")
        
        # A. Simulate Data
        theta = proposal.sample((sims_per_round,))
        
        theta_np = theta.cpu().numpy()
        x_sim_np = task.simulator(theta_np) # Returns (N, n, d)
        x_sim_torch = torch.from_numpy(x_sim_np).float()
        
        # Calculate summary stats (flatten)
        x = calculate_summary_statistics(x_sim_torch)
        
        # Move to device
        theta = theta.to(device_sbi)
        x = x.to(device_sbi)
        
        # B. Train
        final_round = (r == num_rounds - 1)
        
        # Ensure autograd is enabled (BayesFlow/others might disable it globally)
        with torch.enable_grad():
            density_estimator = inference.append_simulations(theta, x, proposal=proposal).train(
                max_num_epochs=max_epochs, 
                final_round=final_round
            )
            
        # C. Build Posterior & Update Proposal
        posterior = inference.build_posterior(density_estimator).set_default_x(x_obs_stats)
        proposal = posterior
        
    # 4. Final Sampling
    print("Sampling from final posterior...")
    
    n_samples = 2000 # Default sample count
    samples = posterior.sample((n_samples,))
    
    return samples.detach().cpu().numpy()
