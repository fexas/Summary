import math
import torch
import numpy as np
import torch.distributions as D
from sbi.inference import NPE_A, NPE_B, SNPE, NPE
from sbi.utils import BoxUniform


def _flatten_trajectory(x):
    """
    Flatten SIR trajectories into feature vectors for SBI.

    Args:
        x: array/tensor of shape (batch, n_points, d_x) or (n_points, d_x)

    Returns:
        Tensor of shape (batch, n_points * d_x)
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected x with 2 or 3 dims, got shape {x.shape}")
    batch_size = x.shape[0]
    return x.view(batch_size, -1)


def run_sbi_model(model_type, train_loader, x_obs, theta_true, task, device="cpu",
                  num_rounds=1, sims_per_round=1000, max_epochs=1000, **kwargs):
    print(f"=== Running {model_type.upper()} (Rounds={num_rounds}, Sims/Round={sims_per_round}, MaxEpochs={max_epochs}) ===")
    if isinstance(x_obs, np.ndarray):
        x_obs = torch.from_numpy(x_obs).float()
    x_obs_flat = _flatten_trajectory(x_obs)
    # Log-normal prior consistent with SIRTask.sample_prior
    loc = torch.tensor(
        [math.log(0.4), math.log(1.0 / 8.0)], dtype=torch.float32
    )
    scale = torch.tensor([0.5, 0.2], dtype=torch.float32)
    prior = D.Independent(D.LogNormal(loc=loc, scale=scale), 1)
    device_sbi = "cpu"
    model_type_lower = model_type.lower()
    if model_type_lower == "snpe_a":
        inference = NPE_A(prior=prior, device=device_sbi)
    elif model_type_lower == "snpe_b":
        inference = NPE_B(prior=prior, device=device_sbi)
    elif model_type_lower in ["snpe_c", "npe", "snpe"]:
        inference = NPE(prior=prior, device=device_sbi)
    else:
        inference = NPE(prior=prior, device=device_sbi)
    proposal = prior
    initial_training_data = kwargs.pop("initial_training_data", None)
    for r in range(num_rounds):
        if r == 0 and initial_training_data is not None:
            x_train_np, theta_train_np = initial_training_data
            n0 = min(sims_per_round, theta_train_np.shape[0])
            theta_np = theta_train_np[:n0].astype(np.float32)
            x_sim_np = x_train_np[:n0].astype(np.float32)
            theta = torch.from_numpy(theta_np)
            x_sim_torch = torch.from_numpy(x_sim_np)
        else:
            theta = proposal.sample((sims_per_round,))
            theta_np = theta.cpu().numpy()
            x_sim_np = task.simulator(theta_np)
            x_sim_torch = torch.from_numpy(x_sim_np).float()
        x = _flatten_trajectory(x_sim_torch)
        theta = theta.to(device_sbi)
        x = x.to(device_sbi)
        train_kwargs = {"max_num_epochs": max_epochs}
        if isinstance(inference, NPE_A):
            final_round = r == num_rounds - 1
            train_kwargs["final_round"] = final_round
        with torch.enable_grad():
            inference = inference.append_simulations(theta, x, proposal=proposal)
            density_estimator = inference.train(**train_kwargs)
        posterior = inference.build_posterior(density_estimator, sample_with='mcmc').set_default_x(x_obs_flat[0])
        proposal = posterior
    n_samples = 1000
    samples = posterior.sample((n_samples,))
    return samples.detach().cpu().numpy()
