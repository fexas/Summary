import torch
import numpy as np
from sbi.inference import NPE_A, NPE_B, SNPE, NPE
from sbi.utils import BoxUniform


def calculate_summary_statistics(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)


def run_sbi_model(
    model_type,
    train_loader,
    x_obs,
    theta_true,
    task,
    device="cpu",
    num_rounds=1,
    sims_per_round=1000,
    max_epochs=1000,
    **kwargs,
):
    print(f"=== Running {model_type.upper()} (Rounds={num_rounds}, Sims/Round={sims_per_round}, MaxEpochs={max_epochs}) ===")

    if isinstance(x_obs, np.ndarray):
        x_obs = torch.from_numpy(x_obs).float()

    if x_obs.ndim == 2:
        x_obs_input = x_obs.unsqueeze(0)
    elif x_obs.ndim == 3:
        x_obs_input = x_obs
    else:
        raise ValueError(f"Expected x_obs shape (n, d) or (1, n, d), got {x_obs.shape}")

    x_obs_stats = calculate_summary_statistics(x_obs_input)
    print(f"Observation Summary Stats Shape: {x_obs_stats.shape}")

    low = torch.tensor(task.lower, dtype=torch.float32)
    high = torch.tensor(task.upper, dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)

    device_sbi = "cpu"

    model_type_lower = model_type.lower()
    if model_type_lower == "snpe_a":
        inference = NPE_A(prior=prior, device=device_sbi)
    elif model_type_lower == "snpe_b":
        inference = NPE_B(prior=prior, device=device_sbi)
    elif model_type_lower in ["snpe_c", "npe", "snpe"]:
        inference = NPE(prior=prior, device=device_sbi)
    else:
        print(f"Warning: Unknown model_type '{model_type}'. Defaulting to NPE (SNPE-C).")
        inference = NPE(prior=prior, device=device_sbi)

    proposal = prior

    for r in range(num_rounds):
        print(f"--- SBI Round {r+1}/{num_rounds} ---")

        theta = proposal.sample((sims_per_round,))

        theta_np = theta.cpu().numpy()
        x_sim_np = task.simulator(theta_np)
        x_sim_torch = torch.from_numpy(x_sim_np).float()

        x = calculate_summary_statistics(x_sim_torch)

        theta = theta.to(device_sbi)
        x = x.to(device_sbi)

        train_kwargs = {"max_num_epochs": max_epochs}
        if isinstance(inference, NPE_A):
            final_round = r == num_rounds - 1
            train_kwargs["final_round"] = final_round

        with torch.enable_grad():
            inference = inference.append_simulations(theta, x, proposal=proposal)
            density_estimator = inference.train(**train_kwargs)

        posterior = inference.build_posterior(density_estimator).set_default_x(x_obs_stats)
        proposal = posterior

    print("Sampling from final posterior...")

    n_samples = 2000
    samples = posterior.sample((n_samples,))

    return samples.detach().cpu().numpy()
