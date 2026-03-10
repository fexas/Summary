import os
import json
import numpy as np
import torch
import scipy.stats as stats

from bayesflow.simulators.benchmark_simulators.sir import SIR as BayesFlowSIR

N_DEFAULT = 1000000.0
T_MAX_DEFAULT = 160.0
NUM_OBS_DEFAULT = 10

config_path = os.path.join(os.path.dirname(__file__), "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    N = float(config.get("N", N_DEFAULT))
    T_MAX = float(config.get("T_MAX", T_MAX_DEFAULT))
    NUM_OBS = int(config.get("NUM_OBS", NUM_OBS_DEFAULT))
else:
    N = N_DEFAULT
    T_MAX = T_MAX_DEFAULT
    NUM_OBS = NUM_OBS_DEFAULT

OBS_TIMES = np.linspace(0, T_MAX, NUM_OBS)

d = 2
d_x = 1

PRIOR_MIN = np.array([0.01, 0.01])
PRIOR_MAX = np.array([1.0, 1.0])

TRUE_PARAMS = np.array([0.20, 0.12])  # TRUE_PARAMS = np.array([0.12, 0.115])


def simulator(theta, n_points=NUM_OBS, normalize=True):
    """
    Simulates a batch of trajectories with the BayesFlow SIR benchmark simulator.

    theta: (batch_size, d) or (d,)
    Returns: (batch_size, n_points, d_x) with d_x = 1 (infected fraction or scaled count).
    """
    theta = np.atleast_2d(theta).astype(np.float32)
    batch_size = theta.shape[0]
    output = np.zeros((batch_size, n_points, d_x), dtype=np.float32)

    rng = np.random.default_rng()
    sir = BayesFlowSIR(
        N=N,
        T=int(T_MAX),
        subsample=n_points,
        total_count=1000,
        scale_by_total=True,
        rng=rng,
    )

    for i in range(batch_size):
        params = theta[i]
        x = sir.observation_model(params)
        if x.ndim == 1:
            x = x[:, None]
        if normalize:
            output[i, :, 0] = x.squeeze(-1).astype(np.float32)
        else:
            output[i, :, 0] = (x.squeeze(-1) * N / 1000.0).astype(np.float32)

    return output

def sample_prior(batch_size):
    beta = np.random.lognormal(mean=np.log(0.4), sigma=0.5, size=(batch_size, 1)).astype(
        np.float32
    )
    gamma = np.random.lognormal(mean=np.log(1.0 / 8.0), sigma=0.2, size=(batch_size, 1)).astype(
        np.float32
    )
    theta = np.concatenate([beta, gamma], axis=1)
    return theta

def calculate_summary_statistics(x):
    """
    x: (batch, n_points, 2) where col 0 is I/N, col 1 is t/T
    Returns: (batch, 4) [Max I, Time of Max I, Final I, Mean I]
    """
    # x is tensor or numpy
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
        
    # Extract I component (batch, n_points)
    I_curve = x[..., 0] 
    t_curve = x[..., 1]
    
    # 1. Max Infection
    max_I, max_idx = torch.max(I_curve, dim=1)
    
    # 2. Time of Max Infection
    # Gather time corresponding to max_idx
    # t_curve is same for all usually, but x might vary?
    # x shape (batch, n, 2). t is x[:, :, 1]
    time_of_max = torch.gather(t_curve, 1, max_idx.unsqueeze(1)).squeeze(1)
    
    # 3. Final Infection Level (Endemic state?)
    final_I = I_curve[:, -1]
    mean_I = torch.mean(I_curve, dim=1)
    
    return torch.stack([max_I, time_of_max, final_I, mean_I], dim=1)

class SIRTask:
    def __init__(self, N_val=N, t_max=T_MAX, num_obs=NUM_OBS):
        self.N = float(N_val)
        self.t_max = float(t_max)
        self.n_obs = int(num_obs)
        self.d = d
        self.d_x = d_x
        self.lower = PRIOR_MIN.astype(np.float32)
        self.upper = PRIOR_MAX.astype(np.float32)
        self.theta_true = TRUE_PARAMS.astype(np.float32)
        self.obs_times = OBS_TIMES
        self._rng = np.random.default_rng()
        self._sir = BayesFlowSIR(
            N=self.N,
            T=int(self.t_max),
            subsample=self.n_obs,
            total_count=100,
            scale_by_total=True,
            rng=self._rng,
        )
    
    def get_ground_truth(self):
        theta = self.theta_true
        theta_batch = theta[np.newaxis, :]
        obs_batch = self.simulator(theta_batch, n_samples=self.n_obs, normalize=True)
        return theta, obs_batch[0]
    
    def sample_prior(self, batch_size):
        return sample_prior(batch_size)
    
    def log_prior(self, theta):
        theta = np.asarray(theta, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta[np.newaxis, :]
        beta = theta[:, 0]
        gamma = theta[:, 1]
        logp_beta = stats.lognorm(s=0.5, scale=0.4).logpdf(beta)
        logp_gamma = stats.lognorm(s=0.2, scale=1.0 / 8.0).logpdf(gamma)
        logp = logp_beta + logp_gamma
        if logp.shape[0] == 1:
            return logp[0]
        return logp
    
    def simulator(self, theta, n_samples=None, normalize=True):
        n_points = self.n_obs if n_samples is None else n_samples
        theta = np.atleast_2d(theta).astype(np.float32)
        batch_size = theta.shape[0]
        output = np.zeros((batch_size, n_points, self.d_x), dtype=np.float32)
        for i in range(batch_size):
            params = theta[i]
            x = self._sir.observation_model(params)
            if x.ndim == 1:
                x = x[:, None]
            if normalize:
                output[i, :, 0] = x.squeeze(-1).astype(np.float32)
            else:
                output[i, :, 0] = (x.squeeze(-1) * self.N / 1000.0).astype(np.float32)
        return output
