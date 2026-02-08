import os
import numpy as np
import torch
import pyabc
import ot
import scipy.stats as stats
import logging

# Set pyabc logging to avoid clutter
logging.getLogger("pyabc").setLevel(logging.WARNING)

class W2Distance(pyabc.Distance):
    """
    Wasserstein-2 Distance using POT (Python Optimal Transport).
    Computes exact W2 between two empirical distributions in R^d.
    """
    def __call__(self, x, y, t=None, par=None):
        # x, y are dicts {'data': np.array (n, d_x)}
        X = x['data']
        Y = y['data']
        
        # Ensure numpy
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        # Uniform weights
        n = X.shape[0]
        m = Y.shape[0]
        a = np.ones((n,)) / n
        b = np.ones((m,)) / m
        
        # Cost matrix: squared euclidean distance
        # X: (n, d), Y: (m, d)
        # c_ij = ||x_i - y_j||^2
        M = ot.dist(X, Y, metric='sqeuclidean')
        
        # emd2 returns the transport cost sum(T_ij * M_ij)
        # If metric is sqeuclidean, this is W2^2
        w2_sq = ot.emd2(a, b, M)
        
        return np.sqrt(w2_sq)

class SummaryDistance(pyabc.Distance):
    """
    Euclidean Distance on Summary Statistics.
    d(x, y) = ||S(x) - S(y)||_2
    """
    def __init__(self, summary_network, device="cpu"):
        self.summary_network = summary_network
        self.device = device

    def __call__(self, x, y, t=None, par=None):
        # x, y are dicts {'data': np.array (n, d_x)}
        X = x['data']
        Y = y['data']
        
        # Prepare for network
        # Network expects (batch, n, d_x)
        # We need to handle single instance
        
        def get_summary(data):
            # data: (n, d_x)
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Check if summary_network is Keras or Torch
            if hasattr(self.summary_network, "predict"): # Keras/TF (BayesFlow usually)
                 # Keras expect numpy or tensor. BayesFlow networks usually take dict or tensor.
                 # Based on load_models.py, bayesflow model is Keras. 
                 # model.summary_network(x)
                 # We might need to convert to numpy if it's Keras
                 data_np = data_tensor.cpu().numpy()
                 # BayesFlow summary net often takes (batch, n, dx)
                 # Return shape (batch, summary_dim)
                 s = self.summary_network(data_np)
                 if hasattr(s, "cpu"):
                     s = s.cpu()
                 if hasattr(s, "numpy"):
                     s = s.numpy()
                 return torch.tensor(s).squeeze(0) # (summary_dim,)
                 
            else: # Torch
                with torch.no_grad():
                    s = self.summary_network(data_tensor)
                return s.cpu().squeeze(0) # (summary_dim,)

        sx = get_summary(X)
        sy = get_summary(Y)
        
        # Euclidean distance
        return torch.norm(sx - sy).item()

def run_smc_abc(task, x_obs, n_samples=1000, max_populations=10, 
                result_dir="results", distance_metric="w2", summary_network=None, device="cpu"):
    """
    Run ABC using PyABC SMC.
    
    Args:
        task: GaussianTask instance
        x_obs: Observation (n, d_x)
        n_samples: Population size
        max_populations: Max generations
        distance_metric: "w2" or "summary"
        summary_network: Required if distance_metric="summary"
    """
    metric_name = distance_metric.upper()
    print(f"\n=== Running {metric_name}-ABC (Population={n_samples}, Max Gen={max_populations}) ===")
    
    # 1. Define Prior
    prior_dict = {}
    for i in range(task.d):
        loc = task.lower[i]
        scale = task.upper[i] - task.lower[i]
        prior_dict[f"theta{i}"] = pyabc.RV("uniform", loc, scale)
        
    prior = pyabc.Distribution(**prior_dict)
    
    # 2. Define Model Wrapper
    def model_wrapper(params):
        theta = np.zeros(task.d)
        for i in range(task.d):
            theta[i] = params[f"theta{i}"]
        x_sim = task.simulator(theta[np.newaxis, :])[0]
        return {'data': x_sim}
        
    # 3. Define Distance
    if distance_metric == "w2":
        distance = W2Distance()
    elif distance_metric == "summary":
        if summary_network is None:
            raise ValueError("summary_network must be provided for summary distance")
        distance = SummaryDistance(summary_network, device)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # 4. Setup ABC SMC
    abc = pyabc.ABCSMC(
        models=model_wrapper,
        parameter_priors=prior,
        distance_function=distance,
        population_size=n_samples,
        sampler=pyabc.sampler.SingleCoreSampler()
    )
    
    # 5. Prepare Observation
    if x_obs.ndim == 3:
        x_obs_data = x_obs[0]
    else:
        x_obs_data = x_obs
        
    # Database
    os.makedirs(result_dir, exist_ok=True)
    db_name = f"{distance_metric}abc_temp.db"
    db_path = os.path.join(result_dir, db_name)
    if os.path.exists(db_path):
        os.remove(db_path)
    db_url = "sqlite:///" + db_path
    
    abc.new(db_url, {'data': x_obs_data})
    
    # 6. Run
    history = abc.run(max_nr_populations=max_populations)
    
    # 7. Extract Samples
    df, w = history.get_distribution(m=0, t=history.max_t)
    
    samples = np.zeros((len(df), task.d))
    for i in range(task.d):
        samples[:, i] = df[f"theta{i}"].values
        
    print(f"Resampling {metric_name}-ABC posterior...")
    indices = np.random.choice(len(df), size=n_samples, p=w/w.sum(), replace=True)
    final_samples = samples[indices]
    
    print(f"{metric_name}-ABC Completed. Max t={history.max_t}")
    
    return final_samples

# Backward compatibility wrapper
def run_w2abc(task, x_obs, n_samples=1000, max_populations=10, result_dir="results"):
    return run_smc_abc(task, x_obs, n_samples, max_populations, result_dir, distance_metric="w2")
