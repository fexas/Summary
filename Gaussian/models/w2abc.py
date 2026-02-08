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

def run_w2abc(task, x_obs, n_samples=1000, max_populations=10, result_dir="results"):
    """
    Run W2-ABC using PyABC SMC.
    
    Args:
        task: GaussianTask instance
        x_obs: Observation (n, d_x)
        n_samples: Population size (number of particles)
        max_populations: Max number of SMC generations
        
    Returns:
        samples: (n_samples, d) numpy array
    """
    print(f"\n=== Running W2-ABC (Population={n_samples}, Max Gen={max_populations}) ===")
    
    # 1. Define Prior
    # GaussianTask uses Uniform[lower, upper]
    # keys: theta0, theta1, ...
    prior_dict = {}
    for i in range(task.d):
        # pyabc uses scipy.stats distributions
        # Uniform(loc, scale) -> [loc, loc+scale]
        loc = task.lower[i]
        scale = task.upper[i] - task.lower[i]
        prior_dict[f"theta{i}"] = pyabc.RV("uniform", loc, scale)
        
    prior = pyabc.Distribution(**prior_dict)
    
    # 2. Define Model Wrapper
    def model_wrapper(params):
        # params is dict {'theta0': v0, ...}
        # Convert to array (1, d)
        theta = np.zeros(task.d)
        for i in range(task.d):
            theta[i] = params[f"theta{i}"]
            
        # Simulate
        # task.simulator returns (1, n, d_x)
        x_sim = task.simulator(theta[np.newaxis, :])[0]
        
        return {'data': x_sim}
        
    # 3. Define Distance
    distance = W2Distance()
    
    # 4. Setup ABC SMC
    # Use SingleCoreSampler for simplicity and to avoid multiprocessing issues in some envs
    # Or MulticoreEvalParallelSampler if needed
    abc = pyabc.ABCSMC(
        models=model_wrapper,
        parameter_priors=prior,
        distance_function=distance,
        population_size=n_samples,
        sampler=pyabc.sampler.SingleCoreSampler()
    )
    
    # 5. Prepare Observation
    # x_obs might be (1, n, d) or (n, d)
    if x_obs.ndim == 3:
        x_obs_data = x_obs[0]
    else:
        x_obs_data = x_obs
        
    # Database
    os.makedirs(result_dir, exist_ok=True)
    db_path = os.path.join(result_dir, "w2abc_temp.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    db_url = "sqlite:///" + db_path
    
    abc.new(db_url, {'data': x_obs_data})
    
    # 6. Run
    history = abc.run(max_nr_populations=max_populations)
    
    # 7. Extract Samples
    # Get the last population
    df, w = history.get_distribution(m=0, t=history.max_t)
    
    # df has columns theta0, theta1...
    # We need to convert to (n_samples, d) array in correct order
    samples = np.zeros((len(df), task.d))
    for i in range(task.d):
        samples[:, i] = df[f"theta{i}"].values
        
    # Resample to unweighted if needed? 
    # pyabc SMC particles have weights (usually uniform in final generation if using acceptance threshold, but SMC uses importance weights)
    # Let's resample to get equal-weighted samples for plotting consistency
    print("Resampling W2-ABC posterior...")
    indices = np.random.choice(len(df), size=n_samples, p=w/w.sum(), replace=True)
    final_samples = samples[indices]
    
    print(f"W2-ABC Completed. Max t={history.max_t}")
    
    return final_samples
