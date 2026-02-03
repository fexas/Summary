"""
ABC-SMC inference for g-and-k distribution using Learned Summary Statistics (SMMD).
Comparies SMMD-based Generative Network posterior with ABC-SMC posterior.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyabc
import tempfile
from pyabc.sampler import SingleCoreSampler

# Import from local modules
try:
    from data_generation import (
        simulator, 
        prior_generator,
        PRIOR_CONFIGS, 
        TRUE_PARAMS,
        N, n, d, d_x
    )
    from smmd_torch import (
        SMMD_Model, 
        train_smmd_torch, 
        DEVICE
    )
except ImportError:
    from G_and_K.data_generation import (
        simulator, 
        prior_generator,
        PRIOR_CONFIGS, 
        TRUE_PARAMS,
        N, n, d, d_x
    )
    from G_and_K.smmd_torch import (
        SMMD_Model, 
        train_smmd_torch, 
        DEVICE
    )

# Ensure result directory exists
RESULT_DIR = "G_and_K/smmd_pyabc_result"
os.makedirs(RESULT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(RESULT_DIR, "smmd_model.pth")

# ============================================================================
# 1. Train or Load SMMD Model
# ============================================================================

def get_trained_model():
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        print(f"Loading SMMD model from {MODEL_PATH}...")
        model = SMMD_Model().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    
    print("Training SMMD model from scratch...")
    
    # Generate Training Data (Reference Table)
    print(f"Generating {N} training samples...")
    theta_train = []
    x_train = []
    
    # Use 'weak_informative' prior for training
    prior_name = 'weak_informative'
    
    # Batch generation for speed
    batch_size = 1000
    num_batches = N // batch_size
    
    for _ in range(num_batches):
        # Generate batch of parameters
        thetas = []
        for _ in range(batch_size):
            thetas.append(prior_generator(prior_name))
        thetas = np.array(thetas) # (batch, d)
        
        # Simulate
        xs = simulator(thetas, n_samples=n) # (batch, n, 1)
        
        theta_train.append(thetas)
        x_train.append(xs)
        
    theta_train = np.concatenate(theta_train, axis=0)
    x_train = np.concatenate(x_train, axis=0)
    
    # Train
    model = train_smmd_torch(theta_train, x_train, prior_name, RESULT_DIR)
    
    # Save
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model

# ============================================================================
# 2. ABC-SMC Setup
# ============================================================================

# Global model reference for pickling (pyabc requirement for multiprocessing sometimes)
# However, we will pass the model to the sumstat class.
trained_model = None

class LearnedSumstat(pyabc.Sumstat):
    """
    Summary statistic using the trained SMMD Summary Network (T).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        
    def __call__(self, data: dict) -> np.ndarray:
        # data['x'] is expected to be (n_samples, d_x) or (1, n_samples, d_x)
        # The simulator returns dictionary {"x": x}
        x = data["x"]
        
        # Ensure x is (1, n, d_x) for the network
        if x.ndim == 2:
            x = x[np.newaxis, ...]
            
        x_tensor = torch.from_numpy(x).float().to(DEVICE)
        
        with torch.no_grad():
            # T returns (1, p)
            stats = self.model.T(x_tensor)
            
        return stats.cpu().numpy().flatten()

def model_wrapper(p):
    """
    pyABC model wrapper.
    p: dictionary of parameters
    """
    # Convert dict to array [A, B, g, k]
    theta = np.array([p['A'], p['B'], p['g'], p['k']])
    
    # Simulator expects (batch, d), so (1, d)
    # Returns (1, n, 1)
    x = simulator(theta, n_samples=n)
    
    # Return as (n, 1) for consistency with observation
    return {"x": x[0]}

def run_analysis():
    global trained_model
    trained_model = get_trained_model()
    
    # 1. Generate Observation (Ground Truth)
    print("Generating observation from TRUE_PARAMS...")
    x_obs_raw = simulator(TRUE_PARAMS, n_samples=n) # (1, n, 1)
    x_obs_dict = {"x": x_obs_raw[0]}
    
    # 2. Define Prior (pyabc.Distribution)
    # Using 'weak_informative' bounds
    config = PRIOR_CONFIGS['weak_informative']
    prior = pyabc.Distribution(
        A=pyabc.RV("uniform", config['A'][0], config['A'][1] - config['A'][0]),
        B=pyabc.RV("uniform", config['B'][0], config['B'][1] - config['B'][0]),
        g=pyabc.RV("uniform", config['g'][0], config['g'][1] - config['g'][0]),
        k=pyabc.RV("uniform", config['k'][0], config['k'][1] - config['k'][0])
    )
    
    # 3. Define Distance
    # Euclidean distance on learned summary statistics
    # This is equivalent to PNormDistance(p=2) on the output of LearnedSumstat
    distance = pyabc.PNormDistance(p=2, sumstat=LearnedSumstat(trained_model))
    
    # 4. Run ABC-SMC (SMMD Summary Statistics)
    print("Starting ABC-SMC (SMMD Summary Statistics)...")
    abc = pyabc.ABCSMC(
        model_wrapper, 
        prior, 
        distance, 
        population_size=500,
        sampler=SingleCoreSampler() # Use SingleCore to avoid pickling issues with PyTorch model
    )
    
    # Create temp db
    db_path = os.path.join(RESULT_DIR, "abc_smc.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        
    history = abc.new(
        db="sqlite:///" + db_path, 
        observed_sum_stat=x_obs_dict
    )
    
    # Run
    history = abc.run(max_nr_populations=10, minimum_epsilon=0.05)
    
    # 5. Get ABC Posterior Samples
    print("Collecting ABC posterior samples...")
    # Get the last population
    df, w = history.get_distribution(m=0, t=history.max_t)
    abc_samples = df.to_numpy() # Order: A, B, g, k (alphabetical usually?)
    
    # ============================================================================
    # 5.5 ABC-SMC with Wasserstein Distance (W2)
    # ============================================================================
    
    class IdSumstat(pyabc.Sumstat):
        """Identity summary statistic for Wasserstein distance."""
        def __call__(self, data: dict) -> np.ndarray:
            # data['x'] is (n, 1) or (1, n, 1)
            # Wasserstein expects (n_samples, dim)
            x = data["x"]
            if x.ndim == 3:
                x = x[0] # (n, 1)
            return x

    print("Starting ABC-SMC (Wasserstein Distance)...")
    
    # Use 2-Wasserstein distance
    # For 1D data, this uses optimal transport (pot) or sorting.
    distance_w2 = pyabc.distance.WassersteinDistance(p=2, sumstat=IdSumstat())
    
    abc_w2 = pyabc.ABCSMC(
        model_wrapper, 
        prior, 
        distance_w2, 
        population_size=500,
        sampler=SingleCoreSampler()
    )
    
    db_path_w2 = os.path.join(RESULT_DIR, "abc_smc_w2.db")
    if os.path.exists(db_path_w2):
        os.remove(db_path_w2)
        
    history_w2 = abc_w2.new(
        db="sqlite:///" + db_path_w2, 
        observed_sum_stat=x_obs_dict
    )
    
    # Run W2 ABC
    # Wasserstein distances can be large, so we let epsilon decrease adaptively
    history_w2 = abc_w2.run(max_nr_populations=10)
    
    # Get W2 Posterior Samples
    print("Collecting W2-ABC posterior samples...")
    df_w2, w_w2 = history_w2.get_distribution(m=0, t=history_w2.max_t)
    w2_samples = df_w2.to_numpy()

    # ============================================================================
    
    # 6. Get SMMD Generative Posterior Samples
    print("Sampling from SMMD Generative Network...")
    x_obs_tensor = torch.from_numpy(x_obs_raw).float().to(DEVICE)
    n_post = len(abc_samples)
    
    trained_model.eval()
    with torch.no_grad():
        stats = trained_model.T(x_obs_tensor) # (1, p)
        z = torch.randn(1, n_post, d, device=DEVICE)
        smmd_samples = trained_model.G(z, stats) # (1, n_post, d)
        smmd_samples = smmd_samples.squeeze(0).cpu().numpy()
        
    # 7. Plot Comparison
    print("Plotting comparison...")
    param_names = ['A', 'B', 'g', 'k']
    
    # Prepare DataFrame for Seaborn
    data_list = []
    
    for i in range(n_post):
        # ABC (SMMD-Sumstat)
        row_abc = {'Method': 'ABC-SMC (SMMD-Sumstat)'}
        for j, p_name in enumerate(param_names):
            row_abc[p_name] = abc_samples[i, j]
        data_list.append(row_abc)
        
        # SMMD
        row_smmd = {'Method': 'SMMD-Gen'}
        for j, p_name in enumerate(param_names):
            row_smmd[p_name] = smmd_samples[i, j]
        data_list.append(row_smmd)

    # Add W2 Samples (might have different count, so iterate separately)
    n_post_w2 = len(w2_samples)
    for i in range(n_post_w2):
        row_w2 = {'Method': 'ABC-SMC (W2)'}
        for j, p_name in enumerate(param_names):
            row_w2[p_name] = w2_samples[i, j]
        data_list.append(row_w2)
        
    df_plot = pd.DataFrame(data_list)
    
    # PairPlot
    g = sns.pairplot(
        df_plot, 
        hue='Method', 
        kind='kde', 
        diag_kind='kde',
        plot_kws={'alpha': 0.5},
        corner=True
    )
    
    # Add True Params
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                # Diagonal
                g.diag_axes[i].axvline(TRUE_PARAMS[i], color='k', linestyle='--', label='True')
            else:
                # Off-diagonal
                g.axes[i, j].scatter(TRUE_PARAMS[j], TRUE_PARAMS[i], color='k', marker='*', s=100, label='True')

    plt.suptitle("Posterior Comparison: ABC-SMC(SMMD) vs SMMD-Gen vs ABC-SMC(W2)", y=1.02)
    plt.savefig(os.path.join(RESULT_DIR, "posterior_comparison_w2.png"))
    print(f"Comparison plot saved to {os.path.join(RESULT_DIR, 'posterior_comparison_w2.png')}")

if __name__ == "__main__":
    run_analysis()
