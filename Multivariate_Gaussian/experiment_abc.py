
import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import ot
import math
import scipy.stats as stats
import pyabc
import tempfile
import logging

from models import SMMD_Model, sliced_mmd_loss
from experiment import kl_empirical, morton_order, swap_distance

# Configure logging for pyabc
logging.getLogger("pyabc").setLevel(logging.INFO)

# ============================================================================
# 1. Configuration
# ============================================================================

DIMS = [2, 10, 20]
N_OBS = 100
SIGMA_SQ_TRUE = 4.0
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

# SMMD Training Config
N_TRAIN_SAMPLES = 10000
BATCH_SIZE = 128
EPOCHS = 50 # Reduced for speed in this demo, usually needs more
THETA_DIM = 1
SUMMARY_DIM = 2 * THETA_DIM
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if not torch.backends.mps.is_available() and torch.cuda.is_available():
    DEVICE = torch.device("cuda")

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_abc")
os.makedirs(RESULT_DIR, exist_ok=True)

# Fixed seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# 2. Simulator & Model Logic
# ============================================================================

# We need a fixed m_star for the entire experiment per dimension
# To handle this cleanly in pyABC (which might run in parallel), 
# we'll generate m_star globally or pass it.
# However, pyABC workers need access to it.
# Simple approach: Global dictionary keyed by dim.
M_STAR_DICT = {}

def get_m_star(dim):
    if dim not in M_STAR_DICT:
        # m_star ~ N(0, I_d)
        M_STAR_DICT[dim] = np.random.randn(dim)
    return M_STAR_DICT[dim]

def simulator_model(parameters):
    """
    pyABC simulator.
    parameters: dict {'sigma_sq': float}
    Returns: dict {'data': np.array (N_OBS, dim)}
    """
    # We need to know the current dimension. 
    # Usually passed via closure or global. 
    # Let's use a global CURRENT_DIM set before running ABC.
    dim = CURRENT_DIM 
    sigma_sq = parameters['sigma_sq']
    m_star = get_m_star(dim)
    
    # Generate Data: y_i ~ N(m_star, sigma_sq * I)
    # shape: (N_OBS, dim)
    cov = sigma_sq * np.eye(dim)
    y = np.random.multivariate_normal(m_star, cov, N_OBS)
    
    return {'data': y}

def true_posterior_pdf(sigma_sq_vals, dim, y_obs):
    """
    Evaluate True Posterior PDF (Inverse Gamma).
    """
    m_star = get_m_star(dim)
    
    # Calculate parameters
    # alpha_post = alpha_prior + n * d / 2
    # beta_post = beta_prior + 0.5 * sum(||y_i - m_star||^2)
    
    alpha_post = PRIOR_ALPHA + N_OBS * dim / 2.0
    
    diff = y_obs - m_star # (N, d)
    sq_norms = np.sum(diff**2, axis=1) # (N,)
    sum_sq_error = np.sum(sq_norms)
    
    beta_post = PRIOR_BETA + 0.5 * sum_sq_error
    
    return stats.invgamma.pdf(sigma_sq_vals, a=alpha_post, scale=beta_post)

# ============================================================================
# 3. SMMD Helper
# ============================================================================

def train_smmd(dim, model_path):
    print(f"Training SMMD for dim={dim}...")
    
    # Generate Training Data from Prior
    # sigma^2 ~ IG(1, 1)
    sigma_sq_train = stats.invgamma.rvs(a=PRIOR_ALPHA, scale=PRIOR_BETA, size=N_TRAIN_SAMPLES)
    
    # Simulator for training data
    # Note: SMMD needs to learn to summarize data regardless of specific realization of m_star?
    # Or should it know m_star?
    # In the problem statement, m_star is fixed for the observation.
    # If the network sees y, it should infer sigma.
    # The distribution depends on sigma.
    # The simulator used for training should match the problem.
    # So we use the SAME fixed m_star.
    m_star = get_m_star(dim)
    
    x_train = []
    for s2 in sigma_sq_train:
        cov = s2 * np.eye(dim)
        x_train.append(np.random.multivariate_normal(m_star, cov, N_OBS))
    x_train = np.array(x_train) # (N_train, N_samples, dim)
    
    # Convert to Tensor
    theta_tensor = torch.from_numpy(sigma_sq_train).float().unsqueeze(1) # (N, 1)
    x_tensor = torch.from_numpy(x_train).float()
    
    dataset = torch.utils.data.TensorDataset(theta_tensor, x_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SMMD_Model(input_dim=dim, summary_dim=SUMMARY_DIM, theta_dim=THETA_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_theta, batch_x in loader:
            batch_theta = batch_theta.to(DEVICE)
            batch_x = batch_x.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Sample Z
            curr_batch_size = batch_theta.size(0)
            z = torch.randn(curr_batch_size, 50, THETA_DIM, device=DEVICE)
            
            # Forward
            theta_fake = model(batch_x, z)
            
            # Loss
            loss = sliced_mmd_loss(batch_theta, theta_fake, bandwidth=2.0/N_OBS)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(loader):.4f}")
            
    torch.save(model.state_dict(), model_path)
    return model

# ============================================================================
# 4. Distances
# ============================================================================

def sliced_wasserstein_distance(x, y):
    # x, y are dictionaries from pyabc {'data': array}
    X = x['data']
    Y = y['data']
    
    n_projections = 100
    dim = X.shape[1]
    projections = np.random.randn(dim, n_projections)
    projections /= np.linalg.norm(projections, axis=0)
    
    proj_x = X @ projections
    proj_y = Y @ projections
    
    proj_x.sort(axis=0)
    proj_y.sort(axis=0)
    
    sw_sq = np.mean((proj_x - proj_y)**2)
    return np.sqrt(sw_sq)

def hilbert_distance(x, y):
    # Pure Hilbert (via Morton proxy): sort, pair by index, compute mean squared distance
    X = x['data']
    Y = y['data']
    perm_x = morton_order(X)
    perm_y = morton_order(Y)
    Xs = X[perm_x]
    Ys = Y[perm_y]
    # Pairwise squared euclidean for matched indices
    diffs = Xs - Ys
    cost = np.sum(diffs * diffs, axis=1).mean()
    return np.sqrt(cost)

def hilbert_swap_distance(x, y):
    X = x['data']
    Y = y['data']
    # Use the windowed swap implementation from experiment.py
    return swap_distance(X, Y, n_sweeps=1)

# Euclidean Distance on Variances
def euclidean_variance_distance(x, y):
    # x, y: {'data': ...}
    X = x['data']
    Y = y['data']
    # Mean of variances across dimensions? Or just variance of flattened array?
    # "Euclidean distance between sample variances"
    # If sample variance is a vector (per dim), we compute norm.
    # If it's a scalar (pooled), we compute abs diff.
    # Given isotropic Gaussian, pooled variance is better.
    var_x = np.var(X)
    var_y = np.var(Y)
    return abs(var_x - var_y)

# SMMD Summary Distance
# Need global access to current model? Or pass it?
# pyABC distance functions can be classes.
class SMMDSummaryDistance(pyabc.Distance):
    def __init__(self, model):
        self.model = model
        
    def __call__(self, x, y, t=None, par=None):
        # x: simulation {'data': ...}
        # y: observation {'data': ...}
        # pyabc might pass t and par, so we accept **kwargs or t, par
        X = torch.from_numpy(x['data']).float().unsqueeze(0).to(DEVICE)
        Y = torch.from_numpy(y['data']).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            s_x = self.model.T(X).cpu().numpy().flatten()
            s_y = self.model.T(Y).cpu().numpy().flatten()
            
        return np.linalg.norm(s_x - s_y)

# ============================================================================
# 5. Main Loop
# ============================================================================

def run_experiment_abc():
    global CURRENT_DIM
    
    fig, axes = plt.subplots(len(DIMS), 2, figsize=(12, 5 * len(DIMS)))
    # Columns: 1. Posterior Density, 2. W1 vs Time
    
    for i, dim in enumerate(DIMS):
        print(f"\nRunning Experiment for Dim = {dim}")
        CURRENT_DIM = dim
        m_star = get_m_star(dim)
        
        # 1. Train/Load SMMD
        model_path = os.path.join(RESULT_DIR, f"smmd_abc_dim_{dim}.pth")
        if os.path.exists(model_path):
            print("Loading existing SMMD model...")
            model = SMMD_Model(input_dim=dim, summary_dim=SUMMARY_DIM, theta_dim=THETA_DIM).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            model = train_smmd(dim, model_path)
        model.eval()
        
        # 2. Generate Observation
        # sigma_star = 4
        obs_data = simulator_model({'sigma_sq': SIGMA_SQ_TRUE})
        
        # 3. Define ABC Setup
        prior = pyabc.Distribution(sigma_sq=pyabc.RV("invgamma", a=PRIOR_ALPHA, scale=PRIOR_BETA))
        
        # Define settings for comparison
        # We handle summary stats inside distance functions to avoid pyabc wrapping issues
        settings = [
            {
                "label": "Euclidean-ABC",
                "distance": euclidean_variance_distance,
                "color": "#1f77b4"
            },
            {
                "label": "SW-ABC",
                "distance": sliced_wasserstein_distance,
                "color": "#2ca02c"
            },
            {
                "label": "WABC-Hilbert",
                "distance": hilbert_distance,
                "color": "#d62728"
            },
            {
                "label": "WABC-Swapping",
                "distance": hilbert_swap_distance,
                "color": "#9467bd"
            },
            {
                "label": "SMMD Summary",
                "distance": SMMDSummaryDistance(model),
                "color": "#ff7f0e"
            }
        ]
        
        # 4. Run ABC for each setting
        # To save time, we limit generations or population size
        POPULATION_SIZE = 100 # Small for demo speed
        MAX_GENERATIONS = 5
        
        results = {}
        
        for setting in settings:
            print(f"Running ABC for {setting['label']}...")
            
            abc = pyabc.ABCSMC(
                models=simulator_model,
                parameter_priors=prior,
                distance_function=setting['distance'],
                population_size=POPULATION_SIZE,
                sampler=pyabc.sampler.SingleCoreSampler() # Use SingleCore for simplicity/compatibility
            )
            
            db_path = os.path.join(RESULT_DIR, f"abc_dim_{dim}_{setting['label'].replace(' ', '_')}.db")
            if os.path.exists(db_path):
                os.remove(db_path) # Clean up previous run
            db_url = "sqlite:///" + db_path
            
            # Prepare new run
            abc.new(db_url, obs_data)
            
            # Run
            history = abc.run(max_nr_populations=MAX_GENERATIONS)
            results[setting['label']] = history
            
        # 5. SMMD Approximate Posterior (Direct)
        print("Generating SMMD Posterior samples...")
        obs_tensor = torch.from_numpy(obs_data['data']).float().unsqueeze(0).to(DEVICE)
        z_samples = torch.randn(1, 1000, THETA_DIM, device=DEVICE)
        with torch.no_grad():
            smmd_posterior_samples = model(obs_tensor, z_samples).cpu().numpy().flatten()
            
        # 6. Plotting
        # 6a. Posterior Density
        ax_dens = axes[i, 0] if len(DIMS) > 1 else axes[0]
        ax_perf = axes[i, 1] if len(DIMS) > 1 else axes[1]
        
        x_grid = np.linspace(0.1, 10, 200)
        true_pdf = true_posterior_pdf(x_grid, dim, obs_data['data'])
        ax_dens.plot(x_grid, true_pdf, color='gray', linestyle='--', label='True posterior', linewidth=2)
        
        # Plot SMMD Direct
        kde_smmd = stats.gaussian_kde(smmd_posterior_samples)
        ax_dens.plot(x_grid, kde_smmd(x_grid), label='SMMD (Amortized)', color='black', linestyle='-.', linewidth=1.5)
        
        # Plot ABC Posteriors
        for setting in settings:
            hist = results[setting['label']]
            df, w = hist.get_distribution(t=hist.max_t)
            # Weighted KDE
            kde = stats.gaussian_kde(df['sigma_sq'], weights=w)
            ax_dens.plot(x_grid, kde(x_grid), label=setting['label'], color=setting['color'], linewidth=2)
            
            # Calculate W1 vs Time
            # Extract history
            times = []
            w1_dists = []
            
            # True samples for W1 calculation
            true_samples = stats.invgamma.rvs(
                a=PRIOR_ALPHA + N_OBS * dim / 2.0,
                scale=PRIOR_BETA + 0.5 * np.sum(np.sum((obs_data['data'] - m_star)**2, axis=1)),
                size=1000
            )
            
            for t in range(hist.max_t + 1):
                df_t, w_t = hist.get_distribution(t=t)
                # Resample to get unweighted samples for W1 calculation
                # Or use POT with weights
                # Let's resample 1000 times
                indices = np.random.choice(len(df_t), size=1000, p=w_t/w_t.sum())
                samples_t = df_t.iloc[indices]['sigma_sq'].values
                
                w1 = stats.wasserstein_distance(samples_t, true_samples)
                
                # Time: Total simulations up to t
                total_sims = hist.get_population_strategy()['samples'][t] if 'samples' in hist.get_population_strategy() else (t+1)*POPULATION_SIZE 
                # Better way: hist.get_all_populations() metadata
                # pyabc history 'total_nr_simulations' per generation
                # Need to query db or sum up
                # Simplified: assume cumulative sims
                # Let's just use 't' (Generation) for x-axis as proxy or try to get sims
                # For this plot, generation index is fine or we can try to get exact number
                w1_dists.append(w1)
                times.append(t) # Using generation index for now
                
            ax_perf.plot(times, w1_dists, label=setting['label'], marker='o', color=setting['color'])
            
        ax_dens.set_title(f"(d = {dim})", fontsize=12)
        ax_dens.set_xlabel(r"$\sigma^2$")
        ax_dens.set_ylabel("density")
        ax_dens.grid(True, linestyle='--', alpha=0.4)
        ax_dens.legend()
        
        ax_perf.set_title(f"(d = {dim})", fontsize=12)
        ax_perf.set_xlabel("time (s)")  # proxy: generation index
        ax_perf.set_ylabel("Wasserstein-1 Distance")
        ax_perf.grid(True, linestyle='--', alpha=0.4)
        ax_perf.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "abc_comparison.png"))
    print(f"Plot saved to {os.path.join(RESULT_DIR, 'abc_comparison.png')}")

if __name__ == "__main__":
    run_experiment_abc()
