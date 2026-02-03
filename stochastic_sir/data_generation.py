import numpy as np
import torch
import scipy.stats as stats

# ============================================================================
# 1. Configuration & Constants
# ============================================================================

# Population size
N = 1000.0

# Time settings
T_MAX = 50.0 # Standard observation window
NUM_OBS = 50 # Number of observation points (e.g., every 1 unit time)
OBS_TIMES = np.linspace(0, T_MAX, NUM_OBS)

# Parameter dimensions
d = 2  # (beta, gamma)
d_x = 2 # (I(t), t) or just I(t)? Let's use (I(t)/N, t/T_max) normalized

# Prior bounds (Uniform)
# Literature often uses Beta ~ U(0, 1) or higher, Gamma ~ U(0, 1)
# R0 = Beta/Gamma. Epidemics usually have R0 > 1.
# We'll use a range that allows for both epidemic and die-out.
PRIOR_MIN = np.array([0.01, 0.01])
PRIOR_MAX = np.array([1.0, 1.0])

TRUE_PARAMS = np.array([0.4, 0.1]) # Beta=0.4, Gamma=0.1 -> R0=4

# ============================================================================
# 2. Simulator (Gillespie Algorithm)
# ============================================================================

def gillespie_sir(theta, N=N, t_max=T_MAX):
    """
    Simulates one SIR trajectory using Gillespie algorithm.
    theta: [beta, gamma]
    Returns: (times, S, I, R) raw trajectory
    """
    beta, gamma = theta
    
    # Initial state
    t = 0.0
    S = N - 1
    I = 1
    R = 0
    
    times = [0.0]
    S_list = [S]
    I_list = [I]
    R_list = [R]
    
    while t < t_max and I > 0:
        rate_inf = beta * S * I / N
        rate_rec = gamma * I
        total_rate = rate_inf + rate_rec
        
        if total_rate == 0:
            break
            
        # Time step
        dt = np.random.exponential(1.0 / total_rate)
        t += dt
        
        if t > t_max:
            break
            
        # Event selection
        if np.random.rand() < rate_inf / total_rate:
            S -= 1
            I += 1
        else:
            I -= 1
            R += 1
            
        times.append(t)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
        
    return np.array(times), np.array(S_list), np.array(I_list), np.array(R_list)

def interpolate_trajectory(times, values, query_times):
    """
    Interpolates values at query_times using zero-order hold (step function)
    since Gillespie is a jump process.
    """
    # np.interp is linear, we want step function (previous value)
    # searchsorted finds indices where elements should be inserted to maintain order
    indices = np.searchsorted(times, query_times, side='right') - 1
    indices = np.maximum(indices, 0) # Clamp to 0
    return values[indices]

def simulator(theta, n_points=NUM_OBS, normalize=True):
    """
    Simulates a batch of trajectories.
    theta: (batch_size, d) or (d,)
    Returns: (batch_size, n_points, d_x)
    d_x = 2: [I(t)/N, t/T_max]
    """
    theta = np.atleast_2d(theta)
    batch_size = theta.shape[0]
    
    output = np.zeros((batch_size, n_points, d_x))
    
    for i in range(batch_size):
        times, S, I, R = gillespie_sir(theta[i], N, T_MAX)
        
        # Interpolate I at OBS_TIMES
        I_interp = interpolate_trajectory(times, I, OBS_TIMES)
        
        if normalize:
            output[i, :, 0] = I_interp / N
            output[i, :, 1] = OBS_TIMES / T_MAX
        else:
            output[i, :, 0] = I_interp
            output[i, :, 1] = OBS_TIMES
            
    return output.astype(np.float32)

# ============================================================================
# 3. Prior
# ============================================================================

def sample_prior(batch_size):
    return np.random.uniform(low=PRIOR_MIN, high=PRIOR_MAX, size=(batch_size, d)).astype(np.float32)

# ============================================================================
# 4. Summary Statistics (Handcrafted for SNPE)
# ============================================================================

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
    
    # 4. Mean Infection Level (Area under curve proxy)
    mean_I = torch.mean(I_curve, dim=1)
    
    return torch.stack([max_I, time_of_max, final_I, mean_I], dim=1)
