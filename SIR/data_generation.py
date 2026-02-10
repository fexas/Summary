import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ============================================================================
# SIR Task Configuration
# ============================================================================

# Population size
N_POP = 250
# Initial conditions
S0 = 249
I0 = 1
R0 = 0

# Observation Grid
T_MAX = 20.0
DT = 0.1
OBS_TIMES = np.arange(0, T_MAX + 1e-5, DT)
N_OBS = len(OBS_TIMES)

class SIRTask:
    def __init__(self, N_pop=N_POP, t_max=T_MAX, dt=DT):
        """
        SIR Task with Gillespie Simulator.
        State: (S, I, R)
        Parameters: theta = (beta, gamma)
        """
        self.N_pop = N_pop
        self.t_max = t_max
        self.dt = dt
        self.obs_times = np.arange(0, t_max + 1e-5, dt)
        self.n_obs = len(self.obs_times)
        
        # Parameter bounds for Prior
        # beta in [0, 3], gamma in [0, 1]
        self.lower = np.array([0.0, 0.0])
        self.upper = np.array([3.0, 1.0])
        self.d = 2 # beta, gamma
        self.d_x = 3 # S, I, R

    def sample_prior(self, batch_size):
        """
        Sample parameters from Uniform prior.
        """
        u = np.random.rand(batch_size, self.d)
        theta = (self.upper - self.lower) * u + self.lower
        return theta.astype(np.float32)

    def log_prior(self, theta):
        """
        Log prior probability (Uniform).
        """
        theta = np.asarray(theta)
        is_batch = theta.ndim > 1
        if not is_batch:
            theta = theta[np.newaxis, :]
            
        # Check bounds
        out_of_bounds = np.any((theta < self.lower) | (theta > self.upper), axis=1)
        log_probs = np.zeros(theta.shape[0], dtype=np.float32)
        log_probs[out_of_bounds] = -np.inf
        
        # Uniform density constant could be added, but usually 0 is fine for valid range
        # vol = (3-0)*(1-0) = 3. log_pdf = -log(3). 
        # But usually we just care about support.
        
        if not is_batch:
            return log_probs[0]
        return log_probs

    def simulator(self, theta, n_samples=None):
        """
        Vectorized Gillespie Algorithm for SIR model.
        
        Args:
            theta: (batch_size, 2) array of [beta, gamma]
            n_samples: ignored (compatibility)
            
        Returns:
            observations: (batch_size, n_obs, 3)
        """
        batch_size = theta.shape[0]
        beta = theta[:, 0]
        gamma = theta[:, 1]
        
        # Initial State
        S = np.full(batch_size, S0, dtype=np.int32)
        I = np.full(batch_size, I0, dtype=np.int32)
        R = np.full(batch_size, R0, dtype=np.int32)
        
        # Current time
        t = np.zeros(batch_size, dtype=np.float32)
        
        # Output container
        observations = np.zeros((batch_size, self.n_obs, 3), dtype=np.int32)
        
        # Initialize t=0 observation
        observations[:, 0, 0] = S
        observations[:, 0, 1] = I
        observations[:, 0, 2] = R
        
        # Index of next observation to fill
        next_obs_idx = np.ones(batch_size, dtype=np.int32)
        
        # Active simulations mask (those that haven't reached t_max)
        # Actually we check next_obs_idx < n_obs
        active = np.ones(batch_size, dtype=bool)
        
        # Loop until all simulations have recorded all observations
        while np.any(active):
            # Working with active indices only to save computation? 
            # Or masking? Masking is usually faster in numpy than fancy indexing for everything if batch is large.
            # But here states update sparsely. Let's use masking for correctness.
            
            # Compute rates
            # r_inf = beta * S * I / N
            # r_rec = gamma * I
            
            rate_inf = beta * S * I / self.N_pop
            rate_rec = gamma * I
            rate_total = rate_inf + rate_rec
            
            # Handle terminated epidemics (I=0 -> rate_total=0)
            # If rate_total is 0, nothing happens anymore.
            # We can effectively jump to t_max.
            is_static = (rate_total < 1e-9)
            
            # Set dummy rate for static to avoid div/0, will handle dt later
            rate_total_safe = rate_total.copy()
            rate_total_safe[is_static] = 1.0 
            
            # Sample time step
            dt = np.random.exponential(1.0 / rate_total_safe)
            
            # For static trajectories, time jumps to infinity (or past t_max)
            dt[is_static] = 1e9
            
            # Proposed next time
            t_next = t + dt
            
            # Check for observation crossings
            # We need to fill observations[b, k] if t[b] < obs_times[k] <= t_next[b]
            # We do this for all active batches.
            
            # Mask of batches that need to record an observation
            # We check if the NEXT observation time is passed
            # obs_times[next_obs_idx] <= t_next
            
            # Be careful with indices out of bounds
            valid_idx_mask = next_obs_idx < self.n_obs
            # If not valid idx, we shouldn't check time, but they shouldn't be active anyway.
            
            # Identify who crossed the immediate next boundary
            # We use a loop to handle multiple boundary crossings in one large step
            while True:
                # Check which active batches have passed their next observation point
                # Only check where next_obs_idx is valid
                mask_valid_obs = next_obs_idx < self.n_obs
                can_check = active & mask_valid_obs
                
                if not np.any(can_check):
                    break
                    
                # Times of the next target observation for each batch
                # Use clipped indices to avoid out of bounds, result masked by can_check
                safe_indices = np.minimum(next_obs_idx, self.n_obs - 1)
                target_times = self.obs_times[safe_indices]
                
                # Who crossed?
                crossed = can_check & (target_times <= t_next)
                
                if not np.any(crossed):
                    break
                    
                # Record state for crossed
                # The state remains constant in [t, t_next), so at target_time it is (S, I, R)
                observations[crossed, next_obs_idx[crossed], 0] = S[crossed]
                observations[crossed, next_obs_idx[crossed], 1] = I[crossed]
                observations[crossed, next_obs_idx[crossed], 2] = R[crossed]
                
                # Advance index
                next_obs_idx[crossed] += 1
            
            # Update states for those who had an event (and weren't static)
            # Determine event type
            # prob_inf = rate_inf / rate_total
            prob_inf = rate_inf / rate_total_safe
            u = np.random.rand(batch_size)
            is_infection = u < prob_inf
            
            # Only update real events (not static ones)
            # And only if t_next <= t_max (though we handle obs crossing above, 
            # we should technically stop simulation if t exceeds t_max? 
            # The user wants traces up to t_max.
            # If t_next > t_max, the event happens AFTER the window. 
            # The state doesn't change within the window.
            # But we already filled observations up to t_max in the inner loop if t_next went past it.
            
            event_happens = (~is_static) & (t_next <= self.t_max + 1.0) # Buffer
            
            # Update S, I, R
            # Infection: S-1, I+1
            # Recovery: I-1, R+1
            
            inf_mask = event_happens & is_infection
            rec_mask = event_happens & (~is_infection)
            
            S[inf_mask] -= 1
            I[inf_mask] += 1
            
            I[rec_mask] -= 1
            R[rec_mask] += 1
            
            # Update time
            t[~is_static] = t_next[~is_static]
            # For static, we can just set t to t_max + epsilon to finish?
            # Or just rely on next_obs_idx
            t[is_static] = self.t_max + 1.0
            
            # Update active status
            active = next_obs_idx < self.n_obs
            
        return observations


# ============================================================================
# Testing and Plotting
# ============================================================================

def plot_trajectories(obs, theta, filename="sir_trajectories.png"):
    """
    Plot S, I, R trajectories.
    obs: (batch, n_obs, 3)
    theta: (batch, 2)
    """
    # Plot first few
    n_plot = min(5, obs.shape[0])
    times = OBS_TIMES
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(n_plot):
        beta, gamma = theta[i]
        label = f"$\\beta$={beta:.2f}, $\\gamma$={gamma:.2f}"
        
        axes[0].plot(times, obs[i, :, 0], label=label, alpha=0.7)
        axes[1].plot(times, obs[i, :, 1], label=label, alpha=0.7)
        axes[2].plot(times, obs[i, :, 2], label=label, alpha=0.7)
        
    axes[0].set_title("Susceptible (S)")
    axes[1].set_title("Infected (I)")
    axes[2].set_title("Recovered (R)")
    
    for ax in axes:
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')
        
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved trajectories to {filename}")
    plt.close()

def plot_outbreak_distribution(obs, filename="sir_outbreak_size.png"):
    """
    Plot distribution of total outbreak size (final R).
    """
    # Final R is at the last time step
    final_R = obs[:, -1, 2]
    
    plt.figure(figsize=(8, 5))
    plt.hist(final_R, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.title(f"Distribution of Outbreak Sizes (N={len(final_R)})")
    plt.xlabel("Total Recovered (Final Size)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    print(f"Saved distribution to {filename}")
    plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Test Setup
    task = SIRTask()
    
    # 1. Reproduce Website Single/Few Run
    # Website: beta=1, gamma=0.5
    print("Running fixed parameter simulation (beta=1, gamma=0.5)...")
    theta_fixed = np.array([[1.0, 0.5]] * 10) # 10 runs
    start_time = time.time()
    obs_fixed = task.simulator(theta_fixed)
    print(f"Simulation took {time.time() - start_time:.4f}s")
    
    plot_trajectories(obs_fixed, theta_fixed, os.path.join(results_dir, "sir_fixed_traj.png"))
    
    # 2. Random Prior Sampling
    print("Running random prior simulation (N=1000)...")
    theta_prior = task.sample_prior(1000)
    start_time = time.time()
    obs_prior = task.simulator(theta_prior)
    print(f"Simulation (N=1000) took {time.time() - start_time:.4f}s")
    
    plot_trajectories(obs_prior, theta_prior, os.path.join(results_dir, "sir_prior_traj.png"))
    plot_outbreak_distribution(obs_prior, os.path.join(results_dir, "sir_outbreak_dist.png"))
    
    print("Done.")
