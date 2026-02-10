import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ============================================================================
# Lotka-Volterra Task Configuration
# ============================================================================

# Initial populations (from ExperimentDescription.md)
# X = Predators, Y = Prey
X0 = 50
Y0 = 100

# Time settings
T_MAX = 30.0
DT = 0.2
OBS_TIMES = np.arange(0, T_MAX + 1e-5, DT)
N_OBS = len(OBS_TIMES)

class LVTask:
    def __init__(self, t_max=T_MAX, dt=DT):
        """
        Lotka-Volterra Task with Vectorized Gillespie Simulator.
        
        State: (X, Y) where X=Predators, Y=Prey
        Parameters: theta = (theta1, theta2, theta3, theta4) in LOG scale
        
        Reactions (rates):
        1. X -> X+1 (Predator Born):  exp(theta1) * X * Y
        2. X -> X-1 (Predator Die):   exp(theta2) * X
        3. Y -> Y+1 (Prey Born):      exp(theta3) * Y
        4. Y -> Y-1 (Prey Eaten):     exp(theta4) * X * Y
        """
        self.t_max = t_max
        self.dt = dt
        self.obs_times = np.arange(0, t_max + 1e-5, dt)
        self.n_obs = len(self.obs_times)
        self.d = 4
        self.d_x = 2 # X, Y
        
        # Prior bounds (Broad/Vague)
        self.lower = np.full(4, -5.0)
        self.upper = np.full(4, 2.0)
        
        # Ground Truth Parameters (from description)
        # log(0.01), log(0.5), log(1), log(0.01)
        self.theta_true = np.log([0.01, 0.5, 1.0, 0.01])

    def get_ground_truth(self):
        """
        Generate a single ground truth observation.
        """
        theta = self.theta_true
        # Expand for simulator
        theta_batch = theta[np.newaxis, :]
        obs_batch = self.simulator(theta_batch)
        return theta, obs_batch[0]

    def sample_prior(self, batch_size, prior_type="vague"):
        """
        Sample parameters from prior.
        
        Args:
            batch_size: Number of samples
            prior_type: "vague" (Broad) or "informative" (Oscillating)
        """
        if prior_type == "vague":
            # Uniform [-5, 2]
            u = np.random.rand(batch_size, self.d)
            theta = (self.upper - self.lower) * u + self.lower
            return theta.astype(np.float32)
            
        elif prior_type == "informative":
            # Truncated Normal centered at theta_true with std=0.5
            # Rejection sampling for truncation
            theta = np.zeros((batch_size, self.d), dtype=np.float32)
            accepted = np.zeros(batch_size, dtype=bool)
            
            while not np.all(accepted):
                # Generate remaining needed samples
                n_needed = np.sum(~accepted)
                
                # Normal(theta_true, 0.5^2)
                samples = np.random.normal(
                    loc=self.theta_true, 
                    scale=0.5, 
                    size=(n_needed, self.d)
                )
                
                # Check bounds
                in_bounds = np.all((samples >= self.lower) & (samples <= self.upper), axis=1)
                
                # Fill accepted
                indices = np.where(~accepted)[0]
                theta[indices[in_bounds]] = samples[in_bounds]
                accepted[indices[in_bounds]] = True
                
            return theta
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")

    def log_prior(self, theta, prior_type="vague"):
        """
        Log prior probability.
        """
        theta = np.asarray(theta)
        is_batch = theta.ndim > 1
        if not is_batch:
            theta = theta[np.newaxis, :]
            
        # 1. Check bounds (common to both)
        out_of_bounds = np.any((theta < self.lower) | (theta > self.upper), axis=1)
        log_probs = np.zeros(theta.shape[0], dtype=np.float32)
        log_probs[out_of_bounds] = -np.inf
        
        if prior_type == "informative":
            # Add Gaussian log prob for in-bounds samples
            # log N(theta | mu, sigma^2) = -0.5 * ((theta-mu)/sigma)^2 - const
            sigma = 0.5
            diff = theta - self.theta_true
            # Sum over dimensions
            log_gauss = -0.5 * np.sum((diff / sigma)**2, axis=1)
            # Add to valid samples
            log_probs[~out_of_bounds] += log_gauss[~out_of_bounds]
            # Note: Normalization constant ignored as is typical in MCMC ratios
            
        if not is_batch:
            return log_probs[0]
        return log_probs

    def simulator(self, theta, n_samples=None):
        """
        Vectorized Gillespie Algorithm for Lotka-Volterra model.
        
        Args:
            theta: (batch_size, 4) array of log-parameters
            
        Returns:
            observations: (batch_size, n_obs, 2)
        """
        batch_size = theta.shape[0]
        
        # Convert log-params to rates
        # theta = [log(c1), log(c2), log(c3), log(c4)]
        rates_params = np.exp(theta)
        c1 = rates_params[:, 0] # Predator Born (XY)
        c2 = rates_params[:, 1] # Predator Die (X)
        c3 = rates_params[:, 2] # Prey Born (Y)
        c4 = rates_params[:, 3] # Prey Die (XY)
        
        # Initial State
        X = np.full(batch_size, X0, dtype=np.int32)
        Y = np.full(batch_size, Y0, dtype=np.int32)
        
        # Current time
        t = np.zeros(batch_size, dtype=np.float32)
        
        # Output container
        observations = np.zeros((batch_size, self.n_obs, 2), dtype=np.int32)
        
        # Initialize t=0 observation
        observations[:, 0, 0] = X
        observations[:, 0, 1] = Y
        
        # Tracking observation indices
        next_obs_idx = np.ones(batch_size, dtype=np.int32)
        
        # Active simulations mask
        active = np.ones(batch_size, dtype=bool)
        
        # Max steps safety (optional, but good for avoiding infinite loops)
        max_steps = 100000
        steps = 0
        
        while np.any(active) and steps < max_steps:
            steps += 1
            
            # 1. Calculate Reaction Rates for active simulations
            # Need to handle potential overflows or zeros if populations explode
            # X, Y are (batch,)
            
            curr_X = X[active]
            curr_Y = Y[active]
            
            # Reactions:
            # 1. X -> X+1 (rate c1 * X * Y)
            # 2. X -> X-1 (rate c2 * X)
            # 3. Y -> Y+1 (rate c3 * Y)
            # 4. Y -> Y-1 (rate c4 * X * Y)
            
            r1 = c1[active] * curr_X * curr_Y
            r2 = c2[active] * curr_X
            r3 = c3[active] * curr_Y
            r4 = c4[active] * curr_X * curr_Y
            
            total_rate = r1 + r2 + r3 + r4
            
            # Handle extinction (total_rate = 0)
            # If total_rate is 0, no more events happen. 
            # We just fast forward to T_MAX.
            zero_rate_mask = (total_rate < 1e-10)
            
            if np.any(zero_rate_mask):
                # For zero rate trajectories, they stay constant forever
                # Just mark them as done for simulation logic, but we need to fill observations later
                # Actually, simpler to set total_rate to small epsilon to avoid div/0, 
                # or better: set dt to infinity (large value)
                total_rate[zero_rate_mask] = 1e-10 # avoid div/0
                # We will handle the "jump to end" logic by dt
            
            # 2. Sample Time Step (Exponential)
            # dt ~ Exp(total_rate)
            dt_step = -np.log(np.random.rand(np.sum(active))) / total_rate
            
            if np.any(zero_rate_mask):
                dt_step[zero_rate_mask] = 1e9 # Effectively infinite time
            
            # 3. Update Time
            t_next = t[active] + dt_step
            
            # 4. Check for Observations BEFORE updating state
            # Logic: The state (X, Y) is constant during [t, t+dt).
            # So for any obs_time in [t, t+dt), the observation is (X, Y).
            
            # We iterate to catch multiple observations in one step (if dt is large)
            # Vectorized "while" for observations is tricky. 
            # We can use a loop here because number of observations per step is small (usually 0 or 1)
            
            # Get indices of currently active simulations in the full batch
            active_indices = np.where(active)[0]
            
            # Check which active sims crossed an observation time
            # We need to loop because one large step might skip multiple observation points
            still_recording = np.ones(len(active_indices), dtype=bool)
            
            while np.any(still_recording):
                # Check current next_obs_time for each active sim
                # Map back to full batch indices
                current_indices = active_indices[still_recording]
                
                # Safety check for finished observations
                finished_obs = next_obs_idx[current_indices] >= self.n_obs
                if np.any(finished_obs):
                    still_recording[np.where(still_recording)[0][finished_obs]] = False
                    current_indices = active_indices[still_recording]
                    if len(current_indices) == 0:
                        break
                
                next_times = self.obs_times[next_obs_idx[current_indices]]
                crossed = t_next[still_recording] > next_times
                
                if not np.any(crossed):
                    break
                    
                # For crossed trajectories, record the state (Pre-Jump State)
                crossed_indices = current_indices[crossed]
                # Indices within the 'still_recording' subset
                subset_crossed_mask = crossed
                
                # Record
                observations[crossed_indices, next_obs_idx[crossed_indices], 0] = X[crossed_indices]
                observations[crossed_indices, next_obs_idx[crossed_indices], 1] = Y[crossed_indices]
                
                # Advance index
                next_obs_idx[crossed_indices] += 1
                
                # Update still_recording: those who didn't cross are done for this step
                # But those who crossed might cross another one (if dt is huge), so we loop again
                # Ideally, we only loop for those who crossed.
                # Optimization: just loop the 'crossed' ones? 
                # For simplicity, we just loop 'still_recording' but condition check handles it.
                # If a sim didn't cross, 'crossed' is False, so we don't increment.
                # But we need to remove them from 'still_recording' to avoid infinite loop if they never cross again in this step.
                
                # Actually, logic:
                # If t_next > next_time, we record and increment.
                # If t_next <= next_time, we are done recording for this step.
                
                still_recording[np.where(still_recording)[0][~crossed]] = False
            
            # 5. Determine Reaction Type
            # Probabilities: r1/R, r2/R, r3/R, r4/R
            # We use uniform sampling to decide
            
            u = np.random.rand(np.sum(active)) * total_rate
            
            # Cumulative sums
            cum_r1 = r1
            cum_r2 = r1 + r2
            cum_r3 = r1 + r2 + r3
            
            # Reaction masks
            reaction_1 = (u < cum_r1)
            reaction_2 = (u >= cum_r1) & (u < cum_r2)
            reaction_3 = (u >= cum_r2) & (u < cum_r3)
            reaction_4 = (u >= cum_r3) # & (u < total)
            
            # Update States
            # Only update if not zero_rate (zero rate means no reaction)
            valid_update = ~zero_rate_mask
            
            # Update X
            # R1: X+1, R2: X-1
            X[active_indices[valid_update & reaction_1]] += 1
            X[active_indices[valid_update & reaction_2]] -= 1
            
            # Update Y
            # R3: Y+1, R4: Y-1
            Y[active_indices[valid_update & reaction_3]] += 1
            Y[active_indices[valid_update & reaction_4]] -= 1
            
            # Update Time
            t[active] = t_next
            
            # 6. Check Simulation End
            # End if t >= T_MAX or if all observations recorded
            done_time = t[active] >= self.t_max
            done_obs = next_obs_idx[active] >= self.n_obs
            
            # Mark finished
            # We can stop if we passed T_MAX.
            # If we passed T_MAX, we should have recorded all observations <= T_MAX.
            # (The loop above ensures we recorded everything < t_next)
            
            done = done_time | done_obs
            if np.any(done):
                # For finished sims, we don't simulate anymore
                # But we might need to fill remaining observations if we jumped way past T_MAX?
                # The loop above handles obs < t_next. 
                # If t_next >> T_MAX, we recorded all.
                
                finished_indices_local = np.where(done)[0]
                finished_indices_global = active_indices[finished_indices_local]
                
                active[finished_indices_global] = False
                
        # Fill any remaining observations (e.g. if we hit max_steps or somehow missed exactly T_MAX)
        # Usually for Gillespie, we stop when t > T_MAX.
        # If t exceeds T_MAX, the last state persists until infinity.
        # So any unrecorded observations should be filled with the final state.
        
        for i in range(batch_size):
            if next_obs_idx[i] < self.n_obs:
                observations[i, next_obs_idx[i]:, 0] = X[i]
                observations[i, next_obs_idx[i]:, 1] = Y[i]
                
        return observations


# ============================================================================
# Testing and Plotting
# ============================================================================

def plot_trajectories(obs, theta, filename="lv_trajectories.png"):
    """
    Plot X and Y trajectories.
    obs: (batch, n_obs, 2)
    theta: (batch, 4)
    """
    n_plot = min(5, obs.shape[0])
    times = OBS_TIMES
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(n_plot):
        # theta = [log(c1), log(c2), log(c3), log(c4)]
        # We can just show first few digits
        p = theta[i]
        label = f"$\\theta$=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}, {p[3]:.2f}]"
        
        # Predators (X)
        axes[0].plot(times, obs[i, :, 0], label=label, alpha=0.7)
        # Prey (Y)
        axes[1].plot(times, obs[i, :, 1], label=label, alpha=0.7)
        
    axes[0].set_title("Predators (X)")
    axes[0].set_ylabel("Population")
    axes[0].set_xlabel("Time")
    axes[0].legend(fontsize='x-small')
    
    axes[1].set_title("Prey (Y)")
    axes[1].set_ylabel("Population")
    axes[1].set_xlabel("Time")
    axes[1].legend(fontsize='x-small')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved trajectories to {filename}")
    plt.close()

if __name__ == "__main__":
    # Create results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    task = LVTask()
    
    print("=== Lotka-Volterra Data Generation Test ===")
    
    # 1. Test Ground Truth Simulation
    print(f"\n1. Simulating Ground Truth (Theta*)...")
    theta_true = task.theta_true
    # Repeat for batch
    batch_size = 10
    theta_fixed = np.tile(theta_true, (batch_size, 1))
    
    start_time = time.time()
    obs_fixed = task.simulator(theta_fixed)
    duration = time.time() - start_time
    print(f"   Simulation took {duration:.4f}s for {batch_size} trajectories")
    print(f"   Observation shape: {obs_fixed.shape}")
    
    plot_trajectories(obs_fixed, theta_fixed, os.path.join(results_dir, "lv_ground_truth.png"))
    
    # 2. Test Vague Prior
    print(f"\n2. Simulating Vague Prior (N=10000)...")
    theta_vague = task.sample_prior(10000, "vague")
    
    start_time = time.time()
    obs_vague = task.simulator(theta_vague)
    duration = time.time() - start_time
    print(f"   Simulation took {duration:.4f}s for 10000 trajectories")
    
    plot_trajectories(obs_vague, theta_vague, os.path.join(results_dir, "lv_vague_prior.png"))
    
    # 3. Test Informative Prior
    print(f"\n3. Simulating Informative Prior (N=10000)...")
    theta_info = task.sample_prior(10000, "informative")
    
    start_time = time.time()
    obs_info = task.simulator(theta_info)
    duration = time.time() - start_time
    print(f"   Simulation took {duration:.4f}s for 10000 trajectories")
    
    plot_trajectories(obs_info, theta_info, os.path.join(results_dir, "lv_informative_prior.png"))
    
    print("\nDone.")
