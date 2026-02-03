"""
BayesFlow implementation for g-and-k distribution inference.
Uses the same DeepSets summary network architecture as smmd_torch.py.
Uses Invertible Network (NPE) for posterior estimation.
"""

import os
os.environ["KERAS_BACKEND"] = "torch" # Force PyTorch backend

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import keras
from keras import layers, ops
import bayesflow as bf
from bayesflow.networks import SummaryNetwork
import torch

# Import from local data_generation
try:
    from data_generation import (
        simulator, 
        prior_generator,
        PRIOR_CONFIGS, 
        TRUE_PARAMS,
        n, d, d_x
    )
except ImportError:
    from G_and_K.data_generation import (
        simulator, 
        prior_generator,
        PRIOR_CONFIGS, 
        TRUE_PARAMS,
        n, d, d_x
    )

# ============================================================================
# 1. Configuration & Device
# ============================================================================
p = 10      # Summary statistics dimension
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_SIMULATIONS = 5000 # Number of simulations per epoch for online training or total for offline

# Check Device
if torch.backends.mps.is_available():
    print("Using MPS (Apple Silicon) acceleration via PyTorch backend.")
    # Keras 3 with PyTorch backend usually follows PyTorch's default device or placement
    # We might need to configure it explicitly if Keras doesn't pick it up
    torch.set_default_device("mps")
elif torch.cuda.is_available():
    print("Using CUDA acceleration.")
    torch.set_default_device("cuda")
else:
    print("Using CPU.")

# ============================================================================
# 2. Custom Summary Network (DeepSets) - Keras 3 Implementation
# ============================================================================

class InvariantModule(keras.Layer):
    def __init__(self, settings, **kwargs):
        super().__init__(**kwargs)
        
        # S1: Dense layers before pooling
        self.s1 = keras.Sequential([
            layers.Dense(settings["dense_s1_args"]["units"], activation="relu")
            for _ in range(settings["num_dense_s1"])
        ])
        
        # S2: Dense layers after pooling
        self.s2 = keras.Sequential([
            layers.Dense(settings["dense_s2_args"]["units"], activation="relu")
            for _ in range(settings["num_dense_s2"])
        ])

    def call(self, x):
        # x: (batch, n_points, input_dim)
        x_s1 = self.s1(x) # (batch, n_points, s1_out_dim)
        # Mean pooling over n_points (axis 1)
        x_reduced = ops.mean(x_s1, axis=1) # (batch, s1_out_dim)
        return self.s2(x_reduced) # (batch, s2_out_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s2.layers[-1].units)

class EquivariantModule(keras.Layer):
    def __init__(self, settings, **kwargs):
        super().__init__(**kwargs)
        self.invariant_module = InvariantModule(settings)
        
        # S3: Dense layers
        self.s3 = keras.Sequential([
            layers.Dense(settings["dense_s3_args"]["units"], activation="relu")
            for _ in range(settings["num_dense_s3"])
        ])

    def call(self, x):
        # x: (batch, n_points, input_dim)
        n_points = ops.shape(x)[1]
        
        # Invariant path
        out_inv = self.invariant_module(x) # (batch, inv_out_dim)
        
        # Expand and repeat: (batch, n_points, inv_out_dim)
        # expand_dims -> (batch, 1, inv_out_dim)
        # repeat -> (batch, n_points, inv_out_dim)
        out_inv_rep = ops.repeat(ops.expand_dims(out_inv, 1), n_points, axis=1)
        
        # Concatenate: (batch, n_points, input_dim + inv_out_dim)
        out_c = ops.concatenate([x, out_inv_rep], axis=-1)
        
        return self.s3(out_c) # (batch, n_points, s3_out_dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.s3.layers[-1].units)

class DeepSetsSummary(SummaryNetwork):
    def __init__(self, n_points=n, input_dim=d_x, output_dim=p, **kwargs):
        super().__init__(**kwargs)
        
        settings = dict(
            num_dense_s1=2, num_dense_s2=2, num_dense_s3=2,
            dense_s1_args={"units": 32},
            dense_s2_args={"units": 32},
            dense_s3_args={"units": 32}
        )
        
        # Layer 1: Equivariant
        self.equiv1 = EquivariantModule(settings)
        
        # Layer 2: Equivariant
        self.equiv2 = EquivariantModule(settings)
        
        # Layer 3: Invariant
        self.inv = InvariantModule(settings)
        
        # Output Layer
        self.out_layer = layers.Dense(output_dim, activation='linear')
        
    def call(self, x, **kwargs):
        # x: (batch, n_points, input_dim)
        x = self.equiv1(x)
        x = self.equiv2(x)
        x = self.inv(x)
        return self.out_layer(x)

# ============================================================================
# 3. Generative Model (Prior + Simulator)
# ============================================================================

def get_generative_model(prior_type='weak_informative'):
    
    def prior_fun():
        # Returns (d_params,)
        return prior_generator(prior_type).astype(np.float32)
        
    def simulator_fun(theta):
        # BayesFlow passes theta as (d_params,) for single sim or (batch, d) for batch
        # Our simulator handles (batch, d) or (d,) and returns (batch, n, 1)
        # BayesFlow expects (n_obs, d_obs) for single sim
        
        # Ensure numpy
        theta = np.array(theta, dtype=np.float32)
        
        # Check if batch
        if theta.ndim == 1:
            x = simulator(theta, n_samples=n) # (1, n, 1)
            return x[0] # (n, 1)
        else:
            x = simulator(theta, n_samples=n) # (batch, n, 1)
            return x
            
    return prior_fun, simulator_fun

# ============================================================================
# 4. Main Training Workflow
# ============================================================================

class MyContinuousApproximator(bf.ContinuousApproximator):
    def call(self, inputs, **kwargs):
        # Fix for Keras 3 passing dictionary as single argument
        if isinstance(inputs, dict):
            return self.compute_metrics(**inputs, **kwargs)
        return super().call(inputs, **kwargs)

def train_bayesflow(prior_type='weak_informative', result_dir='bayesflow_result'):
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Setting up BayesFlow for {prior_type} prior...")
    
    # 1. Networks
    summary_net = DeepSetsSummary(output_dim=p)
    
    # Build summary network manually to avoid Keras 3/PyTorch build issues
    dummy_summary_input = np.zeros((2, n, 1), dtype=np.float32)
    summary_net(dummy_summary_input)
    
    inference_net = bf.networks.CouplingFlow(num_params=d, num_coupling_layers=4)
    
    # 2. Adapter
    # We must explicitly map dictionary keys to BayesFlow's expected variable names
    adapter = bf.ContinuousApproximator.build_adapter(
        inference_variables=["inference_variables"],
        summary_variables=["summary_variables"]
    )
    
    # 3. Approximator (Custom to fix call() issue)
    approximator = MyContinuousApproximator(
        inference_network=inference_net,
        summary_network=summary_net,
        adapter=adapter,
        standardize=None # Disable standardization to avoid MPS/Meta device mismatch
    )
    
    # 4. Data Generation (Reference Table)
    print("Generating Reference Table (Training Data)...")
    prior_fun, simulator_fun = get_generative_model(prior_type)
    
    # Generate batch
    train_data_size = 10000
    train_params = []
    for _ in range(train_data_size):
        train_params.append(prior_fun())
    train_params = np.array(train_params) # (N, 4)
    train_data = simulator_fun(train_params) # (N, n, 1)
    
    # Convert to dict for BayesFlow
    train_dict = {
        "inference_variables": train_params,
        "summary_variables": train_data # Input to summary net
    }
    
    # 5. Dataset
    dataset = bf.datasets.OfflineDataset(
        data=train_dict,
        adapter=adapter,
        batch_size=BATCH_SIZE
    )
    
    print("Starting Training...")
    
    # Compile the approximator
    approximator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    )
    
    # Fit directly using the approximator
    history = approximator.fit(
        dataset=dataset,
        epochs=EPOCHS
    )
    
    # Save Loss
    loss = history.history['loss']
    plt.figure()
    plt.plot(loss)
    plt.title(f"BayesFlow Loss ({prior_type})")
    plt.savefig(os.path.join(result_dir, f"loss_{prior_type}.png"))
    plt.close()
    
    return approximator

def evaluate(model, prior_type, result_dir):
    print(f"Evaluating {prior_type}...")
    
    # Ground Truth
    theta_true, x_obs = TRUE_PARAMS, simulator(TRUE_PARAMS, n_samples=n)[0]
    # x_obs: (n, 1)
    
    # BayesFlow expects (batch, n, 1) for prediction
    x_obs_batch = x_obs[np.newaxis, ...] 
    
    # Sample from posterior
    # BasicWorkflow.sample expects dict or conditions
    input_dict = {"summary_variables": x_obs_batch}
    samples_dict = model.sample(conditions=input_dict, num_samples=2000)
    
    # samples shape: (batch, n_samples, d) -> (1, 2000, 4)
    samples = samples_dict['inference_variables'][0]
    
    # Plot Corner Plot
    param_names = ['A', 'B', 'g', 'k']
    df = pd.DataFrame(samples, columns=param_names)
    
    # Prior limits for plotting
    limit = PRIOR_CONFIGS[prior_type]['bounds_limit'] if 'bounds_limit' in PRIOR_CONFIGS[prior_type] else 10.0
    plot_limit = limit * 1.1
    
    g = sns.PairGrid(df, diag_sharey=False, corner=True)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.kdeplot, fill=True)
    
    # Set limits
    for i in range(d):
        for j in range(i + 1):
             if i == j:
                 g.diag_axes[i].set_xlim(-plot_limit, plot_limit)
             else:
                 g.axes[i, j].set_xlim(-plot_limit, plot_limit)
                 g.axes[i, j].set_ylim(-plot_limit, plot_limit)

    # Mark True Params
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                g.diag_axes[i].axvline(theta_true[i], color='r', linestyle='--')
            else:
                g.axes[i, j].scatter(theta_true[j], theta_true[i], color='r', marker='*', s=100)
                
    g.fig.suptitle(f"BayesFlow Posterior ({prior_type})", y=1.02)
    plt.savefig(os.path.join(result_dir, f"posterior_bayesflow_{prior_type}.png"))
    plt.close()

if __name__ == "__main__":
    result_dir = "bayesflow_result"
    prior_type = "weak_informative"
    
    model = train_bayesflow(prior_type, result_dir)
    evaluate(model, prior_type, result_dir)
    print("Done.")
