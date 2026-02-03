"""
BayesFlow implementation for Stochastic SIR.
Uses PyTorch backend and MPS acceleration.
"""

import os
# Set backend BEFORE imports
os.environ["KERAS_BACKEND"] = "torch"

# Workaround for MPS issues with Keras 3 + BayesFlow (Meta device mismatch)
import torch
# Patch is_available to False to force CPU
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False


import numpy as np
import keras
import bayesflow as bf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Force CPU device for Keras to avoid MPS issues
# Note: Keras 3 with Torch backend on MPS can be unstable for mixed device operations (meta/mps)
keras.config.set_floatx("float32")
# There is no direct "disable MPS" in Keras config, but we can try to rely on torch not being default mps
# or we can patch torch.backends.mps.is_available = lambda: False if needed.

# Import from local data_generation
try:
    from data_generation import (
        simulator, 
        sample_prior,
        TRUE_PARAMS,
        PRIOR_MIN,
        PRIOR_MAX,
        d, d_x, NUM_OBS
    )
except ImportError:
    from stochastic_sir.data_generation import (
        simulator, 
        sample_prior,
        TRUE_PARAMS,
        PRIOR_MIN,
        PRIOR_MAX,
        d, d_x, NUM_OBS
    )

# ============================================================================
# 1. Configuration
# ============================================================================
SUMMARY_DIM = 10
BATCH_SIZE = 64
EPOCHS = 5
ITERATIONS_PER_EPOCH = 500 # Offline training simulation

# ============================================================================
# 2. Neural Networks (Consistent with SMMD)
# ============================================================================

class InvariantModule(keras.layers.Layer):
    def __init__(self, settings, **kwargs):
        super().__init__(**kwargs)
        self.s1_layers = []
        in_dim = settings["input_dim"]
        for i in range(settings["num_dense_s1"]):
            self.s1_layers.append(keras.layers.Dense(settings["dense_s1_args"]["units"], activation="relu"))
        
        self.s2_layers = []
        for i in range(settings["num_dense_s2"]):
            self.s2_layers.append(keras.layers.Dense(settings["dense_s2_args"]["units"], activation="relu"))

    def call(self, x):
        # x: (batch, n_points, input_dim)
        out = x
        for layer in self.s1_layers:
            out = layer(out)
        
        # Pooling (mean over n_points)
        out_reduced = keras.ops.mean(out, axis=1)
        
        for layer in self.s2_layers:
            out_reduced = layer(out_reduced)
            
        return out_reduced
        
    def compute_output_shape(self, input_shape):
        # Output is (batch, s2_units)
        return (input_shape[0], self.s2_layers[-1].units)

class EquivariantModule(keras.layers.Layer):
    def __init__(self, settings, invariant_module, **kwargs):
        super().__init__(**kwargs)
        self.invariant_module = invariant_module
        
        self.s3_layers = []
        for i in range(settings["num_dense_s3"]):
            self.s3_layers.append(keras.layers.Dense(settings["dense_s3_args"]["units"], activation="relu"))

    def call(self, x):
        # x: (batch, n_points, input_dim)
        batch_size = keras.ops.shape(x)[0]
        n_points = keras.ops.shape(x)[1]
        
        # Invariant path
        out_inv = self.invariant_module(x) # (batch, inv_out_dim)
        
        # Expand: (batch, n_points, inv_out_dim)
        # Keras 3 repeat/tile
        out_inv_rep = keras.ops.repeat(keras.ops.expand_dims(out_inv, axis=1), repeats=n_points, axis=1)
        
        # Concat: (batch, n_points, input_dim + inv_out_dim)
        out_c = keras.ops.concatenate([x, out_inv_rep], axis=-1)
        
        out = out_c
        for layer in self.s3_layers:
            out = layer(out)
            
        return out
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.s3_layers[-1].units)

class DeepSetsSummary(keras.Model):
    def __init__(self, n_points=NUM_OBS, input_dim=d_x, output_dim=SUMMARY_DIM, **kwargs):
        super().__init__(**kwargs)
        
        settings = dict(
            num_dense_s1=2, num_dense_s2=2, num_dense_s3=2,
            dense_s1_args={"units": 32},
            dense_s2_args={"units": 32},
            dense_s3_args={"units": 32},
            input_dim=input_dim
        )
        
        # Layer 1: Equivariant
        self.inv1 = InvariantModule(settings) # Helper for Eq1
        # Re-create internal structure for Equivariant 1
        # Actually, Equivariant wraps Invariant.
        # But Invariant takes same input.
        # Let's simplify and just define layers directly or use the class structure.
        # The SMMD implementation nests them.
        
        self.equiv1 = EquivariantModule(settings, InvariantModule(settings))
        
        # Layer 2
        settings_l2 = settings.copy()
        settings_l2["input_dim"] = 32 # Output of Equiv1
        self.equiv2 = EquivariantModule(settings_l2, InvariantModule(settings_l2))
        
        # Layer 3: Invariant
        settings_l3 = settings.copy()
        settings_l3["input_dim"] = 32 # Output of Equiv2
        self.inv3 = InvariantModule(settings_l3)
        
        self.out_layer = keras.layers.Dense(output_dim)
        
    def call(self, x):
        x = self.equiv1(x)
        x = self.equiv2(x)
        x = self.inv3(x)
        return self.out_layer(x)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_layer.units)

    def compute_metrics(self, x, stage=None):
        out = self(x)
        return {"outputs": out}

# ============================================================================
# 3. Generative Model & Workflow
# ============================================================================

def get_approximator(adapter):
    # 1. Summary Network
    summary_net = DeepSetsSummary()
    
    # 2. Inference Network (Invertible)
    inference_net = bf.networks.CouplingFlow(
        num_params=d,
        num_coupling_layers=4,
        coupling_settings={"dense_args": dict(units=64, activation="relu")}
    )
    
    # 3. Approximator
    approximator = MyContinuousApproximator(
        inference_network=inference_net,
        summary_network=summary_net,
        adapter=adapter
    )
    
    return approximator

# Custom Approximator to handle Keras 3 Dict inputs
class MyContinuousApproximator(bf.ContinuousApproximator):
    def call(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return self.compute_metrics(**inputs, **kwargs)
        return super().call(inputs, **kwargs)

# ============================================================================
# 4. Training
# ============================================================================

def train_bayesflow():
    print("Initializing BayesFlow...")
    
    # Offline Data Generation
    print("Generating training data...")
    N_TRAIN = 5000
    theta_train = sample_prior(N_TRAIN)
    x_train = simulator(theta_train, n_points=NUM_OBS)
    
    # Convert to Dict for BayesFlow
    train_data = {
        "inference_variables": theta_train.astype(np.float32),
        "summary_variables": x_train.astype(np.float32)
    }
    
    # Adapter
    adapter = bf.ContinuousApproximator.build_adapter(
        inference_variables=["inference_variables"],
        summary_variables=["summary_variables"]
    )
    
    # Create OfflineDataset
    dataset = bf.OfflineDataset(
        data=train_data,
        adapter=adapter,
        batch_size=BATCH_SIZE
    )
    
    approximator = get_approximator(adapter)
    
    # Compile
    approximator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )

    # Build Model manually to avoid Keras symbolic build issues
    print("Building model manually...")
    dummy_data = {
        "inference_variables": theta_train[:2].astype(np.float32),
        "summary_variables": x_train[:2].astype(np.float32)
    }
    # We need to call compute_metrics directly or call the model
    # MyContinuousApproximator.call handles dict input now
    try:
        approximator(dummy_data)
        print("Model built successfully.")
    except Exception as e:
        print(f"Manual build failed: {e}")
        # Proceed anyway, maybe fit will work if build succeeded partially
    
    # Custom Training Loop
    print("Starting custom training loop...")
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    approximator.compile(optimizer=optimizer)
    
    import time
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        batch_losses = []
        for i, batch in enumerate(dataset):
            # OfflineDataset with adapter usually returns dict
            # train_step expects data
            try:
                metrics = approximator.train_step(batch)
                # metrics is a dict
                loss = metrics["loss"] if "loss" in metrics else list(metrics.values())[0]
                batch_losses.append(float(loss))
            except Exception as e:
                print(f"Error in train_step: {e}")
                break
        
        if batch_losses:
            print(f"Mean Loss: {np.mean(batch_losses):.4f}")
        else:
            print("No batches processed.")
            break
            
    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # Save model (optional)
    # approximator.save("bayesflow_sir_model.keras")
    
    return approximator
    
    # Plot Loss
    plt.figure()
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.legend()
    os.makedirs("stochastic_sir/bayesflow_result", exist_ok=True)
    plt.savefig("stochastic_sir/bayesflow_result/loss.png")
    
    return approximator

# ============================================================================
# 5. Evaluation
# ============================================================================

def evaluate(model):
    print("Evaluating...")
    # Observation
    x_obs = simulator(TRUE_PARAMS, n_points=NUM_OBS) # (1, N, 2)
    
    # Sample Posterior
    # Note: sample requires keyword arguments
    samples = model.sample(conditions={"summary_variables": x_obs}, num_samples=1000)
    
    if isinstance(samples, dict):
        samples = samples["inference_variables"]
    
    # samples shape: (1, 1000, d) -> flatten to (1000, d)
    samples = samples.reshape(-1, d)
    
    # Plot
    df = pd.DataFrame(samples, columns=["beta", "gamma"])
    g = sns.pairplot(df, diag_kind="kde", corner=True)
    g.fig.suptitle("BayesFlow Posterior (SIR)", y=1.02)
    
    axes = g.axes
    axes[0,0].axvline(TRUE_PARAMS[0], color='r', linestyle='--')
    axes[1,0].scatter(TRUE_PARAMS[0], TRUE_PARAMS[1], color='r', s=50, zorder=5)
    axes[1,1].axvline(TRUE_PARAMS[1], color='r', linestyle='--')
    
    # Ensure directory exists
    os.makedirs("stochastic_sir/bayesflow_result", exist_ok=True)
    plt.savefig("stochastic_sir/bayesflow_result/posterior.png")
    print("Posterior plot saved.")

if __name__ == "__main__":
    model = train_bayesflow()
    evaluate(model)
