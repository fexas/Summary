"""
BayesFlow implementation for Gaussian Task.
Refactored to match Stochastic SIR implementation style.
Uses PyTorch backend and MPS acceleration workaround.
"""

import os
# Set backend BEFORE imports
os.environ["KERAS_BACKEND"] = "torch"

import torch
# Patch is_available to False to force CPU if needed for Keras compatibility
# (Matches stochastic_sir implementation)
# if hasattr(torch.backends, "mps"):
#    torch.backends.mps.is_available = lambda: False

import numpy as np
import keras
import bayesflow as bf
from bayesflow.adapters.transforms import Transform

# Force CPU device for Keras to avoid MPS issues
keras.config.set_floatx("float32")

# ============================================================================
# 0. Custom Adapter Transform (Fix MPS Issue)
# ============================================================================

class ToNumpy(Transform):
    """
    Custom Transform to convert Torch Tensors (CPU or MPS) to Numpy arrays
    AND cast to float32.
    Replaces standard ToArray and ConvertDType to handle MPS devices and dicts gracefully.
    """
    def forward(self, data, **kwargs):
        if isinstance(data, dict):
            return {k: self._to_numpy(v) for k, v in data.items()}
        return self._to_numpy(data)
    
    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        return x.astype("float32")

    def inverse(self, data, **kwargs):
        """
        Inverse transform: Identity.
        BayesFlow calls this during sampling.
        """
        return data

# ============================================================================
# 1. Neural Networks
# ============================================================================

class InvariantModule(keras.layers.Layer):
    def __init__(self, settings, **kwargs):
        super().__init__(**kwargs)
        self.s1_layers = []
        # in_dim = settings["input_dim"] # Not explicitly used for layer creation in Keras 3
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
        n_points = keras.ops.shape(x)[1]
        
        # Invariant path
        out_inv = self.invariant_module(x) # (batch, inv_out_dim)
        
        # Expand: (batch, n_points, inv_out_dim)
        # Keras 3 repeat/tile. Note: expand_dims axis=1 makes it (batch, 1, inv_out_dim)
        out_inv_rep = keras.ops.repeat(keras.ops.expand_dims(out_inv, axis=1), repeats=n_points, axis=1)
        
        # Concat: (batch, n_points, input_dim + inv_out_dim)
        out_c = keras.ops.concatenate([x, out_inv_rep], axis=-1)
        
        out = out_c
        for layer in self.s3_layers:
            out = layer(out)
            
        return out
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.s3_layers[-1].units)

class TimeSeriesSummary(keras.Model):
    def __init__(self, input_dim, output_dim=10, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        # 1D CNN for Time Series Summary (MPS compatible)
        self.conv1 = keras.layers.Conv1D(filters=hidden_dim, kernel_size=10, padding="same", activation="relu")
        self.pool1 = keras.layers.MaxPooling1D(pool_size=2)
        self.conv2 = keras.layers.Conv1D(filters=hidden_dim*2, kernel_size=10, padding="same", activation="relu")
        self.global_pool = keras.layers.GlobalAveragePooling1D()
        self.dense = keras.layers.Dense(output_dim)
        
    def call(self, x):
        # x: (batch, time_steps, input_dim)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        return self.dense(x)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dense.units)

    def compute_metrics(self, x, stage=None):
        out = self(x)
        return {"outputs": out}

class DeepSetsSummary(keras.Model):
    def __init__(self, input_dim, output_dim=10, **kwargs):
        super().__init__(**kwargs)
        
        settings = dict(
            num_dense_s1=2, num_dense_s2=2, num_dense_s3=2,
            dense_s1_args={"units": 64},
            dense_s2_args={"units": 64},
            dense_s3_args={"units": 64},
            input_dim=input_dim
        )
        
        # Layer 1: Equivariant
        self.inv1 = InvariantModule(settings) 
        self.equiv1 = EquivariantModule(settings, self.inv1)
        
        # Layer 2
        settings_l2 = settings.copy()
        settings_l2["input_dim"] = 32 # Output of Equiv1
        self.inv2 = InvariantModule(settings_l2) # Helper for Equiv2
        self.equiv2 = EquivariantModule(settings_l2, self.inv2)
        
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
        
    def build(self, input_shape):
        super().build(input_shape)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_layer.units)

    def compute_metrics(self, x, stage=None):
        out = self(x)
        return {"outputs": out}

# Custom Approximator to handle Keras 3 Dict inputs
class MyContinuousApproximator(bf.ContinuousApproximator):
    def call(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return self.compute_metrics(**inputs, **kwargs)
        return super().call(inputs, **kwargs)

def build_bayesflow_model(d, d_x, summary_dim=10):
    """
    Builds the BayesFlow model components: SummaryNetwork, InferenceNetwork, and ContinuousApproximator.
    
    Args:
        d (int): Dimensionality of the parameters (theta).
        d_x (int): Dimensionality of the data (x).
        summary_dim (int): Dimensionality of the summary statistics output.
        
    Returns:
        model (bf.approximators.ContinuousApproximator): The compiled BayesFlow model.
    """
    # 1. Summary Network
    # Use TimeSeriesSummary for Lotka-Volterra (Time Series Data)
    summary_net = TimeSeriesSummary(input_dim=d_x, output_dim=summary_dim)
    
    # 2. Inference Network (Invertible)
    inference_net = bf.networks.CouplingFlow(
        num_params=d,
        num_coupling_layers=4,
        coupling_settings={"dense_args": dict(units=64, activation="relu")}
    )
    
    # 3. Adapter
    # Custom Adapter to handle MPS tensors (replaces default build_adapter)
    # The default build_adapter uses ToArray which fails on MPS tensors.
    # Also handles dtype conversion internally to avoid 'dict' object has no attribute 'astype' error.
    adapter = bf.Adapter([
        ToNumpy()
    ])
    
    # 4. Approximator
    approximator = MyContinuousApproximator(
        inference_network=inference_net,
        summary_network=summary_net,
        adapter=adapter
    )
    
    return approximator
