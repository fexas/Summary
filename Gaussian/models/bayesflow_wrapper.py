
import os
import numpy as np
import torch
import torch.nn as nn

# Try importing BayesFlow and Keras
try:
    # Force torch backend for Keras 3
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    import bayesflow as bf
    
    # MPS Workaround
    if hasattr(torch.backends, "mps"):
        # We might need to disable MPS for Keras/BayesFlow if it causes issues
        # But let's try to keep it if possible, or disable if user memory suggests
        pass
        
    BAYESFLOW_AVAILABLE = True
except ImportError:
    BAYESFLOW_AVAILABLE = False
    print("BayesFlow or Keras not available.")

class BayesFlowWrapper:
    """
    Wrapper for BayesFlow model to be compatible with Refinement Utilities.
    Exposes:
    - .T(x_obs): Returns summary statistics
    - .sample_posterior(x_obs, n_samples): Returns posterior samples
    """
    def __init__(self, approximator, d=5):
        self.approximator = approximator
        self.d = d
        self.summary_net = approximator.summary_net
        
    def T(self, x_obs):
        # x_obs: torch tensor (batch, n, d_x)
        # BayesFlow expects numpy or tensor. Keras 3 with torch backend handles tensors.
        # But we need to ensure shape is correct.
        
        # Check if x_obs is tensor
        if isinstance(x_obs, torch.Tensor):
            x_obs_np = x_obs.detach().cpu().numpy()
        else:
            x_obs_np = x_obs
            
        # BayesFlow Summary Net Call
        # summary_net(x)
        # Output is (batch, p)
        
        # We need to make sure we return a torch tensor for compatibility with utilities
        device = torch.device("cpu") # Default to cpu for utilities unless specified
        if isinstance(x_obs, torch.Tensor):
            device = x_obs.device
            
        # Call model
        # Keras model call
        # We might need to cast to float32
        x_obs_np = x_obs_np.astype(np.float32)
        
        # Assuming summary_net returns tensor or numpy
        # With Keras Torch backend, it returns Torch tensor?
        # Let's verify behavior or cast safely.
        
        out = self.summary_net(x_obs_np)
        
        if not isinstance(out, torch.Tensor):
            out = torch.from_numpy(np.array(out))
            
        return out.to(device)

    def sample_posterior(self, x_obs, n_samples):
        # x_obs: (1, n, d_x) or (n, d_x)
        # Returns: (n_samples, d)
        
        if isinstance(x_obs, torch.Tensor):
            x_obs = x_obs.detach().cpu().numpy()
            
        # Handle shape: BayesFlow expects (batch, n, d_x)
        if x_obs.ndim == 2:
            x_obs = x_obs[np.newaxis, ...]
            
        # Conditions
        conditions = {"summary_variables": x_obs.astype(np.float32)}
        
        # Sample
        # num_samples per batch item
        samples = self.approximator.sample(conditions=conditions, num_samples=n_samples)
        
        # samples: dict or array
        if isinstance(samples, dict):
            samples = samples["inference_variables"]
            
        # Shape: (1, n_samples, d) -> (n_samples, d)
        return samples.reshape(-1, self.d)
        
    def train_step(self, data):
        return self.approximator.train_step(data)

    def compute_stats(self, x):
        return self.T(x)

# Factory function to build the model
def build_bayesflow_model(input_dim=3, summary_dim=10, d=5):
    if not BAYESFLOW_AVAILABLE:
        raise ImportError("BayesFlow not installed.")
        
    # Define Networks (using Keras)
    
    # Invariant Module
    class InvariantModule(keras.layers.Layer):
        def __init__(self, settings, **kwargs):
            super().__init__(**kwargs)
            self.s1_layers = []
            for i in range(settings["num_dense_s1"]):
                self.s1_layers.append(keras.layers.Dense(settings["dense_s1_args"]["units"], activation="relu"))
            self.s2_layers = []
            for i in range(settings["num_dense_s2"]):
                self.s2_layers.append(keras.layers.Dense(settings["dense_s2_args"]["units"], activation="relu"))

        def call(self, x):
            out = x
            for layer in self.s1_layers:
                out = layer(out)
            out_reduced = keras.ops.mean(out, axis=1)
            for layer in self.s2_layers:
                out_reduced = layer(out_reduced)
            return out_reduced

    class EquivariantModule(keras.layers.Layer):
        def __init__(self, settings, invariant_module, **kwargs):
            super().__init__(**kwargs)
            self.invariant_module = invariant_module
            self.s3_layers = []
            for i in range(settings["num_dense_s3"]):
                self.s3_layers.append(keras.layers.Dense(settings["dense_s3_args"]["units"], activation="relu"))

        def call(self, x):
            n_points = keras.ops.shape(x)[1]
            out_inv = self.invariant_module(x)
            out_inv_rep = keras.ops.repeat(keras.ops.expand_dims(out_inv, axis=1), repeats=n_points, axis=1)
            out_c = keras.ops.concatenate([x, out_inv_rep], axis=-1)
            out = out_c
            for layer in self.s3_layers:
                out = layer(out)
            return out

    class DeepSetsSummary(keras.Model):
        def __init__(self, input_dim=input_dim, output_dim=summary_dim, **kwargs):
            super().__init__(**kwargs)
            settings = dict(
                num_dense_s1=2, num_dense_s2=2, num_dense_s3=2,
                dense_s1_args={"units": 64},
                dense_s2_args={"units": 64},
                dense_s3_args={"units": 64},
                input_dim=input_dim
            )
            self.inv1 = InvariantModule(settings)
            self.equiv1 = EquivariantModule(settings, InvariantModule(settings))
            
            settings_l2 = settings.copy()
            settings_l2["input_dim"] = 64 # output of s3
            self.equiv2 = EquivariantModule(settings_l2, InvariantModule(settings_l2))
            
            settings_l3 = settings.copy()
            settings_l3["input_dim"] = 64
            self.inv3 = InvariantModule(settings_l3)
            
            self.out_layer = keras.layers.Dense(output_dim)
            
        def call(self, x):
            x = self.equiv1(x)
            x = self.equiv2(x)
            x = self.inv3(x)
            return self.out_layer(x)

    summary_net = DeepSetsSummary()
    
    # Inference Net
    inference_net = bf.networks.CouplingFlow(
        num_params=d,
        num_coupling_layers=4,
        coupling_settings={"dense_args": dict(units=64, activation="relu")}
    )
    
    # Adapter for Keras 3 dicts
    class MyContinuousApproximator(bf.ContinuousApproximator):
        def call(self, inputs, **kwargs):
            if isinstance(inputs, dict):
                return self.compute_metrics(**inputs, **kwargs)
            return super().call(inputs, **kwargs)

    adapter = bf.ContinuousApproximator.build_adapter(
        inference_variables=["inference_variables"],
        summary_variables=["summary_variables"]
    )
    
    approximator = MyContinuousApproximator(
        inference_network=inference_net,
        summary_network=summary_net,
        adapter=adapter
    )
    
    return BayesFlowWrapper(approximator, d=d)
