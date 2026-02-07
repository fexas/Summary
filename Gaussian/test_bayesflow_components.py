
import os
os.environ["KERAS_BACKEND"] = "torch"
import bayesflow as bf
import torch

try:
    print(f"BayesFlow version: {bf.__version__}")
except:
    pass

try:
    net = bf.networks.InferenceNetwork(num_params=5)
    print("InferenceNetwork created successfully.")
except Exception as e:
    print(f"Error creating InferenceNetwork: {e}")

try:
    summary_net = bf.networks.DeepSet(summary_dim=10)
    print("DeepSet created successfully.")
except Exception as e:
    print(f"Error creating DeepSet: {e}")
