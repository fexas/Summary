
import os
import torch
import numpy as np
import keras

# Import model builders
from models.bayesflow_net import build_bayesflow_model
from models.smmd import SMMD_Model
from models.mmd import MMD_Model
from data_generation import d, d_x

def load_bayesflow_model(round_id, save_dir="saved_models/bayesflow", summary_dim=10):
    """
    Loads a saved BayesFlow model (weights only).
    Rebuilds the model structure and initializes it with dummy data before loading weights.
    
    Args:
        round_id (int or str): The round identifier.
        save_dir (str): Directory where models are saved.
        summary_dim (int): Dimension of summary statistics.
        
    Returns:
        model (keras.Model): The loaded BayesFlow model.
    """
    print(f"Loading BayesFlow model for round {round_id}...")
    
    # 1. Build Model Structure
    model = build_bayesflow_model(d, d_x, summary_dim)
    
    # 2. Initialize with dummy data (Required for Keras 3 build)
    # Using small dummy data to trigger build
    dummy_x = np.zeros((1, 10, d_x), dtype=np.float32)
    dummy_theta = np.zeros((1, d), dtype=np.float32)
    dummy_dict = {
        "inference_variables": dummy_theta,
        "summary_variables": dummy_x
    }
    
    try:
        # Trigger adapter
        if hasattr(model, "adapter"):
            _ = model.adapter(dummy_dict)
            
        # Trigger internal networks build
        _ = model.log_prob(dummy_dict)
        
        # Trigger top-level model build
        _ = model(dummy_dict)
        
        print("Model structure built and initialized.")
    except Exception as e:
        print(f"Warning during model initialization: {e}")

    # 3. Load Weights
    weights_path = os.path.join(save_dir, f"round_{round_id}.weights.h5")
    
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded BayesFlow weights from {weights_path}")
            return model
        except Exception as e:
            print(f"Failed to load weights from {weights_path}: {e}")
    else:
        print(f"Full weights file not found at {weights_path}")

    print("Attempting to load sub-network weights...")
    summary_path = os.path.join(save_dir, f"round_{round_id}_summary.weights.h5")
    inference_path_h5 = os.path.join(save_dir, f"round_{round_id}_inference.weights.h5")
    inference_path_pt = os.path.join(save_dir, f"round_{round_id}_inference.pt")
    
    if os.path.exists(summary_path):
        try:
            model.summary_network.load_weights(summary_path)
            print("Loaded Summary Network weights.")
        except Exception as e:
            print(f"Failed to load Summary Network weights: {e}")
    
    if os.path.exists(inference_path_pt):
        try:
            model.inference_network.load_state_dict(torch.load(inference_path_pt, map_location="cpu"))
            print("Loaded Inference Network state_dict.")
        except Exception as e:
            print(f"Failed to load Inference Network state_dict: {e}")
    elif os.path.exists(inference_path_h5):
        try:
            model.inference_network.load_weights(inference_path_h5)
            print("Loaded Inference Network weights (h5).")
        except Exception as e:
            print(f"Failed to load Inference Network weights (h5): {e}")
            
    return model

def load_torch_model(model_type, round_id, save_dir="saved_models", device="cpu"):
    """
    Loads PyTorch models (SMMD, MMD).
    """
    model_dir = os.path.join(save_dir, model_type)
    path = os.path.join(model_dir, f"round_{round_id}.pt")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
        
    if model_type == "smmd":
        model = SMMD_Model(summary_dim=10, d=d, d_x=d_x, n=50) # Note: n might need to match training
    elif model_type == "mmd":
        model = MMD_Model(summary_dim=10, d=d, d_x=d_x, n=50)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded {model_type} model from {path}")
    return model

if __name__ == "__main__":
    # Test loading
    try:
        model = load_bayesflow_model(round_id=1)
        print("Test load successful.")
    except Exception as e:
        print(f"Test load failed: {e}")
