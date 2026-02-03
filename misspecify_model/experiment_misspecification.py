import sys
import os

# Set environment variables to avoid Arviz PermissionError
os.environ['ARVIZ_DATA'] = '/tmp/arviz_data'
os.environ['HOME'] = '/tmp'

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sbi.inference import SNPE
import numpy as np
import torch.distributions as dist

# Add path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Multivariate_Gaussian')))
from models import SMMD_Model, sliced_mmd_loss

from gaussian_task import GaussianTask

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use CPU for now to avoid potential MPS issues if any, or stick to device
    # MPS sometimes has issues with specific ops, but let's try.
    # If SMMD uses standard ops, it should be fine.
    # SBI/PyTorch interactions on MPS can be tricky. 
    # Let's default to cpu for safety unless user explicitly asks for GPU speedup, 
    # but for neural nets cpu might be slow. 
    # Given the task size (10000 samples, simple nets), CPU is fine.
    device = 'cpu' 
    print(f"Using device: {device}")

    task = GaussianTask(device=device)
    
    # 1. Generate Training Data
    print("Generating training data...")
    num_simulations = 10000
    theta = task.sample_prior(num_simulations)
    x_raw = task.simulate(theta)
    x_summary = task.get_summary_stats(x_raw)
    
    # 2. Train NPE (on summary stats)
    print("Training NPE...")
    # Prior for SBI
    prior_dist = dist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([5.0])) # sqrt(25)=5
    
    inference = SNPE(prior=prior_dist, density_estimator='maf')
    inference.append_simulations(theta.cpu(), x_summary.cpu())
    density_estimator = inference.train()
    posterior_npe = inference.build_posterior(density_estimator)
    
    # 3. Train SMMD (on raw data)
    print("Training SMMD...")
    # input_dim=1 because each point in the set is a scalar (1D)
    # User requested summary_dim = 2 * theta_dim = 2
    smmd_model = SMMD_Model(input_dim=1, summary_dim=2, theta_dim=1).to(device)
    optimizer = torch.optim.Adam(smmd_model.parameters(), lr=1e-3)
    
    batch_size = 128
    epochs = 100
    
    # x_raw: (N, 100) -> (N, 100, 1)
    dataset = torch.utils.data.TensorDataset(theta, x_raw.unsqueeze(-1)) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Bandwidth setting
    n_obs = 100
    bandwidth = 5.0 / n_obs

    smmd_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_theta, batch_x in dataloader:
            batch_theta = batch_theta.to(device)
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            
            # Generate M=10 samples per x
            M = 10
            # z shape: (batch, M, z_dim)
            z = torch.randn(batch_theta.shape[0], M, 1, device=device)
            
            theta_fake = smmd_model(batch_x, z) # (batch, M, 1)
            
            # Use user-specified bandwidth
            loss = sliced_mmd_loss(batch_theta, theta_fake, bandwidth=bandwidth)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
            
    # 4. Generate Misspecified Observation
    print("Generating misspecified observation...")
    # Increase misspecification to ensure NPE fails (OOD summary statistics)
    # Training: likelihood_var = 1.0
    # Observation: dgp_var = 10.0 (High variance mismatch)
    task.dgp_var = 10.0 
    
    # Generate observation from true process
    theta_true, y_raw, y_summary = task.generate_observation(misspecified=True)
    print(f"True Theta: {theta_true.item():.4f}")
    print(f"Observed Summary (Mean, Var): {y_summary[0].tolist()}")
    
    # 5. Evaluate
    print("Evaluating...")
    num_samples = 10000
    
    # True Posterior (Correct)
    # y_summary[0, 0] is the mean of the observation
    true_post_samples = task.get_true_posterior_samples(y_summary[0, 0], num_samples, use_dgp=True)
    
    # NPE Posterior
    # Sample from NPE
    npe_samples = posterior_npe.sample((num_samples,), x=y_summary.cpu())
    
    # SMMD Posterior
    smmd_model.eval()
    with torch.no_grad():
        # Generate samples
        z_eval = torch.randn(1, num_samples, 1, device=device)
        y_input = y_raw.unsqueeze(-1).to(device) # (1, 100, 1)
        smmd_samples = smmd_model(y_input, z_eval).squeeze(0) # (num_samples, 1)
        
    # 6. Plot
    print("Plotting...")
    plt.figure(figsize=(10, 6))
    
    # Convert to numpy
    true_samples_np = true_post_samples.cpu().numpy().flatten()
    npe_samples_np = npe_samples.cpu().numpy().flatten()
    smmd_samples_np = smmd_samples.cpu().numpy().flatten()
    
    # Style configuration based on user request
    # True: Green
    # NPE: Blue
    # SMMD: Orange
    
    sns.kdeplot(true_samples_np, label='True', color='green', linewidth=2)
    sns.kdeplot(npe_samples_np, label='NPE', color='skyblue', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(smmd_samples_np, label='SMMD', color='orange', fill=True, alpha=0.3, linewidth=2)
    
    # Remove vertical line for True Theta to match the reference style better (or keep it subtle)
    # The reference image doesn't explicitly show the true parameter value as a vertical line, 
    # but shows the "True" distribution. We'll keep the distribution.
    
    plt.legend(fontsize=12)
    plt.title('Posterior Inference', fontsize=16)
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.ylabel('') # Reference image has no y-label text, just ticks or empty
    plt.yticks([]) # Reference image has grid but maybe no y-ticks? Let's keep it simple.
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(__file__), 'misspecification_results.png')
    plt.xlim(-10, 10) # Fix x-axis range for better visualization
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    run()
