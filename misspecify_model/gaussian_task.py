import torch
import numpy as np

class GaussianTask:
    def __init__(self, device='cpu'):
        self.x_raw_dim = 100
        self.prior_var = 25.0
        self.likelihood_var = 1.0
        self.dgp_var = 2.0 # Misspecified variance
        self.device = device

    def sample_prior(self, num_samples):
        # theta ~ N(0, 25)
        # return shape: (num_samples, 1)
        return torch.randn(num_samples, 1, device=self.device) * np.sqrt(self.prior_var)

    def simulate(self, theta):
        # x_raw ~ N(theta, 1)
        # theta: (batch, 1)
        # x: (batch, 100)
        batch_size = theta.shape[0]
        epsilon = torch.randn(batch_size, self.x_raw_dim, device=self.device) * np.sqrt(self.likelihood_var)
        x = theta + epsilon
        return x

    def get_summary_stats(self, x):
        # x: (batch, 100)
        # return: (batch, 2) -> [mean, var]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True) # Unbiased estimator by default in torch
        return torch.cat([mean, var], dim=1)

    def generate_observation(self, misspecified=True):
        # Generate single observation from true process
        theta_true = self.sample_prior(1)
        
        var = self.dgp_var if misspecified else self.likelihood_var
        
        epsilon = torch.randn(1, self.x_raw_dim, device=self.device) * np.sqrt(var)
        y_raw = theta_true + epsilon
        
        y_summary = self.get_summary_stats(y_raw)
        
        return theta_true, y_raw, y_summary

    def get_true_posterior_samples(self, obs_mean, num_samples, use_dgp=True):
        # Analytic posterior
        # If use_dgp=True, use dgp_var (correct posterior)
        # If use_dgp=False, use likelihood_var (what simulator thinks)
        
        l_var = self.dgp_var if use_dgp else self.likelihood_var
        p_var = self.prior_var
        n = self.x_raw_dim
        
        # Likelihood: N(mean_obs | theta, l_var/n)
        # Prior: N(theta | 0, p_var)
        # Posterior variance: (1/p_var + n/l_var)^-1
        # Posterior mean: post_var * (n/l_var * mean_obs)
        
        post_var = 1.0 / (1.0/p_var + n/l_var)
        post_mean = post_var * (n/l_var * obs_mean)
        
        post_std = np.sqrt(post_var)
        
        samples = torch.randn(num_samples, 1, device=self.device) * post_std + post_mean
        return samples
