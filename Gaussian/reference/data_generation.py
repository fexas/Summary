"""
This script handles data generation for a Toy Example in Simulation-Based Inference.
Key functionalities:
1. Definition of the Prior distribution and the Simulator Model.
2. Implementation of MCMC sampling to obtain reference posterior samples for observed data.
3. Generation of synthetic training datasets, including MMD weight calculations and sample selection.
"""
import os
import scipy
import numpy as np
import tensorflow as tf
import bayesflow as bf
import math
import pickle
from sklearn.metrics.pairwise import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
# 1. Data Generation Hyperparameters
# ============================================================================
# Data parameters
N = 12800  # Size of the training dataset
n = 50     # Number of samples per observation (sample size)
d = 5      # Dimension of parameter theta
d_x = 3    # Dimension of x
p = 10     # Dimension of summary statistics

# Create data storage folder
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# ============================================================================
# 2. Define Toy Example (Simulator Model & Prior)
# ============================================================================
# Classes for Toy Example

class Uniform:
    """
    Parent class for uniform distributions.
    """

    def __init__(self, n_dims):

        self.n_dims = n_dims

    def grad_log_p(self, x):
        """
        Calculate gradient of log p(x).
        :param x: rows are datapoints
        :return: d/dx log p(x)
        """

        x = np.asarray(x)
        assert (x.ndim == 1 and x.size == self.n_dims) or (
            x.ndim == 2 and x.shape[1] == self.n_dims
        ), "wrong size"

        return np.zeros_like(x)


class BoxUniform(Uniform):
    """
    Implements a uniform PDF constrained within a box.
    """

    def __init__(self, lower, upper):
        """
        :param lower: array with lower limits
        :param upper: array with upper limits
        """

        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        assert (
            lower.ndim == 1 and upper.ndim == 1 and lower.size == upper.size
        ), "wrong sizes"
        assert np.all(lower < upper), "invalid upper and lower limits"

        Uniform.__init__(self, lower.size)

        self.lower = lower
        self.upper = upper
        self.volume = np.prod(upper - lower)

    def eval(self, x, ii=None, log=True):
        """
        Evaluate probability at x.
        :param x: rows to evaluate
        :param ii: list of indices for marginal evaluation; if None, evaluate joint
        :param log: whether to return log probability
        :return: probability at rows of x
        """

        x = np.asarray(x)

        if x.ndim == 1:
            return self.eval(x[np.newaxis, :], ii, log)[0]

        if ii is None:

            in_box = np.logical_and(self.lower <= x, x <= self.upper)
            in_box = np.logical_and.reduce(in_box, axis=1)

            if log:
                prob = -float("inf") * np.ones(in_box.size, dtype=float)
                prob[in_box] = -np.log(self.volume)

            else:
                prob = np.zeros(in_box.size, dtype=float)
                prob[in_box] = 1.0 / self.volume

            return prob

        else:
            assert len(ii) > 0, "list of indices can" "t be empty"
            marginal = BoxUniform(self.lower[ii], self.upper[ii])
            return marginal.eval(x, None, log)

    def gen(self, n_samples=None, rng=np.random):
        """
        Generate samples.
        :param n_samples: int, number of samples to generate
        :return: numpy array, rows are samples. Returns 1 sample (vector) if None.
        """

        one_sample = n_samples is None
        u = rng.rand(1 if one_sample else n_samples, self.n_dims)
        x = (self.upper - self.lower) * u + self.lower

        return x[0] if one_sample else x


class SimulatorModel:
    """
    Base class for a simulator model.
    """

    def __init__(self):

        self.n_sims = 0

    def sim(self, ps):

        raise NotImplementedError("Simulator model must be implemented in a subclass")


class Stats:
    """
    Identity summary statistics (returns parameters directly).
    """

    def __init__(self):
        pass

    @staticmethod
    def calc(ps):
        return ps


def prepare_cond_input(xy, dtype):
    """
    Prepare conditional input for model evaluation.
    :param xy: tuple (x, y) for evaluating p(y|x)
    :param dtype: data type
    :return: prepared x, y and flag for single datapoint input
    """

    x, y = xy
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    one_datapoint = False

    if x.ndim == 1:

        if y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
            one_datapoint = True

        else:
            x = np.tile(x, [y.shape[0], 1])

    else:

        if y.ndim == 1:
            y = np.tile(y, [x.shape[0], 1])

        else:
            assert x.shape[0] == y.shape[0], "wrong sizes"

    return x, y, one_datapoint


class Model(SimulatorModel):
    """
    Concrete implementation of the simulator model.
    """

    def __init__(self):

        SimulatorModel.__init__(self)
        self.n_data = n

    def sim(self, ps, rng=np.random):
        """
        Simulate data at parameters ps.
        """

        ps = np.asarray(ps, float)

        if ps.ndim == 1:
            return self.sim(ps[np.newaxis, :], rng=rng)[0]

        n_sims = ps.shape[0]

        m0, m1, s0, s1, r = self._unpack_params(ps)

    def sim_preserved_shape(self, ps, rng=np.random):
        """
        Simulate data at parameters ps while preserving shape.
        """

        ps = np.asarray(ps, float)

        if ps.ndim == 1:
            return self.sim(ps[np.newaxis, :], rng=rng)[0]

        n_sims = ps.shape[0]

        m0, m1, s0, s1, r = self._unpack_params(ps)

        us = rng.randn(n_sims, self.n_data, 2)  
        xs = np.empty_like(us)

        xs[:, :, 0] = s0 * us[:, :, 0] + m0
        xs[:, :, 1] = s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r**2) * us[:, :, 1]) + m1

        self.n_sims += n_sims  

        return xs

    def eval(self, px, log=True):
        """
        Evaluate probability of data given parameters.
        """

        ps, xs, one_datapoint = prepare_cond_input(px, float)

        m0, m1, s0, s1, r = self._unpack_params(ps)
        logdet = np.log(s0) + np.log(s1) + 0.5 * np.log(1.0 - r**2)

        xs = xs.reshape([xs.shape[0], self.n_data, 2])
        us = np.empty_like(xs)

        us[:, :, 0] = (xs[:, :, 0] - m0) / s0
        us[:, :, 1] = (xs[:, :, 1] - m1 - s1 * r * us[:, :, 0]) / (
            s1 * np.sqrt(1.0 - r**2)
        )
        us = us.reshape([us.shape[0], 2 * self.n_data])

        L = np.sum(scipy.stats.norm.logpdf(us), axis=1) - self.n_data * logdet[:, 0]
        L = L[0] if one_datapoint else L

        return L if log else np.exp(L)

    @staticmethod
    def _unpack_params(ps):
        """
        Unpack parameters ps into m0, m1, s0, s1, r.
        """

        assert ps.shape[1] == 5, "wrong size"

        m0 = ps[:, [0]]
        m1 = ps[:, [1]]
        s0 = ps[:, [2]] ** 2
        s1 = ps[:, [3]] ** 2
        r = np.tanh(ps[:, [4]])

        return m0, m1, s0, s1, r


def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    est_ps = [1, 1, -1.0, -0.9, 0.6]

    rng = np.random.RandomState()
    obs_xs = Stats().calc(Model().sim(est_ps, rng=rng))

    return est_ps, obs_xs


def _Prior(d=5):
    """
    Generate prior samples.
    """
    lower = [-3.0] * d
    upper = [+3.0] * d
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    one_sample = d is None
    u = np.random.rand(d)
    x = (upper - lower) * u + lower
    a = np.random.randn(1)
    x[1] = x[0] ** 2 + a[0] * 0.1
    return x[0] if one_sample else x

# ============================================================================
# 3. MCMC Components
# ============================================================================
def log_posterior(theta, obs_xs):
    """
    Calculate log posterior for parameter theta given observed data obs_xs.
    """

    if (
        abs(theta[0]) >= 3
        or abs(theta[2]) >= 3
        or abs(theta[3]) >= 3
        or abs(theta[4]) >= 3
    ):
        return -np.inf
    else:
        u = [theta[0], theta[1]]
        s1 = theta[2] ** 2
        s2 = theta[3] ** 2
        rho = math.tanh(theta[4])

        Sigma = [
            [s1**2, rho * s1 * s2],
            [rho * s1 * s2, s2**2],
        ]

        IS = np.linalg.inv(Sigma)
        quad_form = np.matrix.trace(
            np.matmul(np.matmul(IS, (obs_xs - u).T), (obs_xs - u))
        )
        log_likelihood = -0.5 * quad_form - 0.5 * n * np.log(np.linalg.det(Sigma))
        log_prior = -((theta[1] - theta[0] ** 2) ** 2) / (2 * 0.1**2)
        return log_likelihood + log_prior

def generate_initial_proposal_mcmc(N_proposal):
    """
    Generate initial proposal samples for MCMC.
    """
    prior_sampler = bf.simulation.Prior(prior_fun=_Prior)
    return prior_sampler(N_proposal)["prior_draws"]

def log_posterior_array(theta, obs_xs):
    """
    Calculate log posterior for an array of parameters.
    """
    log_posteriors = np.zeros(theta.shape[0])
    for i in range(theta.shape[0]):
        log_posteriors[i] = log_posterior(theta[i], obs_xs)
    return log_posteriors


def mcmc(obs_xs, N_proposal, burn_in_steps):
    """
    Run N_proposal MCMC chains simultaneously.
    """
    Theta_seq = []
    accp = 0
    h = 0.05 

    Theta_proposal = generate_initial_proposal_mcmc(N_proposal)
    log_posterior_0 = log_posterior_array(Theta_proposal, obs_xs)

    for mcmc_step in range(burn_in_steps + 1):
        Theta_new_proposal = np.random.normal(
            loc=Theta_proposal, scale=h, size=(N_proposal, 5)
        )
        log_posterior_1 = log_posterior_array(Theta_new_proposal, obs_xs)
        log_ratio = log_posterior_1 - log_posterior_0
        u = np.log(np.random.uniform(size=N_proposal))
        accept = u <= log_ratio

        Theta_proposal[accept] = Theta_new_proposal[accept]
        log_posterior_0[accept] = log_posterior_1[accept]
        accp += np.sum(accept)

        Theta_seq.append(Theta_proposal.copy())

    Theta_mcmc = tf.concat(Theta_seq[burn_in_steps: burn_in_steps + 1], axis=0)
    accp_rate = accp / (N_proposal * (burn_in_steps + 1))
    print(f"Acceptance rate: {accp_rate:.4f}")

    return Theta_mcmc, accp


def stereo_proj(A):
    """
    Spherical (stereographic) projection transform.
    """
    X_comp = A[..., 0]
    Y_comp = A[..., 1]
    new_X_comp = 2 * X_comp / (1 + X_comp**2 + Y_comp**2)
    new_Y_comp = 2 * Y_comp / (1 + X_comp**2 + Y_comp**2)
    Z_comp = (-1 + X_comp**2 + Y_comp**2) / (1 + X_comp**2 + Y_comp**2)
    result = np.stack([new_X_comp, new_Y_comp, Z_comp], axis=-1)
    return result

# ============================================================================
# 4. Data Generation Process (Main Iteration Loop)
# ============================================================================
# MCMC parameters
N_proposal = 5000
burn_in_steps = 7500

for it in range(10):
    print(f"--- Starting Iteration {it} ---")
    
    # 1. Generate NEW obs_xs for this iteration
    true_ps_val, obs_xs_raw = get_ground_truth()
    obs_xs_it = np.reshape(obs_xs_raw, (n, 2))  
    np.save(os.path.join(data_folder, f"obs_xs_{it}.npy"), obs_xs_it)
    print(f"Generated and saved obs_xs_{it}.npy")

    # 2. Immediately run MCMC for this obs_xs
    print(f"Running MCMC for iteration {it}...")
    Theta_mcmc_tensor, accp = mcmc(obs_xs_it, N_proposal, burn_in_steps)
    ps_it = Theta_mcmc_tensor.numpy()
    np.save(os.path.join(data_folder, f"ps_{it}.npy"), ps_it)
    print(f"Saved posterior samples to ps_{it}.npy")

    # 3. Calculate h_mmd based on CURRENT ps_it
    ps_quantile = ps_it.copy()
    ps_quantile[:, 3] = np.abs(ps_quantile[:, 3])
    ps_quantile[:, 2] = np.abs(ps_quantile[:, 2])
    Diff = pairwise_distances(ps_quantile, metric="euclidean")
    diff = Diff[np.triu_indices(ps_it.shape[0], 1)]
    h_mmd_it = np.median(diff)
    print(f"Iteration {it} h_mmd: {h_mmd_it}")
    # Save h_mmd for each iteration if needed
    np.save(os.path.join(data_folder, f"h_mmd_{it}.npy"), h_mmd_it)

    # 4. Generate training set for this iteration
    prior = bf.simulation.Prior(prior_fun=_Prior)
    model = Model() 
    Theta = prior(N)["prior_draws"]
    X = model.sim_preserved_shape(ps=Theta)
    X = stereo_proj(X)
    XS = X.reshape(-1, n * 3)
    x_train = np.concatenate((Theta, XS), axis=1)

    tf_Theta = tf.convert_to_tensor(Theta, dtype=tf.float32)
    Theta_diff = tf.expand_dims(tf_Theta, 1) - tf.expand_dims(tf_Theta, 0)
    weight_matrix = tf.exp(
        -tf.reduce_sum(Theta_diff**2, axis=-1) / (2 * h_mmd_it**2)
    )

    weight_matrix = tf.linalg.set_diag(weight_matrix, tf.zeros(N))
    weight_matrix = weight_matrix / tf.reduce_sum(weight_matrix, axis=-1, keepdims=True)
    
    Q = 1
    select_index = tf.random.categorical(tf.math.log(weight_matrix), Q)
    select_index = select_index.numpy()

    x_train_nn = np.concatenate((Theta, XS, select_index), axis=1)
    Theta = Theta.astype("float32")
    X = X.astype("float32")
    x_train = x_train.astype("float32")
    x_train_nn = x_train_nn.astype("float32")
    
    keys = [
        "prior_non_batchable_context",
        "prior_batchable_context",
        "prior_draws",
        "sim_non_batchable_context",
        "sim_batchable_context",
        "sim_data",
    ]
    x_train_bf = dict.fromkeys(keys)
    x_train_bf["prior_draws"] = Theta
    x_train_bf["sim_data"] = X

    np.save(os.path.join(data_folder, f"x_train_{it}.npy"), x_train)
    np.save(os.path.join(data_folder, f"x_train_nn_{it}.npy"), x_train_nn)
    np.save(os.path.join(data_folder, f"X_{it}.npy"), X)
    np.save(os.path.join(data_folder, f"Theta_{it}.npy"), Theta)
    
    file_path = os.path.join(data_folder, f"x_train_bf_{it}.pkl")
    with open(file_path, "wb") as pickle_file:
        pickle.dump(x_train_bf, pickle_file)
    
    print(f"Iteration {it} data generation complete.\n")