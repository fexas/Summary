"""
This file implements the Slicing MMD-based amortized inference for the Toy Example.
Key functionalities:
1. Define hyperparameters and file paths for the experiments.
2. Define the simulator model and data generation related classes/functions.
3. Run the amortized inference experiment using a summary network (T) and a generative network (G) with Sliced MMD loss.
4. Use the learned summary statistics for posterior refinement via ABC-MCMC.
5. Summarize and save experimental results.
"""
import os
import gc
import scipy
import numpy as np
import tensorflow as tf
import math
import csv
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import time

# ============================================================================
# 1. Hyperparameters and File Paths
# ============================================================================
# parameters for data
N = 12800  #  data_size
n = 50  # sample_size
d = 5  # dimension of parameter theta
d_x = 3  # dimenision of x
p = 10  # dimension of summary statistics
Q = 1  # number of draw from \exp{\frac{\Vert \theta_i - \theta_j \Vert^2}{w}} in first penalty

## NN's parameters
M = 50  # number of hat_theta_i to estimate MMD
L = 20  # number of unit vector in  S^{d-1} to draw
lambda_1 = 0  # coefficient for 1st penalty
batch_size = 256 # 128 # 256 
Np = 5000 # 5000  # number of estimate theta
default_lr = 0.001 # 0.001 # 0.001 # 0.0005
epochs = 500

# MCMC Parameters Setup
N_proposal = 1000  # 3000
n_samples = 100 # 100
burn_in =  100 #249 # 150 # 150
thin = 20
Ns = 5
proposed_std = 0.5
quantile_level = 0.005
epsilon_upper_bound = 1000 

# color setting
truth_color = "#FF6B6B"
est_color = "#4D96FF"
refined_color = "#6BCB77"
upper_labels=["\\theta_1","\\theta_2","\\theta_3","\\theta_4","\\theta_5"]

# quan1 = 2 * 0.1**2

# file path
current_dir = os.getcwd()

fig_folder = "nn_fig"
os.makedirs(fig_folder, exist_ok=True)

gif_folder = "nn_gif"
os.makedirs(gif_folder, exist_ok=True)

ps_folder = "nn_ps"
os.makedirs(ps_folder, exist_ok=True)

mcmc_samples_folder = "nn_mcmc_samples"
os.makedirs(mcmc_samples_folder, exist_ok=True)

likelihood_bandwidth_path = os.path.join(current_dir, "likelihood_bandwidth_nn.txt")
quan1_record_csv = "quan1_record.csv"


# ============================================================================
# 2. Data Generation Components
# ============================================================================

# class for toy example


class Uniform:
    """
    Parent class for uniform distributions.
    """

    def __init__(self, n_dims):

        self.n_dims = n_dims

    def grad_log_p(self, x):
        """
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
    Implements a uniform pdf, constrained in a box.
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
        :param x: evaluate at rows
        :param ii: a list of indices to evaluate marginal, if None then evaluates joint
        :param log: whether to return the log prob
        :return: the prob at x rows
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
        :param n_samples: int, number of samples to generate
        :return: numpy array, rows are samples. Only 1 sample (vector) if None
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

        raise NotImplementedError("simulator model must be implemented as a subclass")


class Stats:
    """
    Identity summary stats.
    """

    def __init__(self):
        pass

    @staticmethod
    def calc(ps):
        return ps

    # prepare_cond_input


def prepare_cond_input(xy, dtype):
    """
     prepares the conditional input for model evaluation.
    :param xy: tuple (x, y) for evaluating p(y|x)
    :param dtype: data type
    :return: prepared x, y and flag whether single datapoint input
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
    Simulator model.
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

        us = rng.randn(n_sims, self.n_data, 2)  
        xs = np.empty_like(us)

        xs[:, :, 0] = s0 * us[:, :, 0] + m0
        xs[:, :, 1] = s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r**2) * us[:, :, 1]) + m1

        self.n_sims += n_sims 

        return xs.reshape([n_sims, 2 * self.n_data])

    def sim_preserved_shape(self, ps, rng=np.random):
        """
        Simulate data at parameters ps.
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
        Unpack parameters ps to m0, m1, s0, s1, r.
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


# real posterior
rng = np.random
true_ps, _ = get_ground_truth()

# class for data  generation and data trasformation (training datasets)


def stereo_proj(A):
    X_comp = A[..., 0]
    Y_comp = A[..., 1]

    new_X_comp = 2 * X_comp / (1 + X_comp**2 + Y_comp**2)
    new_Y_comp = 2 * Y_comp / (1 + X_comp**2 + Y_comp**2)
    Z_comp = (-1 + X_comp**2 + Y_comp**2) / (1 + X_comp**2 + Y_comp**2)

    result = np.stack([new_X_comp, new_Y_comp, Z_comp], axis=-1)

    return result


def _Prior(d=5):
    """
    :param n_samples: int, number of samples to generate
    :return: numpy array, rows are samples. Only 1 sample (vector) if None
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


# -----------------------------
# NN Module
# -----------------------------


class ConfigurationError(Exception):
    """Class for error in model configuration, e.g. in meta dict"""

    pass


class InvariantModule(tf.keras.Model):
    """Implements an invariant module performing a permutation-invariant transform.

    For details and rationale, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(self, settings, **kwargs):
        """Creates an invariant module according to [1] which represents a learnable permutation-invariant
        function with an option for learnable pooling.

        Parameters
        ----------
        settings : dict
            A dictionary holding the configuration settings for the module.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the `tf.keras.Model` constructor.
        """

        super().__init__(**kwargs)

        # Create internal functions
        self.s1 = Sequential(
            [
                Dense(**settings["dense_s1_args"])
                for _ in range(settings["num_dense_s1"])
            ]
        )
        self.s2 = Sequential(
            [
                Dense(**settings["dense_s2_args"])
                for _ in range(settings["num_dense_s2"])
            ]
        )

        # Pick pooling function
        if settings["pooling_fun"] == "mean":
            pooling_fun = partial(tf.reduce_mean, axis=-2)
        elif settings["pooling_fun"] == "max":
            pooling_fun = partial(tf.reduce_max, axis=-2)
        else:
            if callable(settings["pooling_fun"]):
                pooling_fun = settings["pooling_fun"]
            else:
                raise ConfigurationError("pooling_fun argument not understood!")
        self.pooler = pooling_fun

    def call(self, x, **kwargs):
        """Performs the forward pass of a learnable invariant transform.

        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size,..., x_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size,..., out_dim)
        """

        x_reduced = self.pooler(self.s1(x, **kwargs))
        out = self.s2(x_reduced, **kwargs)
        return out


class EquivariantModule(tf.keras.Model):
    """Implements an equivariant module performing an equivariant transform.

    For details and justification, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(self, settings, **kwargs):
        """Creates an equivariant module according to [1] which combines equivariant transforms
        with nested invariant transforms, thereby enabling interactions between set members.

        Parameters
        ----------
        settings : dict
            A dictionary holding the configuration settings for the module.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the ``tf.keras.Model`` constructor.
        """

        super().__init__(**kwargs)

        self.invariant_module = InvariantModule(settings)
        self.s3 = Sequential(
            [
                Dense(**settings["dense_s3_args"])
                for _ in range(settings["num_dense_s3"])
            ]
        )

    def call(self, x, **kwargs):
        """Performs the forward pass of a learnable equivariant transform.

        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, ..., x_dim)

        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, ..., equiv_dim)
        """

        # Store shape of x, will be (batch_size, ..., some_dim)
        shape = tf.shape(x)

        # Example: Output dim is (batch_size, inv_dim) - > (batch_size, N, inv_dim)
        out_inv = self.invariant_module(x, **kwargs)
        out_inv = tf.expand_dims(out_inv, -2)
        tiler = [1] * len(shape)
        tiler[-2] = shape[-2]
        out_inv_rep = tf.tile(out_inv, tiler)

        # Concatenate each x with the repeated invariant embedding
        out_c = tf.concat([x, out_inv_rep], axis=-1)

        # Pass through equivariant func
        out = self.s3(out_c, **kwargs)
        return out


# class for NN
class NN(keras.Model):
    def __init__(self, G, T, **kwargs):
        super(NN, self).__init__(**kwargs)
        self.G = G
        self.T = T
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")  

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def MMD(self, theta, Gtheta):
        """
        Computes the Maximum Mean Discrepancy (MMD) between two sets of samples theta and Gtheta.
        Args:
            theta: [batch_size, d]
            Gtheta: [batch_size, M, d]
        """

        bandwidth = tf.constant([1 / n, 4 / n, 9 / n, 16 / n, 25 / n], "float32")
        bandwidth = tf.reshape(bandwidth, (bandwidth.shape[0], 1, 1, 1))
        coefficient = bandwidth ** (d / 2)
        coefficient = 1 / coefficient

        theta_ = tf.expand_dims(theta, 1)  # (N,1,d)

        gg = tf.einsum("ijk,ikl->ijl", Gtheta, tf.transpose(Gtheta, perm=[0, 2, 1]))
        gt = tf.einsum("ijk,ikl->ijl", Gtheta, tf.transpose(theta_, perm=[0, 2, 1]))

        rg = tf.reduce_sum(tf.square(Gtheta), axis=2, keepdims=True)  # (N,M,1)
        rt = tf.reduce_sum(tf.square(theta_), axis=2, keepdims=True)  # (N,1,1)

        SE_gg = rg - 2 * gg + tf.transpose(rg, perm=[0, 2, 1])  # (N,M,M)
        SE_gt = rg - 2 * gt + tf.transpose(rt, perm=[0, 2, 1])  # (N,M,1)

        K_gg = tf.exp(-0.5 * tf.expand_dims(SE_gg, axis=0) / bandwidth) * coefficient
        K_gt = tf.exp(-0.5 * tf.expand_dims(SE_gt, axis=0) / bandwidth) * coefficient

        mmd = tf.reduce_mean(K_gg) * M / (M-1) - 2 * tf.reduce_mean(K_gt)

        return mmd

    def SliceMMD(self, theta, Gtheta):
        """
        Args:
        theta: [batch_size,d]
        Gtheta: [batch_size,M,d]

        """

        bandwidth = tf.constant([1 / (2 * n)], "float32")
        constant = tf.sqrt(1 / bandwidth)

        unit_vectors = tf.random.normal(shape=(L, d))
        unit_vectors_norm = tf.norm(unit_vectors, axis=1, keepdims=True)
        unit_vectors = unit_vectors / unit_vectors_norm  # (L,d)
        unit_vectors = tf.transpose(unit_vectors, perm=[1, 0])  # (d,L)

        Gtheta_diff = tf.expand_dims(Gtheta, 1) - tf.expand_dims(
            Gtheta, 2
        )  # (N, M, M, d)
        slice_Gtheta_diff = tf.matmul(Gtheta_diff, unit_vectors)  # (N,M,M,L)

        marginal_p = constant[:, None, None, None, None] * tf.exp(
            -0.5 * tf.square(slice_Gtheta_diff) / bandwidth[:, None, None, None, None]
        )

        loss_term1 = tf.reduce_mean(marginal_p)

        # second_summation
        theta_minus_Gtheta = tf.expand_dims(theta, 1) - Gtheta  # （N,M,d）
        slice_theta_minus_Gtheta = tf.matmul(
            theta_minus_Gtheta, unit_vectors
        )  # (N,M,L)

        marginal_p = constant[:, None, None, None, None] * tf.exp(
            -0.5
            * tf.square(slice_theta_minus_Gtheta)
            / bandwidth[:, None, None, None, None]
        )
        loss_term2 = tf.reduce_mean(marginal_p)

        slice_MMD_loss = loss_term1 * M /(M-1) - 2 * loss_term2

        return slice_MMD_loss

    def train_step(self, data):

        data1 = tf.reshape(data, (batch_size, d + n * d_x + 1))

        theta_ = data1[:, 0:d]
        seg_X = data1[:, d : n * d_x + d]
        seg_X = tf.reshape(seg_X, (batch_size, n, d_x))

        Z = tf.random.normal(shape=[batch_size, M, d])

        with tf.GradientTape() as tape:
            TX_ = self.T(seg_X)
            TX_ = tf.expand_dims(TX_, axis=1)
            TX_ = tf.tile(TX_, [1, M, 1])
            Z_and_TX_ = tf.concat((Z, TX_), axis=-1)
            Z_and_TX_ = tf.reshape(Z_and_TX_, (batch_size * M, d + p))
            Gtheta = self.G(Z_and_TX_)
            Gtheta = tf.reshape(Gtheta, (batch_size, M, d))

            loss = self.SliceMMD(theta_, Gtheta)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)

        return {"loss": self.total_loss_tracker.result()}


def compute_kernel(x, y, h_mmd):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]

    tiled_x = tf.tile(
        tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1])
    )
    tiled_y = tf.tile(
        tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1])
    )
    return tf.exp(-tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2) / (2 * h_mmd))


def compute_mmd(x, y, h_mmd):
    x_kernel = compute_kernel(x, x, h_mmd)
    y_kernel = compute_kernel(y, y, h_mmd)
    xy_kernel = compute_kernel(x, y, h_mmd)
    return (
        tf.reduce_mean(x_kernel)
        + tf.reduce_mean(y_kernel)
        - 2 * tf.reduce_mean(xy_kernel)
    )


# ============================================================================
# 3. Main Experiment Execution
# ============================================================================
def run_experiment(it):
    file_path = os.path.join(current_dir, "data", f"obs_xs_{it}.npy")
    obs_xs = np.load(file_path)

    file_path = os.path.join(current_dir, "data", f"ps_{it}.npy")
    ps = np.load(file_path)
    
    file_path = os.path.join(current_dir, "data", f"h_mmd_{it}.npy")
    h_mmd = np.load(file_path)  # bandwidth of MMD
    h_mmd = h_mmd**2


    file_path = os.path.join(current_dir, "data", f"X_{it}.npy")
    X = np.load(file_path)
    
    file_path = os.path.join(current_dir, "data", f"x_train_nn_{it}.npy")
    x_train = np.load(file_path)
    print("x_train info:", x_train.shape)

    # ---------------------------------------------------------
    # 3.1 Summary Network Definition (DeepSets)
    # ---------------------------------------------------------
    ## summary_net -- deep set
    settings = dict(
        num_dense_s1=2,
        num_dense_s2=2,
        num_dense_s3=2,
        dense_s1_args={
            "units": 64,
            "activation": "relu",
            "kernel_initializer": "glorot_uniform",
        },
        dense_s2_args={
            "units": 64,
            "activation": "relu",
            "kernel_initializer": "glorot_uniform",
        },
        dense_s3_args={
            "units": 64,
            "activation": "relu",
            "kernel_initializer": "glorot_uniform",
        },
        pooling_fun="mean",
    )

    num_equiv = 2
    equiv_layers = Sequential([EquivariantModule(settings) for _ in range(num_equiv)])
    inv = InvariantModule(settings)
    out_layer = layers.Dense(p, activation="linear")
    T_inputs = keras.Input(shape=([n, d_x]))
    x = equiv_layers(T_inputs)
    T_outputs = out_layer(inv(x))
    T = keras.Model(T_inputs, T_outputs, name="T")
    T.summary()

    # ---------------------------------------------------------
    # 3.2 Generative Network Definition (Generator G)
    # ---------------------------------------------------------
    ## generative network

    intermediate_dim_G = 256
    G_inputs = keras.Input(shape=([p + d]))  # Input: z_i & T(x_{1:n}^i)

    x = layers.Dense(
        units=intermediate_dim_G,
        activation="relu",
        kernel_regularizer=regularizers.l1(0.01),
        kernel_initializer="he_normal",
    )(G_inputs)
    x = layers.Dense(
        units=128,
        activation="relu",
        kernel_regularizer=regularizers.l1(0.01),
        kernel_initializer="he_normal",
    )(x)
    x = layers.Dense(
        units=64,
        activation="relu",
        kernel_regularizer=regularizers.l1(0.01),
        kernel_initializer="he_normal",
    )(x)
    x = layers.Dense(
        units=32,
        activation="relu",
        kernel_regularizer=regularizers.l1(0.01),
        kernel_initializer="he_normal",
    )(x)
    G_outputs = layers.Dense(units=d)(x)

    G = keras.Model(G_inputs, G_outputs, name="G")
    G.summary()

    nn = NN(G=G, T=T)

    # ---------------------------------------------------------
    # 3.3 Training Configuration and Execution
    # ---------------------------------------------------------
    # -----------------------------
    # Training Utilities
    # -----------------------------
    class LossHistory(Callback):
        """Callback to track training loss history."""

        def __init__(self):
            super().__init__()
            self.epoch_losses = []

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_losses.append(logs["loss"])

    loss_history = LossHistory()

    schedule = tf.keras.optimizers.schedules.CosineDecay(
        default_lr, epochs * batch_size, name="lr_decay"
    )

    OPTIMIZER_DEFAULTS = {"global_clipnorm": 1.0}
    nn_optimizer = tf.keras.optimizers.Adam(schedule, **OPTIMIZER_DEFAULTS)

    nn.compile(optimizer=nn_optimizer, run_eagerly=False)

    train_start_time = time.time()
    nn.fit(x_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[loss_history])
    train_end_time = time.time()

    del x_train
    gc.collect()

    # plot and save losses
    plt.figure()
    plt.plot(loss_history.epoch_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    loss_plot_path = os.path.join(fig_folder, f"training_loss_lr_{default_lr}_{it}.png")
    plt.savefig(loss_plot_path)
    plt.close()

    elapsed_time = train_end_time - train_start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    x_target = obs_xs.reshape(1, n, 2)
    x_target = stereo_proj(x_target)
    xx = tf.convert_to_tensor(x_target)
    xx = tf.tile(xx, [Np, 1, 1])
    x = nn.T(xx)
    Y = tf.random.normal(shape=[Np, d])
    YY = tf.concat([Y, x], 1)
    Y0 = nn.G(YY)

    mmd_nn = compute_mmd(
        tf.cast(Y0, "float32"),
        tf.convert_to_tensor(ps, dtype="float32"),
        h_mmd,
    )
    proposed_cov_root = tf.math.reduce_std(Y0, axis=0)
    
    # save the results
    nn_ps_path = os.path.join(ps_folder, f"nn_{n}_ps_{it}.npy")
    np.save(nn_ps_path, Y0.numpy())

    # plot the results

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 5, figsize=(25, 6))

    true_ps = [1, 1, -1.0, -0.9, 0.6]

    x_limits = [
        [0.7, 1.3],  # theta_0
        [0.6, 1.4],  # theta_1
        [-1.5, 1.5],  # theta_2
        [-1.5, 1.5],  # theta_3
        [0, 1.2],  # theta_4
    ]

    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for upper_label, j in zip(upper_labels, range(d)):
        sns.kdeplot(
            ps[:, j],
            ax=axs[j],
            fill=False,
            label="posterior",
            color=truth_color,
            linestyle="-.",
            linewidth=2.0,
        )
        sns.kdeplot(
            Y0[:, j],
            ax=axs[j],
            fill=False,
            label="SMMD",
            color=est_color,
            linestyle="-",
            linewidth=2.0,
        )
        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")

    # save figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.tight_layout(pad=3.0)
    graph_path = os.path.join(fig_folder, f"nn_{n}_experiment_{it}.png")
    plt.savefig(graph_path)
    plt.close()

    marginal_mmd_list = []
    for j in range(d):
        mmd_nn_marginal = compute_mmd(
            tf.expand_dims(tf.cast(Y0[:, j], "float32"), axis=1), 
            tf.expand_dims(tf.convert_to_tensor(ps[:, j].astype("float32")), axis=1),
            1 / n,
        )
        marginal_mmd_list.append(mmd_nn_marginal.numpy())

    # -----------------------------
    # MCMC Refinement Overview
    # -----------------------------
    # Refinement using Monte Carlo ABC with weight being calculated as a kernel regression estimator or direct sample estimation
    # This section implements MCMC to refine the parameter estimation results.
    
    TX_target_ = nn.T(tf.convert_to_tensor(x_target, dtype=tf.float32))

    # -----------------------------
    # Calculate Bandwidth for Likelihood Estimator
    # -----------------------------
    N0 = 5000
    xx = tf.convert_to_tensor(x_target)
    xx = tf.tile(xx, [N0, 1, 1])
    x = nn.T(xx)
    Y = tf.random.normal(shape=[N0, d])
    YY = tf.concat([Y, x], 1)
    Theta0 = nn.G(YY)
    Theta0 = tf.cast(Theta0, "float32")

    xn_0 = Stats().calc(Model().sim(Theta0))
    xn_0 = xn_0.reshape(N0, n, 2)
    xn_0 = stereo_proj(xn_0)
    xn_0 = tf.convert_to_tensor(xn_0, dtype=tf.float32)

    TT = nn.T(xn_0)
    Diff = tf.reduce_sum((nn.T(xx) - TT) ** 2, axis=1)
    Diff = tf.sqrt(Diff)
    Diff = tf.cast(Diff, "float32")
    quan1 = np.quantile(Diff.numpy(), quantile_level)
    # record the quan1 value in a CSV file
    with open(quan1_record_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([quan1])
    quan1 = min(quan1, epsilon_upper_bound)

    # -----------------------------

    # -----------------------------
    # Create Folders for Saving Figures
    # -----------------------------
    # Create a new folder under nn_gif_folder named 'nn_gif_{it}'
    temp_gif_folder = os.path.join(gif_folder, f"nn_gif_{it}")
    os.makedirs(temp_gif_folder, exist_ok=True)
    # create d subfolders under temp_gif_folder named 'theta_1', 'theta_2', ..., 'theta_d'
    for ith_element in range(d):
        theta_i_gif_folder = os.path.join(temp_gif_folder, f"theta_{ith_element+1}")
        os.makedirs(theta_i_gif_folder, exist_ok=True)


    # -----------------------------
    # Function to Generate Initial MCMC Proposals
    # -----------------------------
    def generate_initial_proposal_mcmc(N_proposal):
        xx_proposal = tf.convert_to_tensor(x_target)
        xx_proposal = tf.tile(xx_proposal, [N_proposal, 1, 1])
        x_proposal = nn.T(xx_proposal)
        Y = tf.random.normal(shape=[N_proposal, d])
        YY = tf.concat([Y, x_proposal], 1)
        Theta_proposal = nn.G(YY)
        return Theta_proposal

    def prior(theta):
        """Calculate the prior density of the parameters"""

        selected_columns = tf.gather(theta, indices=[0, 2, 3, 4], axis=1)  

        mask = tf.reduce_prod(tf.where(tf.abs(selected_columns) < 3, 1.0, 0.0), axis=1)

        gaussian_part = tf.exp(-((theta[:, 1] - theta[:, 0] ** 2) ** 2) / (2 * 0.1**2))

        prior_ = tf.cast(mask, "float32") * tf.cast(gaussian_part, "float32")
        return prior_

    def simulate_summary_data(theta, nsims):
        """Generate simulated data from the model and calculate summary statistics"""
        sim_X = np.zeros(shape=(theta.shape[0], nsims, n, 3))

        theta_expand = tf.tile(tf.expand_dims(theta, axis=1), [1, nsims, 1])

        for i_sim in range(theta.shape[0]):
            sim_x_ = Stats().calc(Model().sim_preserved_shape(theta_expand[i_sim, :]))
            sim_x_ = stereo_proj(sim_x_)
            sim_X[i_sim] = sim_x_

        TX_ = np.zeros(shape=(theta.shape[0], nsims, p))
        for j_sim in range(nsims):
            TX_[:, j_sim, :] = nn.T(sim_X[:, j_sim, :, :]).numpy()

        TX_ = tf.convert_to_tensor(TX_, dtype=tf.float32)
        return TX_

    def distance(TX_sim, TX_target):
        """Calculate the distance between simulated data and target data"""
        return tf.reduce_sum((TX_sim - TX_target) ** 2, axis=-1)

    def approximate_likelihood(theta, nsims, epsilon):
        """Approximate the likelihood of the parameters"""
        TX_sim = simulate_summary_data(theta, nsims)
        dist = distance(TX_sim, TX_target_)
        kde_ = tf.reduce_mean(tf.exp(-dist / (2 * epsilon**2)), axis=-1)
        return kde_

    def mcmc_refinement(
        N_proposal=1,
        n_samples=5000,
        burn_in=2000,
        thin=10,
        nsims=50,
        epsilon=0.1,
        proposed_std=0.5,
    ):
        """Run ABC-MCMC sampling.

        Args:
            N_proposal (int, optional): Number of MCMC chains to run simultaneously. Defaults to 1.
            n_samples (int, optional): Total number of samples to generate. Defaults to 5000.
            burn_in (int, optional): Length of the burn-in period. Defaults to 2000.
            thin (int, optional): Sampling interval to reduce autocorrelation. Defaults to 10.
            nsims (int, optional): Number of simulations for likelihood approximation. Defaults to 50.
            epsilon (float, optional): Bandwidth parameter for likelihood approximation. Defaults to 0.1.
        """

        samples = []
        accepted = 0
        mcmc_samples = []

        # Initialize MCMC proposals
        current_theta = generate_initial_proposal_mcmc(N_proposal)

        dim_clip = [0, 2, 3, 4]
        all_dims = tf.range(current_theta.shape[1])  
        mask = tf.reduce_any(
            tf.equal(all_dims[:, None], dim_clip), axis=1
        )  
        current_theta = tf.where(
            mask,  
            tf.clip_by_value(current_theta, -3, 3),  
            current_theta,  
        )
        current_prior = prior(current_theta)
        current_likelihood = approximate_likelihood(current_theta, nsims, epsilon)
        current_ratio = current_prior * current_likelihood

        for i_mcmc in range(n_samples + burn_in):
            jump = tf.random.normal(
                shape=current_theta.shape, mean=0, stddev=proposed_cov_root
            ) * proposed_std

            
            proposed_theta = current_theta + jump
            proposed_prior = prior(proposed_theta)
            proposed_likelihood = approximate_likelihood(proposed_theta, nsims, epsilon)
            proposed_ratio = proposed_prior * proposed_likelihood

            acceptance_prob = tf.minimum(1.0, proposed_ratio / current_ratio)
            u = tf.random.uniform(shape=(N_proposal,), minval=0.0, maxval=1.0)

            accept_mask = u < acceptance_prob
            accept_mask_2d = tf.expand_dims(accept_mask, axis=1)
            # if i_mcmc >= burn_in:
            #     accepted += tf.reduce_sum(tf.cast(accept_mask, tf.float32))
            # Alternative
            accepted += tf.reduce_sum(tf.cast(accept_mask, tf.float32))

            current_theta = tf.where(accept_mask_2d, proposed_theta, current_theta)
            current_ratio = tf.where(accept_mask, proposed_ratio, current_ratio)

            if i_mcmc >= burn_in and (i_mcmc - burn_in) % thin == 0:
                samples.append(current_theta)

            # append samples to mcmc_current_samples
            mcmc_samples.append(current_theta.numpy())

        # acceptance_rate = accepted / (n_samples * N_proposal)
        # Alternative
        acceptance_rate = accepted / ( (n_samples + burn_in) * N_proposal)
        samples = tf.concat(samples, axis=0)

        mcmc_samples = np.array(mcmc_samples)

        return samples, acceptance_rate, mcmc_samples

    # -----------------------------
    # Run MCMC Refinement Multiple Times
    # -----------------------------

    refined_time_start = time.time()
    Theta_mcmc, accp_rate, mcmc_samples = mcmc_refinement(
        N_proposal=N_proposal,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        nsims=Ns,
        epsilon=quan1,
        proposed_std=proposed_std,
    )
    refined_time_end = time.time()
    elapsed_refined_time_str = time.strftime(
        "%H:%M:%S", time.gmtime(refined_time_end - refined_time_start)
    )

    # save mcmc results
    mcmc_path = os.path.join(ps_folder, f"nn_{n}_mcmc_{it}.npy")
    np.save(mcmc_path, Theta_mcmc.numpy())
    total_mcmc_samples_path = os.path.join(mcmc_samples_folder, f"nn_{n}_mcmc_samples_{it}.npy")
    np.save(total_mcmc_samples_path, mcmc_samples)

    # -----------------------------
    # Calculate MMD for MCMC Results
    # -----------------------------

    mmd_refinement = compute_mmd(
        tf.cast(Theta_mcmc, "float32"),
        tf.convert_to_tensor(ps, dtype="float32"),
        h_mmd,
    )

    # mmd_refinement = compute_mmd(
    #     tf.cast(Theta_mcmc[0:10000, :], "float32"),
    #     tf.convert_to_tensor(ps[0:10000, :].astype("float32")),
    #     h_mmd,
    # )

    # -----------------------------
    # Plot Final Posterior Estimation
    # -----------------------------

    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 5, figsize=(25, 6))

    true_ps = [1, 1, -1.0, -0.9, 0.6]

    x_limits = [
        [0.7, 1.3],  # theta_0
        [0.6, 1.4],  # theta_1
        [-1.5, 1.5],  # theta_2
        [-1.5, 1.5],  # theta_3
        [0, 1.2],  # theta_4
    ]

    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

    for upper_label, j in zip(upper_labels, range(d)):
        sns.kdeplot(
            ps[:, j],
            ax=axs[j],
            fill=False,
            label="posterior",
            color=truth_color,
            linestyle="-.",
            linewidth=2.0,
        )
        sns.kdeplot(
            Y0[:, j],
            ax=axs[j],
            fill=False,
            label="SMMD",
            color=est_color,
            linestyle="-",
            linewidth=2.0,
        )
        sns.kdeplot(
            Theta_mcmc[:, j],
            ax=axs[j],
            fill=False,
            label="SMMD+ABC-MCMC",
            color=refined_color,
            linestyle="--",
            linewidth=2.0,
        )
        axs[j].set_title(f"${upper_label}$", pad=15)
        axs[j].set_ylabel("")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    plt.tight_layout(pad=3.0)
    graph_path = os.path.join(fig_folder, f"nn_{n}_refined_experiment_{it}.png")
    plt.savefig(graph_path)
    plt.close()

    refined_marginal_mmd_list = []

    for j in range(d):
        mmd_nn_marginal = compute_mmd(
            tf.expand_dims(
                tf.cast(Theta_mcmc[:, j], "float32"), axis=1
            ),
            tf.expand_dims(tf.convert_to_tensor(ps[:, j].astype("float32")), axis=1),
            1 / n,
        )
        refined_marginal_mmd_list.append(mmd_nn_marginal.numpy())

    return (
        mmd_nn,
        mmd_refinement,
        elapsed_time_str,
        elapsed_refined_time_str,
        marginal_mmd_list,
        refined_marginal_mmd_list,
        accp_rate.numpy(),
    )


output_file = f"nn_{n}_result1.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Experiment_Index",
            "train_time",
            "refined_time",
            "mmd_nn",
            "mmd_nn_r",
            "accp_rate",
        ]
    )

marginal_mmd_output_file = f"nn_{n}_marginal_mmd.csv"

with open(marginal_mmd_output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Experiment_Index"]
        + [f"m_mmd theta_{i+1}" for i in range(d)]
        + [f"rm_mmd theta_{i+1}" for i in range(d)]
    )

for i in range(10):
    print(f"Running experiment {i+1}")
    (
        mmd_nn,
        mmd_nn_r,
        train_time,
        elapsed_refined_time_str,
        marginal_mmd_list,
        refined_marginal_mmd_list,
        accp_rate,
    ) = run_experiment(i)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                i + 1,
                train_time,
                elapsed_refined_time_str,
                mmd_nn.numpy(),
                mmd_nn_r.numpy(),
                accp_rate,
            ]
        )

    with open(marginal_mmd_output_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([i + 1] + marginal_mmd_list + refined_marginal_mmd_list)


# ============================================================================
# Generate Summary Statistics
# ============================================================================

# Read the result file and calculate statistics
def time_str_to_seconds(time_str):
    """Convert HH:MM:SS format to seconds"""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def seconds_to_time_str(seconds):
    """Convert seconds to HH:MM:SS format"""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

# Read the main result file
train_times_sec = []
refined_times_sec = []
mmd_nn_values = []
mmd_nn_r_values = []

with open(output_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        train_times_sec.append(time_str_to_seconds(row['train_time']))
        refined_times_sec.append(time_str_to_seconds(row['refined_time']))
        mmd_nn_values.append(float(row['mmd_nn']))
        mmd_nn_r_values.append(float(row['mmd_nn_r']))

# Calculate statistics
train_time_mean_sec = np.mean(train_times_sec)
refined_time_mean_sec = np.mean(refined_times_sec)
train_time_mean_str = seconds_to_time_str(train_time_mean_sec)
refined_time_mean_str = seconds_to_time_str(refined_time_mean_sec)

mmd_nn_mean = np.mean(mmd_nn_values)
mmd_nn_median = np.median(mmd_nn_values)
mmd_nn_r_mean = np.mean(mmd_nn_r_values)
mmd_nn_r_median = np.median(mmd_nn_r_values)

# Write summary to CSV
summary_file = f"nn_{n}_summary.csv"
with open(summary_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Average Training Time", train_time_mean_str])
    writer.writerow(["Average Refinement Time", refined_time_mean_str])
    writer.writerow(["MMD_NN Mean", f"{mmd_nn_mean:.6f}"])
    writer.writerow(["MMD_NN Median", f"{mmd_nn_median:.6f}"])
    writer.writerow(["MMD_NN_Refined Mean", f"{mmd_nn_r_mean:.6f}"])
    writer.writerow(["MMD_NN_Refined Median", f"{mmd_nn_r_median:.6f}"])

print(f"\n✅ Summary statistics saved to {summary_file}")
print(f"   Average Training Time: {train_time_mean_str}")
print(f"   Average Refinement Time: {refined_time_mean_str}")
print(f"   MMD_NN Mean: {mmd_nn_mean:.6f}, Median: {mmd_nn_median:.6f}")
print(f"   MMD_NN_Refined Mean: {mmd_nn_r_mean:.6f}, Median: {mmd_nn_r_median:.6f}")