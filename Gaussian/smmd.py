"""
This file implements the Slicing MMD-based amortized inference for the Toy Example.
Refactored for simplicity and compatibility with data_generation.py.
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

# Import configurations and functions from data_generation
from data_generation import (
    PRIOR_CONFIGS, 
    PRIOR_TYPE, 
    CURRENT_PRIOR, 
    prior_generator, 
    simulator, 
    unpack_params
)

# ============================================================================
# 1. Hyperparameters and File Paths
# ============================================================================
# parameters for data
N = 12800  #  data_size
n = 50  # sample_size
d = 5  # dimension of parameter theta
d_x = 2  # dimension of x (updated to 2)
p = 10  # dimension of summary statistics
Q = 1  # number of draw from \exp{\frac{\Vert \theta_i - \theta_j \Vert^2}{w}} in first penalty

## NN's parameters
M = 50  # number of hat_theta_i to estimate MMD
L = 20  # number of unit vector in  S^{d-1} to draw
lambda_1 = 0  # coefficient for 1st penalty
batch_size = 256 
Np = 5000  # number of estimate theta
default_lr = 0.001
epochs = 500

# MCMC Parameters Setup
N_proposal = 1000 
n_samples = 100 
burn_in =  100
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

# file path
current_dir = os.getcwd()
data_folder = "data" # Ensure we look in data folder

# Create result folder
result_folder = "smmd_result"
os.makedirs(result_folder, exist_ok=True)

fig_folder = os.path.join(result_folder, "nn_fig")
os.makedirs(fig_folder, exist_ok=True)

gif_folder = os.path.join(result_folder, "nn_gif")
os.makedirs(gif_folder, exist_ok=True)

ps_folder = os.path.join(result_folder, "nn_ps")
os.makedirs(ps_folder, exist_ok=True)

mcmc_samples_folder = os.path.join(result_folder, "nn_mcmc_samples")
os.makedirs(mcmc_samples_folder, exist_ok=True)

quan1_record_csv = os.path.join(result_folder, "quan1_record.csv")


# ============================================================================
# 2. NN Module
# ============================================================================

class ConfigurationError(Exception):
    """Class for error in model configuration, e.g. in meta dict"""
    pass


class InvariantModule(tf.keras.Model):
    """Implements an invariant module performing a permutation-invariant transform."""

    def __init__(self, settings, **kwargs):
        super().__init__(**kwargs)

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
        x_reduced = self.pooler(self.s1(x, **kwargs))
        out = self.s2(x_reduced, **kwargs)
        return out


class EquivariantModule(tf.keras.Model):
    """Implements an equivariant module performing an equivariant transform."""

    def __init__(self, settings, **kwargs):
        super().__init__(**kwargs)

        self.invariant_module = InvariantModule(settings)
        self.s3 = Sequential(
            [
                Dense(**settings["dense_s3_args"])
                for _ in range(settings["num_dense_s3"])
            ]
        )

    def call(self, x, **kwargs):
        # shape = tf.shape(x)
        # out_inv = self.invariant_module(x, **kwargs)
        # out_inv = tf.expand_dims(out_inv, -2)
        # tiler = [1] * len(shape)
        # tiler[-2] = shape[-2]
        # out_inv_rep = tf.tile(out_inv, tiler)
        
        # Simplified for rank 3 (batch, n, d) or handling dynamic shape better
        n_points = tf.shape(x)[-2]
        out_inv = self.invariant_module(x, **kwargs)
        out_inv = tf.expand_dims(out_inv, -2) # (batch, 1, d_out)
        out_inv_rep = tf.tile(out_inv, [1, n_points, 1]) # (batch, n, d_out)
        
        out_c = tf.concat([x, out_inv_rep], axis=-1)
        out = self.s3(out_c, **kwargs)
        return out


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
        bandwidth = tf.constant([1 / (2 * n)], "float32")
        constant = tf.sqrt(1 / bandwidth)

        unit_vectors = tf.random.normal(shape=(L, d))
        unit_vectors_norm = tf.norm(unit_vectors, axis=1, keepdims=True)
        unit_vectors = unit_vectors / unit_vectors_norm  # (L,d)
        unit_vectors = tf.transpose(unit_vectors, perm=[1, 0])  # (d,L)

        Gtheta_diff = tf.expand_dims(Gtheta, 1) - tf.expand_dims(Gtheta, 2)
        slice_Gtheta_diff = tf.matmul(Gtheta_diff, unit_vectors)

        marginal_p = constant[:, None, None, None, None] * tf.exp(
            -0.5 * tf.square(slice_Gtheta_diff) / bandwidth[:, None, None, None, None]
        )
        loss_term1 = tf.reduce_mean(marginal_p)

        theta_minus_Gtheta = tf.expand_dims(theta, 1) - Gtheta
        slice_theta_minus_Gtheta = tf.matmul(theta_minus_Gtheta, unit_vectors)

        marginal_p = constant[:, None, None, None, None] * tf.exp(
            -0.5 * tf.square(slice_theta_minus_Gtheta) / bandwidth[:, None, None, None, None]
        )
        loss_term2 = tf.reduce_mean(marginal_p)

        slice_MMD_loss = loss_term1 * M /(M-1) - 2 * loss_term2
        return slice_MMD_loss

    def train_step(self, data):
        # Updated to remove index handling
        # data shape is (batch_size, d + n * d_x)
        data1 = tf.reshape(data, (batch_size, d + n * d_x))

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
    file_path = os.path.join(data_folder, f"obs_xs_{it}.npy")
    obs_xs = np.load(file_path)

    file_path = os.path.join(data_folder, f"ps_{it}.npy")
    ps = np.load(file_path)
    
    file_path = os.path.join(data_folder, f"h_mmd_{it}.npy")
    h_mmd = np.load(file_path)  # bandwidth of MMD
    h_mmd = h_mmd**2

    file_path = os.path.join(data_folder, f"X_{it}.npy")
    X = np.load(file_path)
    
    file_path = os.path.join(data_folder, f"x_train_nn_{it}.npy")
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
    # Use list instead of Sequential to avoid argument issues
    equiv_layers_list = [EquivariantModule(settings, name=f"equiv_{i}") for i in range(num_equiv)]
    inv = InvariantModule(settings)
    out_layer = layers.Dense(p, activation="linear")
    T_inputs = keras.Input(shape=([n, d_x]))
    
    x = T_inputs
    for layer in equiv_layers_list:
        x = layer(x)
        
    T_outputs = out_layer(inv(x))
    T = keras.Model(T_inputs, T_outputs, name="T")
    T.summary()

    # ---------------------------------------------------------
    # 3.2 Generative Network Definition (Generator G)
    # ---------------------------------------------------------
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
    class LossHistory(Callback):
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
    # Also save to data folder
    loss_plot_path_data = os.path.join(data_folder, f"training_loss_lr_{default_lr}_{it}.png")
    plt.savefig(loss_plot_path_data)
    plt.close()

    elapsed_time = train_end_time - train_start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    x_target = obs_xs.reshape(1, n, 2)
    # No stereo_proj
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

    # Dynamic x_limits based on PRIOR_CONFIGS
    limit = CURRENT_PRIOR['bounds_limit']
    # Use slightly wider limits for visualization
    plot_limit = limit * 1.1
    x_limits = [
        [-plot_limit, plot_limit],
        [-plot_limit, plot_limit],
        [-plot_limit, plot_limit],
        [-plot_limit, plot_limit],
        [-plot_limit, plot_limit],
    ]
    # Keep some specific limits if needed, but general approach is safer:
    # However, let's respect the typical range of parameters if they are small
    # But since prior can be large, we should stick to prior bounds.
    # For 'weak_informative' limit is 9.0, for 'vague' it's 15.0.
    
    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        # ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

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

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    plt.tight_layout(pad=3.0)
    graph_path = os.path.join(fig_folder, f"nn_{n}_experiment_{it}.png")
    plt.savefig(graph_path)
    # Also save to data folder
    graph_path_data = os.path.join(data_folder, f"nn_{n}_experiment_{it}.png")
    plt.savefig(graph_path_data)
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

    # Replaced Model().sim and Stats().calc with simulator
    # simulator returns (N0, n, 2)
    xn_0 = simulator(Theta0.numpy(), n_samples=n)
    # No stereo_proj
    xn_0 = tf.convert_to_tensor(xn_0, dtype=tf.float32)

    TT = nn.T(xn_0)
    Diff = tf.reduce_sum((nn.T(xx) - TT) ** 2, axis=1)
    Diff = tf.sqrt(Diff)
    Diff = tf.cast(Diff, "float32")
    quan1 = np.quantile(Diff.numpy(), quantile_level)
    with open(quan1_record_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([quan1])
    quan1 = min(quan1, epsilon_upper_bound)

    # -----------------------------
    # MCMC Helper Functions
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
        limit = CURRENT_PRIOR['bounds_limit']
        noise_std = CURRENT_PRIOR['cond_noise_std']
        
        # Check bounds for all dimensions
        # theta shape (batch, d)
        is_in_bounds = tf.reduce_all(tf.abs(theta) <= limit, axis=1)
        
        # Gaussian dependency for theta[1] ~ N(theta[0]^2, noise_std)
        gaussian_part = tf.exp(-((theta[:, 1] - theta[:, 0] ** 2) ** 2) / (2 * noise_std**2))
        
        prior_ = tf.cast(is_in_bounds, "float32") * tf.cast(gaussian_part, "float32")
        return prior_

    def simulate_summary_data(theta, nsims):
        """Generate simulated data from the model and calculate summary statistics"""
        # theta: (batch, d)
        # return: (batch, nsims, p)
        
        batch_size = theta.shape[0]
        
        # Expand theta to (batch, nsims, d)
        theta_expand = tf.tile(tf.expand_dims(theta, axis=1), [1, nsims, 1])
        
        # Flatten to (batch * nsims, d)
        theta_flat = tf.reshape(theta_expand, (-1, d))
        
        # Simulate data: (batch * nsims, n, 2)
        sim_X_flat = simulator(theta_flat.numpy(), n_samples=n)
        
        # Convert to tensor
        sim_X_flat = tf.convert_to_tensor(sim_X_flat, dtype=tf.float32)
        
        # Calculate summary stats using Network T
        # T expects (batch, n, 2)
        TX_flat = nn.T(sim_X_flat) # (batch * nsims, p)
        
        # Reshape back to (batch, nsims, p)
        TX_ = tf.reshape(TX_flat, (batch_size, nsims, p))
        
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
        samples = []
        accepted = 0
        mcmc_samples = []

        # Initialize MCMC proposals
        current_theta = generate_initial_proposal_mcmc(N_proposal)
        
        # Clip to bounds
        limit = CURRENT_PRIOR['bounds_limit']
        current_theta = tf.clip_by_value(current_theta, -limit, limit)
        
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

            # Avoid division by zero
            acceptance_prob = tf.math.divide_no_nan(proposed_ratio, current_ratio)
            acceptance_prob = tf.minimum(1.0, acceptance_prob)
            
            u = tf.random.uniform(shape=(N_proposal,), minval=0.0, maxval=1.0)

            accept_mask = u < acceptance_prob
            accept_mask_2d = tf.expand_dims(accept_mask, axis=1)

            accepted += tf.reduce_sum(tf.cast(accept_mask, tf.float32))

            current_theta = tf.where(accept_mask_2d, proposed_theta, current_theta)
            current_ratio = tf.where(accept_mask, proposed_ratio, current_ratio)

            if i_mcmc >= burn_in and (i_mcmc - burn_in) % thin == 0:
                samples.append(current_theta)

            mcmc_samples.append(current_theta.numpy())

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

    # -----------------------------
    # Plot Final Posterior Estimation
    # -----------------------------
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 5, figsize=(25, 6))

    for j, ax in enumerate(axs):
        ax.set_xlim(x_limits[j])
        # ax.set_xticks(np.linspace(x_limits[j][0], x_limits[j][1], 5))

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


output_file = os.path.join(result_folder, f"nn_{n}_result1.csv")
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

marginal_mmd_output_file = os.path.join(result_folder, f"nn_{n}_marginal_mmd.csv")

with open(marginal_mmd_output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Experiment_Index"]
        + [f"m_mmd theta_{i+1}" for i in range(d)]
        + [f"rm_mmd theta_{i+1}" for i in range(d)]
    )

# Number of iterations to run (can be adjusted)
n_iterations = 1

for i in range(n_iterations):
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
summary_file = os.path.join(result_folder, f"nn_{n}_summary.csv")
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
