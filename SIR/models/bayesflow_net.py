import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np
import bayesflow as bf
import keras
from bayesflow.adapters.transforms import Transform

keras.config.set_floatx("float32")


class ToNumpy(Transform):
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
        return data


class InvariantModule(keras.layers.Layer):
    def __init__(self, settings, **kwargs):
        super().__init__(**kwargs)
        self.s1_layers = []
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.2)
        for _ in range(settings["num_dense_s1"]):
            self.s1_layers.append(
                keras.layers.Dense(
                    settings["dense_s1_args"]["units"],
                    activation="relu",
                    kernel_initializer=init,
                )
            )

        self.s2_layers = []
        for _ in range(settings["num_dense_s2"]):
            self.s2_layers.append(
                keras.layers.Dense(
                    settings["dense_s2_args"]["units"],
                    activation="relu",
                    kernel_initializer=init,
                )
            )

    def call(self, x):
        out = x
        for layer in self.s1_layers:
            out = layer(out)
        out_reduced = keras.ops.mean(out, axis=1)
        for layer in self.s2_layers:
            out_reduced = layer(out_reduced)
        return out_reduced

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.s2_layers[-1].units)


class EquivariantModule(keras.layers.Layer):
    def __init__(self, settings, invariant_module, **kwargs):
        super().__init__(**kwargs)
        self.invariant_module = invariant_module
        self.s3_layers = []
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.2)
        for _ in range(settings["num_dense_s3"]):
            self.s3_layers.append(
                keras.layers.Dense(
                    settings["dense_s3_args"]["units"],
                    activation="relu",
                    kernel_initializer=init,
                )
            )

    def call(self, x):
        n_points = keras.ops.shape(x)[1]
        out_inv = self.invariant_module(x)
        out_inv_rep = keras.ops.repeat(
            keras.ops.expand_dims(out_inv, axis=1), repeats=n_points, axis=1
        )
        out_c = keras.ops.concatenate([x, out_inv_rep], axis=-1)
        out = out_c
        for layer in self.s3_layers:
            out = layer(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.s3_layers[-1].units)


class RMSNorm(keras.layers.Layer):
    def __init__(self, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.weight = None

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.weight = self.add_weight(
            name="weight",
            shape=(dim,),
            initializer="ones",
            trainable=True,
        )

    def call(self, x):
        mean_sq = keras.ops.mean(keras.ops.square(x), axis=-1, keepdims=True)
        x = x * keras.ops.rsqrt(mean_sq + self.eps)
        return x * self.weight


class TimeSeriesSummary(keras.Model):
    def __init__(self, input_dim, output_dim=10, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.2)
        self.conv1 = keras.layers.Conv1D(
            filters=hidden_dim,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=init,
        )
        self.pool1 = keras.layers.MaxPooling1D(pool_size=2)
        self.conv2 = keras.layers.Conv1D(
            filters=hidden_dim * 2,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer=init,
        )
        self.global_pool = keras.layers.GlobalAveragePooling1D()
        self.norm = RMSNorm()
        self.dense = keras.layers.Dense(output_dim, kernel_initializer=init)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.global_pool(x)
        x = self.norm(x)
        return self.dense(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dense.units)

    def compute_metrics(self, x, stage=None):
        out = self(x)
        return {"outputs": out}


class DeepSetsSummary(keras.Model):
    def __init__(self, input_dim, output_dim=10, **kwargs):
        super().__init__(**kwargs)
        self._init = keras.initializers.RandomNormal(mean=0.0, stddev=0.2)
        settings = dict(
            num_dense_s1=2,
            num_dense_s2=2,
            num_dense_s3=2,
            dense_s1_args={"units": 64},
            dense_s2_args={"units": 64},
            dense_s3_args={"units": 64},
            input_dim=input_dim,
        )
        self.inv1 = InvariantModule(settings)
        self.equiv1 = EquivariantModule(settings, self.inv1)
        settings_l2 = settings.copy()
        settings_l2["input_dim"] = 32
        self.inv2 = InvariantModule(settings_l2)
        self.equiv2 = EquivariantModule(settings_l2, self.inv2)
        settings_l3 = settings.copy()
        settings_l3["input_dim"] = 32
        self.inv3 = InvariantModule(settings_l3)
        self.out_layer = keras.layers.Dense(output_dim, kernel_initializer=self._init)

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


class MyContinuousApproximator(bf.ContinuousApproximator):
    def call(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            return self.compute_metrics(**inputs, **kwargs)
        return super().call(inputs, **kwargs)


def build_bayesflow_model(d, d_x, summary_dim=10):
    summary_net = TimeSeriesSummary(input_dim=d_x, output_dim=summary_dim)
    inference_net = bf.networks.CouplingFlow(
        num_params=d,
        num_coupling_layers=4,
        coupling_settings={"dense_args": dict(units=64, activation="relu")},
    )
    adapter = bf.Adapter([ToNumpy()])
    approximator = MyContinuousApproximator(
        inference_network=inference_net,
        summary_network=summary_net,
        adapter=adapter,
    )
    return approximator
