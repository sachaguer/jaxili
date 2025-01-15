"""
Compressor.

This module contains classes that implement compressors used in JaxILI.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import tensorflow_probability as tfp
from typing import Any, Callable

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions


class MLPCompressor(nn.Module):
    """
    MLP Compressor.

    Defines a MLP compressor to send the summary statistic to the same dimension than the parameters.

    Parameters
    ----------
    hidden_size : list
        List with the size of the hidden layers.
    activation : Callable
        Activation function. Preferably from `jax.nn` or `jax.nn.activation`.
    output_size : int
        Size of the output layer.
    """

    hidden_size: list
    activation: Callable
    output_size: int

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_size:
            x = nn.Dense(size)(x)
            x = self.activation(x)
        x = nn.Dense(self.output_size)(x)
        return x


class CNN2DCompressor(nn.Module):
    """
    CNN2D Compressor

    Defines a 2 dimensional Convolutional Neural Network to compress the data to the same dimension as the parameters

    Parameters
    ----------
    output_size : int
        Size of the output layer
    activation : Callable
        Activation function. Preferably from `jax.nn` or `jax.nn.activation`.
    """

    output_size: int
    activation: Callable

    @nn.compact
    def __call__(self, inputs):
        net_x = nn.Conv(32, 3, 2)(inputs)
        net_x = self.activation(net_x)
        net_x = nn.Conv(64, 3, 2)(net_x)
        net_x = self.activation(net_x)
        net_x = nn.Conv(128, 3, 2)(net_x)
        net_x = self.activation(net_x)
        net_x = nn.avg_pool(net_x, (16, 16), (8, 8), padding="SAME")
        # Flatten the tensor
        net_x = net_x.reshape((net_x.shape[0], -1))
        net_x = nn.Dense(64)(net_x)
        net_x = self.activation(net_x)
        net_x = nn.Dense(self.output_size)(net_x)
        return net_x.squeeze()
