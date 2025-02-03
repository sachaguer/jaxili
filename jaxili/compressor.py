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
    MLP Compressor

    Defines a MLP compressor to send the summary statistic to the same dimension than the parameters
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