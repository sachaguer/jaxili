"""
Direct Posterior.

This module contains the Direct Posterior class. It is used when doing neural posterior estimation where the neural network is trained to approximate the posterior directly.
"""

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array

from jaxili.model import NDENetwork
from jaxili.posterior import NeuralPosterior
from jaxili.train import TrainState


class DirectPosterior(NeuralPosterior):
    r"""
    Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods.

    The class wraps the trained neural network using Neural Posterior Estimation (NPE).
    """

    def __init__(
        self,
        model: NDENetwork,
        state: TrainState,
        verbose: bool = False,
        x: Optional[Array] = None,
    ):
        """
        Initialize the Neural Posterior.

        Parameters
        ----------
        model : NDENetwork
            The neural network used to generate the posterior.
        state : dict
            The state of the neural network.
        verbose : bool
            Whether to print information. (Default: False)
        """
        super().__init__(model, state, verbose, x)

    def sample(self, num_samples: int, key: Array, x: Optional[Array] = None):
        r"""
        Sample from the posterior.

        Parameters
        ----------
        num_samples : int
            The number of samples to draw.
        key : Array
            The random key used to generate the samples.
        x : Array
            The data used to condition the posterior.

        Returns
        -------
        theta : Array
            The samples from the posterior.
        """
        if x is None:
            x = self.x
            if x is None:
                raise ValueError(
                    "Please set the default data `x` using `set_default_x()` or provide `x` as an argument."
                )
        params = self.state.params
        samples = self.model.apply(
            {"params": params}, x, num_samples, key, method="sample"
        )
        return samples

    def unnormalized_log_prob(self, theta: Array, x: Optional[Array] = None):
        r"""
        Compute the unnormalized log probability of the posterior.

        Parameters
        ----------
        theta : Array
            The parameters to evaluate the log probability.
        x : Array
            The data used to condition the posterior.

        Returns
        -------
        log_prob : Array
            The unnormalized log probability.
        """
        if x is None:
            x = self.x
            if x is None:
                raise ValueError(
                    "Please set the default data `x` using `set_default_x()` or provide `x` as an argument."
                )
        params = self.state.params
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)
        if (x.shape[0] == 1) and (theta.shape[0] > 1):
            x = jnp.repeat(x, theta.shape[0], axis=0)
        elif x.shape[0] != theta.shape[0]:
            raise ValueError(
                "The batch size of `x` must be the same as the batch size of parameters `theta`."
            )
        log_prob = self.model.apply({"params": params}, theta, x, method="log_prob")
        return log_prob
