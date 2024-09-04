import jax
import jax.numpy as jnp

from typing import Any
from jaxtyping import Array, Float, PyTree

def loss_nll_npe(model: Any, params: PyTree, batch: Any)->Array:
    """
    Negative log-likelihood loss function for NPE methods
    using a given neural network as a model.
    In NPE, the log-probability is given by the density estimator in
    parameter space conditioned on the data.

    Parameters
    ----------
    model : Any
        Neural network model from `jaxili.model`.
    params : PyTree
        Parameters of the neural network.
    batch : Any
        Batch of (parameters, outputs) to compute the loss.

    Returns
    -------
    Array
        Mean of the negative log-likelihood loss across the batch.
    """

    thetas, xs = batch

    output =  model.apply({'params': params}, thetas, xs, method='log_prob')
    return -jnp.mean(output)

def loss_nll_nle(model: Any, params: PyTree, batch: Any):
    """
    Negative log-likelihood loss function for NLE methods
    using a given neural network as a model.
    In NPE, the log-probability is given by the density estimator in
    parameter space conditioned on the data.

    Parameter
    ---------
    model : Any
        Neural network model from `jaxili.model`.
    params : PyTree
        Parameters of the neural network.
    batch : Any
        Batch of (parameters, outputs) to compute the loss.

    Returns
    -------
    Array
        Mean of the negative log-likelihood loss across the batch.
    """

    thetas, xs = batch

    return -jnp.mean(model.apply({'params': params}, xs, thetas, method='log_prob'))