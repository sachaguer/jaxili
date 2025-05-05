"""Loss.

This module contains useful loss functions used in the neural network training.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

MMD_BANDWIDTH_LIST = [
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    100,
    1e3,
    1e4,
    1e5,
    1e6,
]


def loss_nll_npe(model: Any, params: PyTree, batch: Any) -> Array:
    """
    Negative log-likelihood loss function for NPE methods using a given neural network as a model.

    In NPE, the log-probability is given by the density estimator in parameter space conditioned on the data. The batch should take the form of a Tuple of Arrays were the first one corresponds to the parameters and the second to the simulation outputs.

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

    output = model.apply({"params": params}, thetas, xs, method="log_prob")
    return -jnp.mean(output)


def loss_nll_nle(model: Any, params: PyTree, batch: Any):
    """
    Negative log-likelihood loss function for NLE methods using a given neural network as a model.

    In NLE, the log-probability is given by the density estimator in observation space conditioned on the parameter. The batch should take the form of a Tuple of Arrays were the first one corresponds to the parameters and the second to the simulation outputs.

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

    return -jnp.mean(model.apply({"params": params}, xs, thetas, method="log_prob"))


def gaussian_kernel_matrix(x, y, sigmas=None):
    """
    Compute a Gaussian radial basis functions (RBFs) between the samples of x and y.

    We create a sum of multiple Gaussian kernels each having a width sigma_i.

    Parameters
    ----------
    x: array of shape (num_draws_x, num_features)
    y: array of shape (num_draws_y, num_features)
    sigmas: list(float), optional, default: None
        List which denotes the width of each of the Gaussian in the kernel. A default range is used if sigmas is None.

    Returns
    -------
    kernel values: array of shape (num_draws_x, num_draws_y)
    """
    if sigmas is None:
        sigmas = jnp.array(MMD_BANDWIDTH_LIST)
    norm = lambda v: jnp.sum(v**2, axis=1)
    beta = 1.0 / (2.0 * (jnp.expand_dims(sigmas, 1)))
    dist = jnp.transpose(norm(jnp.expand_dims(x, 2) - jnp.transpose(y)))
    s = jnp.matmul(beta, jnp.reshape(dist, (1, -1)))
    kernel = jnp.reshape(jnp.sum(jnp.exp(-s), axis=0), jnp.shape(dist))
    return kernel


def mmd_kernel(x, y, kernel):
    """
    Compute the Maximum Mean Discrepancy (MMD) between samples of x and y.

    Parameters
    ----------
    x: array of shape (num_draws_x, num_features)
    y: array of shape (num_draws_y, num_features)
    kernel: function
        A kernel function which computes the similarity between two sets of samples.

    Returns
    -------
    float
        Maximum Mean Discrepancy (MMD) between x and y.
    """
    return jnp.mean(kernel(x, x)) + jnp.mean(kernel(y, y)) - 2 * jnp.mean(kernel(x, y))


def maximum_mean_discrepancy(
    source_samples, target_samples, kernel="gaussian", mmd_weight=1.0, minimum=0.0
):
    """
    Compute the Maximum Mean Discrepancy (MMD) between source and target samples.

    Parameters
    ----------
    source_samples: samples from the source distribution. Shape: (N, num_features)
    target_samples: samples from the target distribution. Shape: (M, num_features)
    kernel: kernel function to use for the MMD computation. str: "gaussian"
    mmd_weight: weight for the MMD loss. Default: 1.0
    minimum: minimum value for the MMD loss. Default: 0.0

    Returns
    -------
    float
        Maximum Mean Discrepancy (MMD) between source and target samples.
    """
    assert kernel == "gaussian", "Only Gaussian kernel is supported for now"

    if kernel == "gaussian":
        kernel_fun = gaussian_kernel_matrix

    loss_value = mmd_kernel(source_samples, target_samples, kernel=kernel_fun)
    loss_value = jnp.maximum(loss_value, minimum) * mmd_weight
    return loss_value


def mmd_summary_space(summary_outputs, rng, z_dist="gaussian", kernel="gaussian"):
    """
    Compute the Maximum Mean Discrepancy (MMD) between the summary outputs and samples from a unit Gaussian distribution.

    Parameters
    ----------
    summary_outputs: array of shape (num_samples, num_features)
        Summary outputs from the neural network.
    rng: jax.random.PRNGKey
        Random key for reproducibility.
    z_dist: str, optional
        Distribution of the samples. Default: "gaussian"
    kernel: str, optional
        Kernel function to use for the MMD computation. Default: "gaussian"

    Returns
    -------
    float
        Maximum Mean Discrepancy (MMD) between the summary outputs and samples from a unit Gaussian distribution.
    """
    assert z_dist == "gaussian", "Only Gaussian distribution is supported for now"
    assert kernel == "gaussian", "Only Gaussian kernel is supported for now"

    z_samples = jax.random.normal(rng, shape=summary_outputs.shape)
    mmd_loss = maximum_mean_discrepancy(summary_outputs, z_samples, kernel=kernel)
    return mmd_loss


def loss_mmd_npe(model, params, batch):
    """
    Compute the Maximum Mean Discrepancy (MMD) loss for Neural Posterior Estimation.

    Parameters
    ----------
    compress: function
        Neural network function to compress the data.
    nf: function
        Neural network function to compute the log-probability conditionally to a random variable.
    params: jnp.array
        Parameters of the neural network.
    batch: jnp.array
        Batch of data.

    Returns
    -------
    float
        Maximum Mean Discrepancy (MMD) loss.
    """
    compress = lambda params, x: model.apply({"params": params}, x, method="compress")
    nf = lambda params, theta, z: model.apply(
        {"params": params}, z, theta, model="NPE", method="log_prob_from_compressed"
    )
    theta, x = batch

    # compress the data
    z = compress(params, x)

    # Compute the MMD loss
    rng_key = jax.random.PRNGKey(0)  # Could probably be improved
    mmd_loss = mmd_summary_space(z, rng_key)
    # Compute the log-probability
    log_prob = nf(params, theta, z)
    return -jnp.mean(log_prob) + mmd_loss

def loss_mse(model: Any, params: PyTree, batch: Any):
    """
    Mean Squared Error (MSE).
    """
    thetas, xs = batch
    compressed_xs = model.apply({"params": params}, xs)
    return jnp.mean((compressed_xs - thetas)**2)

def loss_mae(model: Any, params: PyTree, batch: Any):
    """
    Mean Absolute Error (MAE).
    """
    xs, thetas = batch
    compressed_xs = model.apply({"params": params}, xs)
    return jnp.mean(jnp.abs(compressed_xs - thetas))
