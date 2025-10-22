"""Validation.

Scripts to perform validation of the models. (e.g. coverage plots, rank statistics etc...)
"""
import jax
import jax.numpy as jnp
import warnings
from jaxili.posterior.base_posterior import NeuralPosterior
from jaxili.posterior import DirectPosterior
from typing import Any, Dict, Optional
import numpy as np
from jaxtyping import Array
import tarp
from tqdm import tqdm
import jax.random as jr
import matplotlib.pyplot as plt


def get_tarp_coverage(
    posterior: NeuralPosterior,
    x_test: Array,
    theta_test: Array,
    key: Array,
    num_samples: int = 1000,
    num_simulations: int = 100,
    **kwargs: Any,
):
    """
    Compute the TARP coverage of the posterior.

    Parameters
    ----------
    posterior : NeuralPosterior
        The posterior to evaluate.
    x_test : Array
        Test simulation data.
    theta_test : Array
        True values of the parameters.
    key : Array
        The random key to use for sampling.
    num_samples : int
        The number of samples to draw from the posterior.
    num_simulations : int
        The number of simulations to run. Must be less than or equal to the number of test samples.
    **kwargs : Any
        Additional keyword arguments:
            mcmc_method : str
                The MCMC method to use for sampling.
            mcmc_kwargs : dict
                The keyword arguments to pass to the MCMC method.
            verbose : bool
                Whether to print information. (Default: True)
            relevant_variables : Array
                The relevant variables to consider. Marginalize out the rest. If None, all variables are considered.

    Returns
    -------
    Tuple[Array, Array]
        Expected coverage probability (ecp) and credibility values (alpha).
    """
    if x_test.shape[0] != theta_test.shape[0]:
        raise ValueError("Number of test samples must be equal for x and theta.")
    if num_simulations > x_test.shape[0]:
        raise ValueError(
            "Number of simulations cannot be greater than the number of test samples."
        )
    relevant_variables = np.ravel(
        kwargs.get("relevant_variables", np.arange(theta_test.shape[-1]))
    )
    if len(relevant_variables) == 0:
        raise ValueError("List of variables to consider cannot be empty.")
    if np.max(relevant_variables) >= theta_test.shape[-1]:
        raise ValueError(
            "List of variables to consider cannot be greater than the number of variables in the posterior."
        )
    if np.min(relevant_variables) < 0:
        raise ValueError("List of variables to consider cannot be less than 0.")
    relevant_variables = np.unique(np.arange(theta_test.shape[-1])[relevant_variables])
    verbose = kwargs.get("verbose", True)

    if verbose:
        print("Computing TARP coverage...")
    samples = []
    key, selection_key = jr.split(key, num=2)
    selection = jr.permutation(key, x_test.shape[0])[:num_simulations]
    pbar = tqdm(x_test[selection]) if verbose else x_test[selection]
    for t_ in pbar:
        key, sample_key = jr.split(key, num=2)
        sample = posterior.sample(
            num_samples=num_samples,
            key=sample_key,
            x=t_.reshape(1, -1),
            **kwargs,
        )
        samples.append(sample)
    samples = np.stack(samples)
    samples = np.moveaxis(samples, 0, 1)

    if verbose and len(relevant_variables) < theta_test.shape[-1]:
        print(
            f"Marginalizing out {theta_test.shape[-1] - len(relevant_variables)} variables."
        )
    samples = samples[:, :, relevant_variables]
    theta = theta_test[selection][:, relevant_variables]
    return tarp.get_tarp_coverage(samples, theta)

def get_fom_ij_from_samples(samples_i, samples_j):
    """
    Compute the Figure of Merit of the 2D posterior of two parameters indexed i and j from samples drawn from the posterior.

    Parameters
    ----------
    samples_i : np.array
        Samples of the first parameter.
    samples_j : np.array
        Samples of the second parameter.

    Returns
    -------
    float
        Figure of Merit of the 2D posterior between parameters i and j.
    """
    samples_column = np.column_stack([samples_i, samples_j])
    cov_params = np.cov(samples_column.T)

    return np.sqrt(np.linalg.det(np.linalg.inv(cov_params)))

def get_fom_from_samples(samples):
    """
    Compute the Figure of Merit of the 2D posterior of all pairs of parameters from samples drawn from the posterior.

    Parameters
    ----------
    samples : Array
        Samples drawn from the posterior.

    Returns
    -------
    Array
        Array containing the FoM between each pair of parameters.
    """
    n_parameters = samples.shape[1]
    fom_matrix = jnp.zeros((n_parameters, n_parameters))

    for i in range(n_parameters):
        for j in range(i+1, n_parameters):
            fom_matrix = fom_matrix.at[i,j].set(get_fom_ij_from_samples(samples[:, i], samples[:, j]))
    return fom_matrix + fom_matrix.T

def get_fisher_from_posterior(
    observation,
    fiducial_params,
    posterior : DirectPosterior
):
    """
    Compute the Fisher matrix of the sampled parameters using the posterior object produced from NPE.

    Parameters
    ----------
    observation : Array
        The observation conditionning the NDE.
    fiducial_params : Array
        The fiducial parameters at which the Fisher matrix is evaluated.
    posterior : DirectPosterior
        The posterior object used to compute the Fisher matrix with AutoDiff.

    Returns
    -------
    Array
        Fisher matrix of the sampled parameters.
    """
    log_prob_fn = lambda params: posterior.unnormalized_log_prob(theta=params, x=observation)

    H = jax.hessian(log_prob_fn)(fiducial_params)
    F = -H.squeeze()

    assert F.shape == (fiducial_params.shape[1], fiducial_params.shape[1])
    return F

def get_fom_ij_from_posterior(
    index_i,
    index_j,
    observation = None,
    fiducial_params = None,
    posterior : DirectPosterior = None,
    fisher_matrix : Array = None
):
    """
    Computes the Figure of Merit of the 2D posterior of two parameters indexed i and j using the posterior and AutoDiff to compute the fisher matrix.

    Parameters
    ----------
    index_i : int
        The index of the first parameter.
    index_j : int
        The index of the second parameters.
    observation : Array
        The observation conditionning the NDE.
    fiducial_params : Array
        The fiducial parameters at which the Fisher matrix is evaluated.
    posterior : DirectPosterior
        The posterior object used to compute the Fisher matrix with AutoDiff. If None, uses the fisher_matrix.
    fisher_matrix : Array
        Fisher matrix of the parameters. If None, uses the posterior object to compute it.

    Returns
    -------
    float
        Figure of Merit of the 2D posterior between parameters i and j. Raise an error if posterior and fisher_matrix are both None.
    """
    assert ~((posterior is None) & (fisher_matrix is None)), "You must specify either a posterior object or a fisher matrix to compute the Figure of Merit."

    if (fisher_matrix is not None) & (posterior is not None):
        warnings.warn("You specified a posterior object but a fisher matrix is already provided. Using the fisher matrix")

    if fisher_matrix is None:
        assert (observation is not None) & (fiducial_params is not None), "You need an observation and fiducial parameters to compute the Fisher matrix."
        fisher_matrix = get_fisher_from_posterior(observation, fiducial_params, posterior)

    rows = jnp.array([index_i, index_j])
    cols = jnp.array([index_i, index_j])
    F_sub = fisher_matrix[jnp.ix_(rows, cols)]

    return jnp.sqrt(jnp.linalg.det(F_sub))

def get_fom_from_posterior(
    observation = None,
    fiducial_params = None,
    posterior : DirectPosterior = None,
    fisher_matrix = None
):
    """
    Computes the Figure of Merit of the 2D posterior of all pairs of parameters using the posterior and AutoDiff to compute the fisher matrix.

    Parameters
    ----------
    observation : Array
        The observation conditionning the NDE.
    fiducial_params : Array
        The fiducial parameters at which the Fisher matrix is evaluated.
    posterior : DirectPosterior
        The posterior object used to compute the Fisher matrix with AutoDiff. If None, uses the fisher_matrix.
    fisher_matrix : Array
        Fisher matrix of the parameters. If None, uses the posterior object to compute it.

    Returns
    -------
    float
        Figure of Merit of the 2D posterior between parameters i and j. Raise an error if posterior and fisher_matrix are both None.
    """
    assert ~((posterior is None) & (fisher_matrix is None)), "You must specify either a posterior object or a fisher matrix to compute the Figure of Merit."

    if (fisher_matrix is not None) & (posterior is not None):
        warnings.warn("You specified a posterior object but a fisher matrix is already provided. Using the fisher matrix")

    if fisher_matrix is None:
        assert (observation is not None) & (fiducial_params is not None), "You need an observation and fiducial parameters to compute the Fisher matrix."
        fisher_matrix = get_fisher_from_posterior(observation, fiducial_params, posterior)

    fom_matrix = jnp.zeros_like(fisher_matrix)

    n_params = fom_matrix.shape[0]
    for i in range(n_params):
        for j in range(i+1, n_params):
            fom_matrix = fom_matrix.at[i, j].set(get_fom_ij_from_posterior(i, j, observation, fiducial_params, fisher_matrix=fisher_matrix))
    return fom_matrix + fom_matrix.T
    
