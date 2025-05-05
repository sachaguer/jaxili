"""Validation.

TBD. Scripts to perform validation of the models. (e.g. coverage plots, rank statistics etc...)
"""
from jaxili.posterior.base_posterior import NeuralPosterior
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
    num_samples:int = 1000,
    num_simulations: int = 100,
    **kwargs: Any
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

    if(x_test.shape[0] != theta_test.shape[0]):
        raise ValueError("Number of test samples must be equal for x and theta.")
    if(num_simulations > x_test.shape[0]):
        raise ValueError("Number of simulations cannot be greater than the number of test samples.")
    relevant_variables = np.ravel(kwargs.get("relevant_variables", np.arange(theta_test.shape[-1])))
    if len(relevant_variables) == 0:
        raise ValueError("List of variables to consider cannot be empty.")
    if np.max(relevant_variables) >= theta_test.shape[-1]:
        raise ValueError("List of variables to consider cannot be greater than the number of variables in the posterior.")
    if np.min(relevant_variables) < 0:
        raise ValueError("List of variables to consider cannot be less than 0.")
    relevant_variables = np.unique(np.arange(theta_test.shape[-1])[relevant_variables])
    verbose = kwargs.get("verbose", True)
    
    if(verbose):
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
        print(f"Marginalizing out {theta_test.shape[-1] - len(relevant_variables)} variables.")
    samples = samples[:, :, relevant_variables]
    theta = theta_test[selection][:, relevant_variables]
    return tarp.get_tarp_coverage(samples, theta)