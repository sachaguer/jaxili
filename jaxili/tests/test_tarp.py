import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np

from jaxili.inference.nle import NLE
from jaxili.inference.npe import NPE
from jaxili.validation import get_tarp_coverage
import numpyro.distributions as dist

master_key = jax.random.PRNGKey(0)
num_samples = 10_000
n_dim = 3
n_reals = 11
def simulator(theta, rng_key):
    batch_size = theta.shape[0]
    return (theta[..., None] + jax.random.normal(rng_key, shape=(batch_size, n_dim, n_reals))*0.1).reshape(batch_size, n_dim*n_reals)

train_set_size = 10_000
master_key, theta_key = jax.random.split(master_key)
theta_train = jax.random.uniform(theta_key, shape=(train_set_size, n_dim), minval=jnp.array([-2., -2., -2.]), maxval=jnp.array([2., 2., 2.]))
master_key, simkey = jax.random.split(master_key)
x_train = simulator(theta_train, simkey)

test_set_size = 10_000
master_key, theta_key = jax.random.split(master_key)
theta_test = jax.random.uniform(theta_key, shape=(test_set_size, n_dim), minval=jnp.array([-2., -2., -2.]), maxval=jnp.array([2., 2., 2.]))
master_key, simkey = jax.random.split(master_key)
x_test = simulator(theta_test, simkey)

def test_tarp_coverage_NLE():
    checkpoint_path = "~/tests/"
    checkpoint_path = os.path.expanduser(checkpoint_path)

    # Define the model and inference
    inference = NLE()

    # Append simulations
    inference = inference.append_simulations(theta_train, x_train)

    logger_params = {
        "checkpoint_path": checkpoint_path,
    }
    # Train the model
    inference.train(num_epochs=10, checkpoint_path=checkpoint_path)

    prior_distr = dist.Uniform(low=jnp.array([-2, -2, -2]), high=jnp.array([2, 2, 2]))
    posterior = inference.build_posterior(prior_distr=prior_distr)

    # Get the TARP coverage
    ecp, alpha = get_tarp_coverage(posterior=posterior, theta_test=theta_test, x_test=x_test, key=master_key, num_samples=100, num_simulations=10)
    
    assert ecp is not None, "The exected coverage is None."
    assert alpha is not None, "The credibility values are None."
    assert ecp.shape == alpha.shape, "The shapes of the expected coverage and credibility values do not match."
    assert np.all(np.isfinite(ecp)), "The expected coverage values are not finite."
    assert np.all(np.isfinite(alpha)), "The credibility values are not finite."
    assert np.all(ecp >= 0), "The expected coverage values are not greater than or equal to 0."
    assert np.all(ecp <= 1), "The expected coverage values are not less than or equal to 1."
    assert np.all(alpha >= 0), "The credibility values are not greater than or equal to 0."
    assert np.all(alpha <= 1), "The credibility values are not less than or equal to 1."

    # Clean up the checkpoint directory
    shutil.rmtree(checkpoint_path)


def test_tarp_coverage_NPE():
    checkpoint_path = "~/tests/"
    checkpoint_path = os.path.expanduser(checkpoint_path)

    # Define the model and inference
    inference = NPE()

    # Append simulations
    inference = inference.append_simulations(theta_train, x_train)

    logger_params = {
        "checkpoint_path": checkpoint_path,
    }
    # Train the model
    inference.train(num_epochs=10, checkpoint_path=checkpoint_path)
    posterior = inference.build_posterior()

    # Get the TARP coverage
    ecp, alpha = get_tarp_coverage(posterior=posterior, theta_test=theta_test, x_test=x_test, key=master_key)
    
    assert ecp is not None, "The exected coverage is None."
    assert alpha is not None, "The credibility values are None."
    assert ecp.shape == alpha.shape, "The shapes of the expected coverage and credibility values do not match."
    assert np.all(np.isfinite(ecp)), "The expected coverage values are not finite."
    assert np.all(np.isfinite(alpha)), "The credibility values are not finite."
    assert np.all(ecp >= 0), "The expected coverage values are not greater than or equal to 0."
    assert np.all(ecp <= 1), "The expected coverage values are not less than or equal to 1."
    assert np.all(alpha >= 0), "The credibility values are not greater than or equal to 0."
    assert np.all(alpha <= 1), "The credibility values are not less than or equal to 1."

    # Clean up the checkpoint directory
    shutil.rmtree(checkpoint_path)