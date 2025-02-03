import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import numpyro.distributions as dist
import sbibm

from jaxili.inference import NLE, NPE
from jaxili.inference.npe import default_maf_hparams
from jaxili.model import NDENetwork
from jaxili.posterior import DirectPosterior
from jaxili.posterior.mcmc_posterior import nuts_numpyro_kwargs_default
from jaxili.train import TrainState

task = sbibm.get_task("gaussian_linear_uniform")
simulator = task.get_simulator()
prior = task.get_prior()
simulator = task.get_simulator()

train_set_size = 10_000
theta_train = prior(num_samples=train_set_size)
x_train = simulator(theta_train)

theta_train, x_train = np.array(theta_train), np.array(x_train)


def test_direct_posterior():

    checkpoint_path = "~/tests/"
    checkpoint_path = os.path.expanduser(checkpoint_path)

    inference = NPE()

    inference = inference.append_simulations(theta_train, x_train)

    optimizer_hparams = {"lr": 5e-4}

    logger_params = {
        "base_log_dir": checkpoint_path,
    }

    inference.create_trainer(
        optimizer_hparams=optimizer_hparams,
        logger_params=logger_params,
    )

    posterior = inference.build_posterior()

    assert isinstance(
        posterior, DirectPosterior
    ), "The posterior has not the correct type."
    assert isinstance(posterior.model, NDENetwork), "The model is not a NDENetwork."
    assert isinstance(posterior.state, TrainState), "The state is not a TrainState."

    # Test if the density estimator can return a log_prob
    log_prob = posterior.unnormalized_log_prob(theta_train[0:10], x_train[0:10])
    assert log_prob.shape == (
        10,
    ), "The shape of the output of log_prob method is wrong."

    # Test if the density estimator can return samples
    samples = posterior.sample(
        x=x_train[0], num_samples=10_000, key=jax.random.PRNGKey(0)
    )
    assert samples.shape == (
        10_000,
        theta_train[0].shape[0],
    ), "The shape of the samples is wrong."

    # Test if the correct Error are returned.
    try:
        posterior.unnormalized_log_prob(theta_train[0:10])
    except ValueError:
        pass

    try:
        posterior.sample(num_samples=10_000, key=jax.random.PRNGKey(0))
    except ValueError:
        pass

    posterior.set_default_x(x_train[0])

    # Test log_prob with default x
    log_prob = posterior.unnormalized_log_prob(theta_train[0:10])
    assert log_prob.shape == (
        10,
    ), "The shape of the output of log_prob method is wrong."

    # Test sample with default x
    samples = posterior.sample(num_samples=10_000, key=jax.random.PRNGKey(0))
    assert samples.shape == (
        10_000,
        theta_train[0].shape[0],
    ), "The shape of the samples is wrong."

    shutil.rmtree(checkpoint_path)


def test_mcmc_posterior():

    checkpoint_path = "~/tests/"
    checkpoint_path = os.path.expanduser(checkpoint_path)

    inference = NLE()

    inference = inference.append_simulations(theta_train, x_train)

    optimizer_hparams = {"lr": 5e-4}

    logger_params = {
        "base_log_dir": checkpoint_path,
    }

    inference.create_trainer(
        optimizer_hparams=optimizer_hparams,
        logger_params=logger_params,
    )

    posterior = inference.build_posterior(
        prior_distr=dist.Uniform(low=-1.0 * jnp.ones(10), high=1.0 * jnp.ones(10))
    )

    assert posterior.mcmc_method == "nuts_numpyro", "The default mcmc method is wrong."
    assert (
        posterior.mcmc_kwargs == nuts_numpyro_kwargs_default
    ), "The default mcmc kwargs are wrong."

    # Test if the density estimator returns a log_prior
    log_prior = posterior.log_prior(theta_train[0:10])
    assert log_prior.shape == (
        10,
    ), "The shape of the output of log_prior method is wrong."

    # Test if the density estimator returns a log_likelihood
    log_likelihood = posterior.log_likelihood(x_train[0:10], theta_train[0:10])
    assert log_likelihood.shape == (
        10,
    ), "The shape of the output of log_likelihood method is wrong."

    # Test if the density estimator returns a log_prob
    log_prob = posterior.unnormalized_log_prob(theta_train[0:10], x_train[0:10])
    assert log_prob.shape == (
        10,
    ), "The shape of the output of log_prob method is wrong."

    # Sample using NUTS numpyro
    samples = posterior.sample(
        x=x_train[0].reshape((1, -1)), num_samples=2_000, key=jax.random.PRNGKey(0)
    )
    assert samples.shape == (
        2_000,
        theta_train[0].shape[0],
    ), "The shape of the samples is wrong."

    # Sample using HMC numpyro
    posterior.set_mcmc_method("hmc_numpyro")
    samples = posterior.sample(
        x=x_train[0].reshape((1, -1)), num_samples=2_000, key=jax.random.PRNGKey(0)
    )
    assert samples.shape == (
        2_000,
        theta_train[0].shape[0],
    ), "The shape of the samples is wrong."

    # Test if the correct Error are returned.
    try:
        posterior.set_mcmc_method("wrong_method")
    except NotImplementedError:
        pass

    shutil.rmtree(checkpoint_path)
