"""
MCMC Posterior.

This module contains the MCMCPosterior class that wraps the NeuralPosterior class to perform MCMC sampling.
"""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
# from flowMC.resource.local_kernel.MALA import MALA
# from flowMC.Sampler import Sampler
from jaxtyping import Array
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.infer.util import init_to_value

from jaxili.model import NDENetwork
from jaxili.posterior import NeuralPosterior
from jaxili.train import TrainState

implemented_method = ["nuts_numpyro", "hmc_numpyro"]

nuts_numpyro_kwargs_default = {}
hmc_numpyro_kwargs_default = {}
mala_flowmc_kwargs_default = {}


class MCMCPosterior(NeuralPosterior):
    r"""
    Likelihood $p(x|\theta)$ with `log_prob()` and `sample()` methods.

    The class wraps the trained neural network using Neural Likelihood Estimation (NLE).
    Sampling is performed using Markov Chain Monte Carlo (MCMC) methods to get samples from the posterior.
    """

    def __init__(
        self,
        model: NDENetwork,
        state: TrainState,
        prior_distr: dist.Distribution,
        verbose: Optional[bool] = False,
        x: Optional[Array] = None,
        mcmc_method: Optional[str] = "nuts_numpyro",
        mcmc_kwargs: Optional[dict] = nuts_numpyro_kwargs_default,
    ):
        """
        Initialize the MCMC Posterior.

        Parameters
        ----------
        model : NDENetwork
            The neural network used to generate the posterior.
        state : TrainState
            The state of the neural network.
        prior_distr : ...
            The prior distribution of the parameters. (One must specify a prior to perform MCMC sampling.)
        verbose : bool
            Whether to print information. (Default: False)
        x : Array
            The data used to condition the posterior. (Default: None)
        mcmc_method : str
            The MCMC method to use. (Default: 'hmc_numpyro')
        mcmc_kwargs : dict
            The keyword arguments for the MCMC method. (Default: hmc_numpyro_kwargs_default)
        """
        super().__init__(model, state, verbose, x)
        self.set_mcmc_method(mcmc_method)
        self.set_mcmc_kwargs(mcmc_kwargs)
        if self.verbose:
            print(f"Using MCMC method: {mcmc_method}")
            print(f"MCMC kwargs: {mcmc_kwargs}")
        self.set_prior(prior_distr)

    def sample(self, num_samples: int, key: Array, x: Optional[Array] = None, **kwargs):
        r"""
        Sample from the posterior using MCMC.

        Parameters
        ----------
        x : Array
            The data used to sample the parameters.
        num_samples : int
            The number of samples to draw.
        key : Array
            The random key used to generate the samples.

        Returns
        -------
        Array
            The samples from the posterior.
        """
        if x is None:
            try:
                x = self.x
            except:
                raise ValueError(
                    "The data x must be specified or loaded in the posterior with `set_default_x()`."
                )
        self.mcmc_kwargs.update({"num_samples": num_samples})
        self.mcmc_kwargs.update({"progress_bar": kwargs.pop("progress_bar", False)})
        num_chains = self.mcmc_kwargs.get("num_chains", 1)
        sample_key, key = jax.random.split(key)
        initial_states = self._get_initial_state(x, num_chains, sample_key, **kwargs)
        if self.mcmc_method == "nuts_numpyro":
            samples = self._nuts_numpyro(x, key, initial_states, self.mcmc_kwargs)
        elif self.mcmc_method == "hmc_numpyro":
            samples = self._hmc_numpyro(x, key, initial_states, self.mcmc_kwargs)
        elif self.mcmc_method == "mala_flowmc":
            samples = self._mala_flowmc(x, key, initial_states, self.mcmc_kwargs)
        else:
            raise NotImplementedError(
                f"The MCMC method {self.mcmc_method} is not implemented. Check print_implemented_methods()."
            )
        self.mcmc_kwargs.pop("num_samples")

        return samples

    def log_prior(self, theta):
        r"""
        Compute the log prior of the parameters.

        Parameters
        ----------
        theta : Array
            The parameters to evaluate the log prior.

        Returns
        -------
        Array
            The log prior of the parameters.
        """
        log_prior = self.prior_distr.log_prob(theta)
        if len(log_prior.shape) > 1:
            log_prior = jnp.sum(log_prior, axis=-1)
        return log_prior

    def log_likelihood(self, x: Array, theta: Array):
        r"""
        Compute the log-likelihood learned by the neural density estimator.

        Parameters
        ----------
        theta : Array
            The parameters to evaluate the log probability.
        x : Array
            The data used to condition the posterior.

        Returns
        -------
        Array
            The unnormalized log probability.
        """
        params = self.state.params
        log_likelihood = self.model.apply(
            {"params": params}, x, theta, method="log_prob"
        ).squeeze()
        return log_likelihood

    def unnormalized_log_prob(self, theta: Array, x: Optional[Array] = None):
        """
        Compute the unnormalized log probability of the posterior.

        Parameters
        ----------
        theta : Array
            The parameters to evaluate the log probability.
        x : Array
            The data used to condition the posterior.

        Returns
        -------
        Array
            The unnormalized log probability.
        """
        return self.log_prior(theta) + self.log_likelihood(x, theta)

    def _build_model_numpyro(self, x: Array):
        """
        Create a function corresponding to the Bayesian model in numpyro.

        Parameters
        ----------
        x : Array
            The data used to condition the posterior.

        Returns
        -------
        Callable
            The model function.
        """

        def model(data):
            theta = numpyro.sample("theta", self.prior_distr)

            z = numpyro.deterministic("z", theta)

            likelihood = self.log_likelihood(x, theta.reshape((1, theta.shape[0])))

            numpyro.factor("log_likelihood", likelihood)

        return model

    def _nuts_numpyro(
        self,
        x: Array,
        key: Array,
        initial_state: Array,
        mcmc_kwargs: Optional[dict] = nuts_numpyro_kwargs_default,
    ):
        """
        Perform MCMC sampling using the No-U-Turn Sampler (NUTS) in numpyro.

        Parameters
        ----------
        x : Array
            The data used to condition the posterior.
        key : Array
            The random key used to generate the samples.
        initial_state : Array
            The initial state of the MCMC sampler.
        mcmc_kwargs: dict
            The keyword arguments for the MCMC method. (Default: nuts_numpyro_kwargs_default)

        Returns
        -------
        Array
            The samples from the posterior.
        """
        model = self._build_model_numpyro(x)
        adapt_step_size = mcmc_kwargs.get("adapt_step_size", True)
        init_values = initial_state
        nuts_kernel = NUTS(
            model,
            adapt_step_size=adapt_step_size,
            init_strategy=init_to_value(values=init_values),
        )

        num_warmup = mcmc_kwargs.get("num_warmup", 500)
        num_samples = mcmc_kwargs.get("num_samples", 2000)
        num_chains = mcmc_kwargs.get("num_chains", 1)
        progress_bar = mcmc_kwargs.get("progress_bar", False)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )

        mcmc.run(key, data=x)

        samples = mcmc.get_samples()["theta"]

        return samples

    def _hmc_numpyro(
        self,
        x: Array,
        key: Array,
        initial_state: Array,
        mcmc_kwargs: Optional[dict] = hmc_numpyro_kwargs_default,
    ):
        """
        Perform MCMC sampling using the Hamiltonian Monte Carlo (HMC) in numpyro.

        Parameters
        ----------
        x : Array
            The data used to condition the posterior.
        key : Array
            The random key used to generate the samples.
        initial_state : Array
            The initial state of the MCMC sampler.
        mcmc_kwargs: dict
            The keyword arguments for the MCMC method. (Default: hmc_numpyro_kwargs_default)

        Returns
        -------
        Array
            The samples from the posterior.
        """
        model = self._build_model_numpyro(x)
        adapt_step_size = mcmc_kwargs.get("adapt_step_size", True)
        init_values = initial_state
        hmc_kernel = HMC(
            model,
            adapt_step_size=adapt_step_size,
            init_strategy=init_to_value(values=init_values),
        )

        num_warmup = mcmc_kwargs.get("num_warmup", 500)
        num_samples = mcmc_kwargs.get("num_samples", 2000)
        num_chains = mcmc_kwargs.get("num_chains", 1)
        progress_bar = mcmc_kwargs.get("progress_bar", False)
        mcmc = MCMC(
            hmc_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )

        mcmc.run(key, data=x)

        samples = mcmc.get_samples()["theta"]

        return samples

    def _mala_flowmc(
        self,
        x: Array,
        key: Array,
        initial_state: Array,
        mcmc_kwargs: Optional[dict] = mala_flowmc_kwargs_default,
    ):
        """
        Perform MCMC sampling using the Metropolis-Adjusted Langevin Algorithm (MALA) in FlowMC.

        WARNING: Current version does not work. Will be updated in future releases.

        Parameters
        ----------
        x : Array
            The data used to condition the posterior.
        key : Array
            The random key used to generate the samples.
        initial_state : Array
            The initial state of the MCMC sampler.
        mcmc_kwargs: dict
            The keyword arguments for the MCMC method. (Default: mala_flowmc_kwargs_default)

        Returns
        -------
        Array
            The samples from the posterior.
        """
        raise NotImplementedError(
            "Sampling with FlowMC is not yet implemented in JaxILI."
        )
        """ num_samples = mcmc_kwargs.get("num_samples", 2000)
        num_chains = mcmc_kwargs.get("num_chains", 1)
        n_dim = self.prior_distr.sample(sample_shape=(1,), key=key).shape[
            0
        ]  # Get the dimension of the parameters (we can probably do better)
        # We can probably inherit this property from the trainer module.

        # Setup the Normalizing Flow
        n_layers = mcmc_kwargs.get("n_layers", 3)
        hidden_size = mcmc_kwargs.get("hidden_size", [64, 64])
        num_bins = mcmc_kwargs.get("num_bins", 8)  # Number of bins in the spline

        nf_key, key = jax.random.split(key)
        model = MaskedCouplingRQSpline(n_dim, n_layers, hidden_size, num_bins, nf_key)

        # Setup the MALA kernel
        step_size = mcmc_kwargs.get("step_size", 1e-1)
        local_sampler = MALA(self.unnormalized_log_prob, True, step_size=step_size)

        # Create the sampler
        n_local_steps = mcmc_kwargs.get("n_local_steps", 50)
        n_global_steps = mcmc_kwargs.get("n_global_steps", 50)
        n_epochs = mcmc_kwargs.get("n_epochs", 30)
        learning_rate = mcmc_kwargs.get("learning_rate", 1e-2)
        nf_sampler = Sampler(
            n_dim,
            key,
            x,
            local_sampler,
            model,
            n_local_steps=n_local_steps,
            n_global_steps=n_global_steps,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=num_samples,
            n_chains=num_chains,
        )

        # Sample!
        initial_state = jnp.expand_dims(initial_state, axis=1)
        nf_sampler.sample(initial_state, x)
        chains, log_prob, local_accs, global_accs = (
            nf_sampler.get_sampler_state().values()
        )
        chains = chains.squeeze()
        return chains """

    def _get_initial_state(self, x: Array, num_chains: int, key: Array, **kwargs):
        """
        Get the initial state of the MCMC sampler. The state is obtained via resampling proposal samples with log_prob weights.

        Parameters
        ----------
        x : Array
            The data used to condition the posterior.
        num_chains : int
            The number of chains.
        key : Array
            The random key used to generate the samples.

        Returns
        -------
        Array
            The initial state of the MCMC sampler.
        """
        initial_state = []

        # Define the potential function
        def potential_fn(theta):
            x_ = x * jnp.ones(
                (theta.shape[0], 1)
            )  # Reshape x to match the shape of theta
            return self.unnormalized_log_prob(theta, x_)

        for _ in range(num_chains):
            proposal_samples, key = self._resample_proposal(potential_fn, key, **kwargs)
            initial_state.append(proposal_samples)
        return jnp.concatenate(initial_state, axis=0)

    def _resample_proposal(
        self,
        potential_fn: Callable,
        key: Array,
        num_candidate_samples: int = 10_000,
        num_batches: int = 1,
        **kwargs,
    ):
        """
        Resample the proposal samples using the neural density estimator.

        Parameters
        ----------
        potential_fn : Callable
            The potential function of the MCMC sampler.
        num_candidate_samples : int
            The number of candidate samples to generate.
        num_batches : int
            The number of batches to generate the candidate samples.

        Returns
        -------
        Array
            The proposal samples.
        Array
            The modified random key used to generate the samples.
        """
        log_weights = []
        init_state_candidates = []
        for _ in range(num_batches):
            subkey, key = jax.random.split(key)
            batch_draws = self.prior_distr.sample(
                sample_shape=(num_candidate_samples,), key=subkey
            )
            init_state_candidates.append(batch_draws)
            log_weights.append(potential_fn(batch_draws))
        log_weights = jnp.concatenate(log_weights, axis=0)
        init_state_candidates = jnp.concatenate(init_state_candidates, axis=0)

        # Normalize the weights in log-space.
        log_weights = log_weights - jax.scipy.special.logsumexp(log_weights, axis=0)
        probs = jnp.exp(log_weights)
        probs = probs.at[jnp.isnan(probs)].set(0.0)
        probs = probs.at[jnp.isinf(probs)].set(0.0)
        probs = probs / jnp.sum(probs)

        subkey, key = jax.random.split(key)
        idxs = jax.random.choice(
            subkey,
            jnp.arange(num_candidate_samples * num_batches),
            shape=(1,),
            replace=True,
            p=probs,
        )

        proposal_samples = init_state_candidates[idxs]

        return proposal_samples, key

    def set_default_x(self, x: Array):
        """Set the default data for the posterior."""
        self.x = x

    def set_prior(self, prior_distr: dist.Distribution):
        """Set the prior distribution for the parameters."""
        self.prior_distr = prior_distr

    def set_mcmc_method(self, mcmc_method: str):
        """Set the MCMC method to use."""
        if mcmc_method not in implemented_method:
            raise NotImplementedError(
                f"The MCMC method {mcmc_method} is not implemented. Check print_implemented_methods()."
            )
        self.mcmc_method = mcmc_method

    def set_mcmc_kwargs(self, mcmc_kwargs: dict):
        """Set the keyword arguments for the MCMC method."""
        self.mcmc_kwargs = mcmc_kwargs

    def print_implemented_methods(self):
        """Print the implemented MCMC methods."""
        print(f"Implemented MCMC methods: {implemented_method}")
