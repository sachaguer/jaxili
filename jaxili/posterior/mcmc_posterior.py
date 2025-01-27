from typing import Optional, Any
from jaxtyping import Array

import jax
import jax.numpy as jnp

from jaxili.model import NDENetwork
from jaxili.train import TrainState
from jaxili.posterior import NeuralPosterior

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC

import blackjax

nuts_numpyro_kwargs_default = {}
hmc_numpyro_kwargs_default = {}
hmc_blackjax_kwargs_default = {}
nuts_blackjax_kwargs_default = {}
mala_flowmc_kwargs_default = {}



class MCMCPosterior(NeuralPosterior):
    r"""
    Likelihood $p(x|\theta)$ with `log_prob()` and `sample()` methods.
    The class wraps the trained neural network using Neural Likelihood Estimation (NPE).
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

    def sample(self, x: Array, num_samples: int, key: Array):
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
        """ 
        self.mcmc_kwargs.update({"num_samples": num_samples})
        num_chains = self.mcmc_kwargs.get("num_chains", 1)
        sample_key, key = jax.random.split(key)
        initial_states = self.prior_distr.sample(sample_shape=(num_chains,), key=sample_key)
        if self.mcmc_method == "nuts_numpyro":
            samples = self._nuts_numpyro(x, key, self.mcmc_kwargs)
        elif self.mcmc_method == "hmc_numpyro":
            samples = self._hmc_numpyro(x, key, self.mcmc_kwargs)
        elif self.mcmc_method == "hmc_blackjax":
            samples = self._hmc_blackjax(x, key, initial_states, self.mcmc_kwargs)
        elif self.mcmc_method == "nuts_blackjax":
            samples = self._nuts_blackjax(x, key, initial_states, self.mcmc_kwargs)
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
        log_prior : Array
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
        log_prob : Array
            The unnormalized log probability.
        """
        params = self.state.params
        log_likelihood = self.model.apply({"params": params}, x, theta, method="log_prob").squeeze()
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
        log_prob : Array
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
        model : Callable
            The model function.
        """

        def model(data):
            theta = numpyro.sample("theta", self.prior_distr)

            z = numpyro.deterministic("z", theta)

            likelihood = self.log_likelihood(x, theta.reshape((1, theta.shape[0])))

            numpyro.factor("log_likelihood", likelihood)

        return model
    
    def _blackjax_inference_loop_multiple_chains(
        self, rng_key: Array, kernel: Any, initial_state: Array, num_samples: int, num_chains: int
    ):
        """
        Function implementing the inference loop for multiple chains in Blackjax.

        Parameters
        ----------
        rng_key : Array
            The random key used to generate the samples.
        kernel : Any
            The transition kernel.
        initial_state : Array
            The initial state of the MCMC sampler.
        num_samples : int
            The number of samples to draw.
        num_chains : int
            The number of chains to run.

        Returns
        -------
        states : Array
            The samples from the posterior.
        """
        @jax.jit
        def one_step(states, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            states, _ = jax.vmap(kernel)(keys, states)
            return states, states
        
        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    def _nuts_numpyro(
        self,
        x : Array,
        key : Array,
        mcmc_kwargs: Optional[dict] = nuts_numpyro_kwargs_default
    ):
        """
        Perform MCMC sampling using the No-U-Turn Sampler (NUTS) in numpyro.

        Parameters
        ----------
        x : Array
            The data used to condition the posterior.
        key : Array
            The random key used to generate the samples.
        mcmc_kwargs: dict
            The keyword arguments for the MCMC method. (Default: nuts_numpyro_kwargs_default)

        Returns
        -------
        samples : Array
            The samples from the posterior.
        """

        model = self._build_model_numpyro(x)
        adapt_step_size = mcmc_kwargs.get("adapt_step_size", True)
        nuts_kernel = NUTS(model, adapt_step_size=adapt_step_size)

        num_warmup = mcmc_kwargs.get("num_warmup", 500)
        num_samples = mcmc_kwargs.get("num_samples", 2000)
        num_chains = mcmc_kwargs.get("num_chains", 1)
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

        mcmc.run(key, data=x)

        samples = mcmc.get_samples()['theta']

        return samples
    
    def _hmc_numpyro(
        self,
        x : Array,
        key : Array,
        mcmc_kwargs: Optional[dict] = hmc_numpyro_kwargs_default
    ):
        """
        Perform MCMC sampling using the Hamiltonian Monte Carlo (HMC) in numpyro.

        Parameters
        ----------
        x : Array
            The data used to condition the posterior.
        key : Array
            The random key used to generate the samples.
        mcmc_kwargs: dict
            The keyword arguments for the MCMC method. (Default: hmc_numpyro_kwargs_default)

        Returns
        -------
        samples : Array
            The samples from the posterior.
        """
        model = self._build_model_numpyro(x)
        adapt_step_size = mcmc_kwargs.get("adapt_step_size", True)
        hmc_kernel = HMC(model, adapt_step_size=adapt_step_size)

        num_warmup = mcmc_kwargs.get("num_warmup", 500)
        num_samples = mcmc_kwargs.get("num_samples", 2000)
        num_chains = mcmc_kwargs.get("num_chains", 1)
        mcmc = MCMC(hmc_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

        mcmc.run(key, data=x)

        samples = mcmc.get_samples()['theta']

        return samples

    def _hmc_blackjax(
        self,
        x : Array,
        key : Array,
        initial_state : Array,
        mcmc_kwargs: Optional[dict] = hmc_blackjax_kwargs_default

    ):
        """
        Perform MCMC sampling using the Hamiltonian Monte Carlo (HMC) in Blackjax.

        Parameters
        ----------
        x : Array
            The data used to condition the posterior.
        key : Array
            The random key used to generate the samples.
        initial_state : Array
            The initial state of the MCMC sampler.
        mcmc_kwargs: dict
            The keyword arguments for the MCMC method. (Default: hmc_blackjax_kwargs_default)

        Returns
        -------
        samples : Array
            The samples from the posterior.
        """
        def log_prob(theta):
            logprob = self.unnormalized_log_prob(theta, x)
            return jnp.sum(logprob)
        
        #Fetch the hyperparameters to create the kernel
        step_size = mcmc_kwargs.get("step_size", 1e-3)
        num_integration_steps = mcmc_kwargs.get("num_integration_steps", 60)
        inverse_mass_matrix = mcmc_kwargs.get("inverse_mass_matrix", jnp.ones(10))
        hmc = blackjax.hmc(log_prob, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix, num_integration_steps=num_integration_steps)

        #Initialize the state and the kernel
        num_chains = mcmc_kwargs.get("num_chains", 1)
        initial_state = jnp.expand_dims(initial_state, axis=1)
        initial_state = jax.vmap(hmc.init, in_axes=(0))(initial_state)
        hmc_kernel = jax.jit(hmc.step)

        #Run the inference loop
        num_samples = mcmc_kwargs.get("num_samples", 2000)
        states = self._blackjax_inference_loop_multiple_chains(
            key, hmc_kernel, initial_state, num_samples, num_chains
        )
        samples = states.position.squeeze()

        return samples

    def _nuts_blackjax(
        self,
        x : Array,
        key : Array,
        initial_state : Array,
        mcmc_kwargs: Optional[dict] = nuts_blackjax_kwargs_default

    ):
        """
        Perform MCMC sampling using the No-U-Turn Sampler (NUTS) in Blackjax.

        Parameters
        ----------
        x : Array
            The data used to condition the posterior.
        key : Array
            The random key used to generate the samples.
        initial_state : Array
            The initial state of the MCMC sampler.
        mcmc_kwargs: dict
            The keyword arguments for the MCMC method. (Default: nuts_blackjax_kwargs_default)

        Returns
        -------
        samples : Array
            The samples from the posterior.
        """
        def log_prob(theta):
            logprob = self.unnormalized_log_prob(theta, x)
            return jnp.sum(logprob)
        
        #Fetch the hyperparameters to create the kernel
        step_size = mcmc_kwargs.get("step_size", 1e-3)
        inverse_mass_matrix = mcmc_kwargs.get("inverse_mass_matrix", jnp.ones(10))
        nuts = blackjax.nuts(log_prob, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix)

        #Initialize the state and the kernel
        num_chains = mcmc_kwargs.get("num_chains", 1)
        initial_state = jnp.expand_dims(initial_state, axis=1)
        initial_state = jax.vmap(nuts.init, in_axes=(0))(initial_state)
        nuts_kernel = jax.jit(nuts.step)

        #Run the inference loop
        num_samples = mcmc_kwargs.get("num_samples", 2000)
        states = self._blackjax_inference_loop_multiple_chains(
            key, nuts_kernel, initial_state, num_samples, num_chains
        )
        samples = states.position.squeeze()

        return samples

    def _mala_flowmc(
        self,
    ):
        pass


    def set_default_x(self, x: Array):
        """
        Set the default data for the posterior.
        """
        self.x = x

    def set_prior(self, prior_distr: dist.Distribution):
        """
        Set the prior distribution for the parameters.
        """
        self.prior_distr = prior_distr

    def set_mcmc_method(self, mcmc_method: str):
        """
        Set the MCMC method to use.
        """
        assert mcmc_method in ["nuts_numpyro", "hmc_numpyro", "nuts_blackjax", "hmc_blackjax"], f"Invalid MCMC method: {mcmc_method}. Implemented methods are: ['nuts_numpyro', 'hmc_numpyro', 'nuts_blackjax', 'hmc_blackjax']."
        self.mcmc_method = mcmc_method

    def set_mcmc_kwargs(self, mcmc_kwargs: dict):
        """
        Set the keyword arguments for the MCMC method.
        """
        self.mcmc_kwargs = mcmc_kwargs
