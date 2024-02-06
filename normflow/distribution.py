import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
import normflow.model as nfn
from functools import partial

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

class MixtureDensityDistribution(tfd.Distribution):
    def __init__(self, n_data, n_components, layers, activation, dtype=jnp.float32, reparameterization_type=None, validate_args=False, allow_nan_stats=True, name="MixtureDensityDistribution"):
        super().__init__(
            dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )
        self.n_data = n_data
        self.n_components = n_components
        self.layers = layers
        self.activation = activation

        self.md_model = self.build_network()

    def build_network(self):
        md_distribution = nfn.MixtureDensityNetwork(
            n_data=self.n_data,
            n_components=self.n_components,
            layers=self.layers,
            activation=self.activation,
        )
        return md_distribution
    
    def compute_distribution(self, params, y):
        return self.md_model.apply(params, y)

    def log_prob(self, params, theta, y):
        distr = self.compute_distribution(params, y)
        return distr.log_prob(theta)
    
    def prob(self, params, theta, y):
        distr = self.compute_distribution(params, y)
        return distr.prob(theta)
    
    def sample(self, params, y, n_samples, key):
        distr = self.compute_distribution(params, y)
        return distr.sample(n_samples, seed=key)
    

class ConditionalRealNVPDistribution(tfd.Distribution):
    def __init__(self, d, n_layers, bijector_fn, dtype=jnp.float32, reparameterization_type=None, validate_args=False, allow_nan_stats=True, name="CondRealNVPDistribution"):
        super().__init__(
            dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )
        self.d = d
        self.n_layers = n_layers
        self.bijector_fn = bijector_fn

        self.real_nvp = self.build_network()

    def build_network(self):
        return partial(
            nfn.ConditionalRealNVP,
            n_layers=self.n_layers,
            bijector_fn=self.bijector_fn
        )

    def compute_distribution(self, params, y):
        return self.real_nvp(self.d).apply(params, y)
    
    def log_prob(self, params, theta, y):
        distr = self.compute_distribution(params, y)
        return distr.log_prob(theta)
    
    def prob(self, params, theta, y):
        distr = self.compute_distribution(params, y)
        return distr.prob(theta)
    
    def sample(self, params, y, n_samples, key):
        distr = self.compute_distribution(params, y)
        return distr.sample(n_samples, seed=key)