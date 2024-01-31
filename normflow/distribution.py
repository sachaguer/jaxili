import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
import normflow.model as nfn

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
        print(distr)
        return distr.log_prob(theta)
    
    def prob(self, params, theta, y):
        distr = self.compute_distribution(params, y)
        return distr.prob(theta)
    
    def sample(self, params, y, n_samples, key):
        distr = self.compute_distribution(params, y)
        print(distr)
        return distr.sample(n_samples, seed=key)