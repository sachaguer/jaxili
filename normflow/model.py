import haiku as hk
from flax import linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

class AffineCoupling(hk.Module):
    """
    Affine Coupling layer
    """
    def __init__(self, y, *args, layers=[128, 128], activation=jax.nn.silu, **kwargs):
        """
        Constructor
        Parameters
        ----------

        y: jnp.Array
            Conditionning variable
        layers: int list
            List of hidden layers size
        activation: callable
            Activation function
        """
        super().__init__(*args, **kwargs)
        self.y = y
        self.layers = layers
        self.activation = activation

    def __call__(self, x, output_units, **condition_args):
        net = jnp.concatenate([x, self.y], axis=-1) #concatenate the input and the conditionning variable ??
        for i, layer_size in enumerate(self.layers):
            net = self.activation(hk.Linear(layer_size, name="layer%d" % i)(net))

        shifter = tfb.Shift(hk.Linear(output_units)(net))
        scaler = tfb.Scale(jnp.clip(jnp.exp(hk.Linear(output_units)(net)), 1e-2, 1e2))
        return tfb.Chain([shifter, scaler])
    
class ConditionalRealNVP(hk.Module):
    """
    Conditional Real NVP

    A normalizing flow using specified bijector functions. (https://arxiv.org/abs/1605.08803)
    """
    def __init__(self, d, *args, n_layers=3, bijector_fn=AffineCoupling, **kwargs):
        """
        Constructor

        Parameters
        ----------

        d: int
            Dimension of the input
        n_layers: int
            Number of layers
        bijector_fn: tfb.Bijector
            Bijector function
        """ 
        super().__init__(*args, **kwargs)
        self.d = d
        self.n_layers = n_layers
        self.bijector_fn = bijector_fn

    def __call__(self, y):
        """
        Create a Conditional Real NVP with self.n_layers layers with input dimension self.d.

        Parameters
        ----------
        y: jnp.Array
            Conditionning variable

        Output
        ------
        nvp : tfd.TransformedDistribution
            Normalizing Flow implemented as a Conditional Real NVP.
        """
        chain = tfb.Chain(
            [
                tfb.Permute(jnp.arange(self.d)[::-1])(
                    tfb.RealNVP(
                        self.d// 2, bijector_fn=self.bijector_fn(y, name="b%d" % i)
                    )
                )
                for i in range(self.n_layers)
            ]
        )

        nvp = tfd.TransformedDistribution(
            tfd.MultivariateNormalDiag(0.5*jnp.ones(self.d), 0.05 * jnp.ones(self.d)),
            bijector=chain
        )

        return nvp


class MixtureDensityNetwork(nn.Module):
    n_data : int #Dimension of data vector
    n_components : int #number of mixture components
    layers : list #list of hidden layers size
    activation : callable #activation function

    @nn.compact
    def __call__(self, x):
        for size in self.layers:
            x = self.activation(nn.Dense(size)(x))
        final_size = self.n_components * (1 + self.n_data + self.n_data*(self.n_data+1)//2)
        x = nn.Dense(final_size)(x)
        print(x.shape)
        logits = x[..., :self.n_components]
        locs = x[..., self.n_components:self.n_components*(self.n_data+1)]
        scale_tril = x[..., self.n_components*(self.n_data+1):]

        distribution = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=jnp.reshape(locs, (-1, self.n_components, self.n_data)),
                scale_tril=tfp.math.fill_triangular(jnp.reshape(scale_tril, (-1, self.n_components, self.n_data*(self.n_data+1)//2)))
            )
        )

        return distribution

