import haiku as hk
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

class MixtureDensityNetwork(hk.Module):
    """
    Mixture Density Network

    A neural network that outputs the parameters of a Gaussian Mixture Model.
    """
    def __init__(self, n_components, *args, layers=[128, 128], activation=jax.nn.silu, **kwargs):
        """
        Constructor

        Parameters
        ----------

        n_components: int
            Number of components of the MDN
        layers: int list
            List of hidden layers size
        activation: callable
            Activation function
        """
        super().__init__(*args, **kwargs)
        self.d = d
        self.layers = layers
        self.activation = activation

    def __call__(self, x):
        """
        Create a Mixture Density Network with self.n_layers layers with input dimension self.d.

        Parameters
        ----------
        x: jnp.Array
            Input

        Output
        ------
        mdn : tfd.MixtureSameFamily
            Mixture Density Network
        """
        for i, layer_size in enumerate(self.layers):
            x = self.activation(hk.Linear(layer_size, name="layer%d" % i)(x))

        locs = hk.Linear(self.d)(x)
        scales = jnp.clip(jnp.exp(hk.Linear(self.d)(x)), 1e-2, 1e2)
        weights = jax.nn.softmax(hk.Linear(self.d)(x))

        mdn = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=weights),
            components_distribution=tfd.MultivariateNormalDiag(loc=locs, scale_diag=scales)
        )

        return mdn

