import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_probability as tfp

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

class MLPCompressor(hk.Module):
    """
    MLP Compressor

    Defines a MLP compressor to send the summary statistic to the same dimension than the parameters
    """

    def __init__(self, dim, *args, layers=[128, 128], activation=jax.nn.silu, **kwargs):
        """
        Constructor
        
        Parameters
        ----------

        dim: int
            Dimension of the output
        layers: int list
            List of hidden layers size
        activation: callable
            Activation function
        """

        super().__init__(*args, **kwargs)
        self.dim = dim
        self.layers = layers
        self.activation = activation

    def __call__(self, x, output_units, **condition_args):

        for i, layer_size in enumerate(self.layers):
            x = self.activation(hk.Linear(layer_size, name="layer%d" % i)(x))

        return hk.Linear(output_units)(x)