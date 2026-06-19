"""Model.

This module contains classes to implement normalizing flows using neural networks.

"""

from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Optional

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability as tfp
from flax import linen as nn
import flax.nnx as nnx
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array

tfp = tfp.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions


class NDENetwork(nnx.Module):
    """
    Base class for a Normalizing Flow.

    A Normalizing Flow parent class to implement normalizing flows using neural networks.
    """

    @abstractmethod
    def log_prob(self, x, y=None, **kwargs):
        """
        Log probability of the data point x conditioned by y.

        Parameters
        ----------
        x : jnp.Array
            Data point.
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        jnp.Array
            Log probability of the data point given y.
        """
        raise NotImplementedError(
            "log_prob method not implemented in your child class of NDENetwork"
        )

    @abstractmethod
    def sample(self, y, num_samples, key):
        """
        Sample from the distribution conditioned by y.

        Parameters
        ----------
        y : jnp.Array
            Conditionning variable.
        num_samples : int
            Number of samples.
        key : jnp.Array
            Random key.

        Returns
        -------
        jnp.Array
            num_samples samples from the distribution.
        """
        raise NotImplementedError(
            "sample method not implemented in tour child class of NDENetwork"
        )


class Compressor_w_NDE(NDENetwork):
    """
    Base class to create a normalizing flow with a compression of the conditionning variable.

    A parent class to implement a compressor followed by a normalizing flow. This is useful to perform Implicit Likelihood Inference in large dimensions where compression is required and can sometimes be done with a normalizing flow.
    """

    @abstractmethod
    def compress(self, x):
        """
        Compress the data point x using the compressor.

        Parameters
        ----------
        x : jnp.Array
            Data point.

        Returns
        -------
        jnp.Array
            Compressed data point.
        """
        raise NotImplementedError(
            "compress method not implemented in your child class of Compressor_w_NDE"
        )

    @abstractmethod
    def log_prob(self, x, y=None, **kwargs):
        """
        Log probability of the data point x conditioned by y.

        Parameters
        ----------
        x : jnp.Array
            Data point.
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        jnp.Array
            Log probability of the data point conditioned by y.
        """
        raise NotImplementedError(
            "log_prob method not implemented in your child class of Compressor_w_NDE"
        )

    @abstractmethod
    def log_prob_from_compressed(self, z, y=None, **kwargs):
        """
        Log probability of the data point z conditioned by y. z has been previously compressed.

        Parameters
        ----------
        z : jnp.Array
            Compressed data point.
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        jnp.Array
            Log probability of the data point conditioned by y.
        """
        raise NotImplementedError(
            "log_prob_from_compressed method not implemented in your child class of Compressor_w_NDE"
        )

    @abstractmethod
    def sample(self, y, num_samples, key):
        """
        Sample from the distribution conditioned by y.

        Parameters
        ----------
        y : jnp.Array
            Conditionning variable.
        num_samples : int
            Number of samples.
        key : jnp.Array
            Random key.

        Returns
        -------
        jnp.Array
            num_samples samples from the distribution.
        """
        raise NotImplementedError(
            "sample method not implemented in your child class of Compressor_w_NDE"
        )


class MixtureDensityNetwork(NDENetwork):
    """
    Base class for a Mixture Density Network.

    A Mixture of Gaussian Density modeled using neural networks. The weights of each gaussian component, the mean and the covariance are learned by the network.

    Parameters
    ----------
    n_int : int
        Dimensionality of the learned distribution.
    n_cond : int
        Dimensionality of the conditioning variable.
    n_components : int
        Number of Gaussian components in the mixture.
    layers : list[int]
        Size of the intermediate layers in the neural network.
    activation : Callable
        Activation function.
    rngs : nnx.Rngs
        Random seed of the network parameters.
    """

    def __init__(
        self,
        n_in: int,
        n_cond: int,
        n_components: int,
        layers: list[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.n_in = n_in
        self.n_cond = n_cond
        self.n_components = n_components
        self.hidden_size = layers
        self.activation = activation
        rngs = rngs

        self.final_size = self.n_components * (
            1 + self.n_in + self.n_in * (self.n_in + 1) // 2
        )

        self.layers = []  # List of Linear layers in the network
        if len(self.hidden_size) > 0:
            self.layers.append(
                nnx.Linear(n_cond, self.hidden_size[0], rngs=rngs)
            )  # Append first layer
            for in_features, out_features in zip(
                self.hidden_size[:-1], self.hidden_size[1:]
            ):  # Append hidden layers
                self.layers.append(nnx.Linear(in_features, out_features, rngs=rngs))
            self.layers.append(
                nnx.Linear(self.hidden_size[-1], self.final_size, rngs=rngs)
            )  # Append final layer
        else:
            self.layers.append(nnx.Linear(self.n_cond, self.final_size, rngs=rngs))

    def __call__(self, y, **kwargs):
        """
        Build a bijector that tranforms a multivariate Gaussian distribution into a Mixture of Gaussian distribution using a neural network.

        The weights, means and covariances are obtained from a conditioned variable y.

        Parameters
        ----------
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        tfd.Distribution
            Mixture of Gaussian distribution.
        """
        for layer in self.layers[:-1]:
            y = self.activation(layer(y))
        y = self.layers[-1](y)
        logits = jax.nn.log_softmax(y[..., : self.n_components])
        locs = y[..., self.n_components : self.n_components * (self.n_in + 1)]
        scale_tril = y[..., self.n_components * (self.n_in + 1) :]

        distribution = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(logits=logits),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=jnp.reshape(locs, (-1, self.n_components, self.n_in)),
                scale_tril=tfp.math.fill_triangular(
                    jnp.reshape(
                        scale_tril,
                        (-1, self.n_components, self.n_in * (self.n_in + 1) // 2),
                    )
                ),
            ),
        )

        return distribution

    def log_prob(self, x, y, **kwargs):
        """
        Return the log probability of the data point x conditioned by y.

        Parameters
        ----------
        x : jnp.Array
            Data point.
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        jnp.Array
            Log probability of the data point.
        """
        distribution = self.__call__(y, **kwargs)
        return distribution.log_prob(x)

    def sample(self, y, num_samples, key, **kwargs):
        """
        Sample from the distribution conditioned by y.

        Parameters
        ----------
        y : jnp.Array
            Conditionning variable.
        num_samples : int
            Number of samples.
        key : jnp.Array
            Random key.

        Returns
        -------
        jnp.Array
            num_samples samples from the distribution
        """
        if y.ndim == 1:
            y = y[None, :]
        distribution = self.__call__(y, **kwargs)
        return distribution.sample(sample_shape=num_samples, seed=key).squeeze()


class AffineCoupling(nnx.Module):
    """
    Base class for an Affine Coupling layer for RealNVP.

    Parameters
    ----------
    y : Any
        Conditionning variable.
    layers : list
        List of hidden layers size.
    activation : Callable
        Activation function.
    """

    def __init__(
        self,
        input_size: int,
        cond_size: int,
        output_size: int,
        layers: list,
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.input_size = input_size
        self.cond_size = cond_size
        self.output_size = output_size
        self.hidden_size = layers
        self.layers = [self.input_size + self.cond_size] + layers
        self.activation = activation

        self.linear_layers = []
        for in_features, out_features in zip(self.layers[:-1], self.layers[1:]):
            self.linear_layers.append(nnx.Linear(in_features, out_features, rngs=rngs))
        self.shift_layer = nnx.Linear(self.layers[-1], self.output_size, rngs=rngs)
        self.scale_layer = nnx.Linear(self.layers[-1], self.output_size, rngs=rngs)

    def __call__(self, x, output_units, y):
        """
        Build the bijector using tensorflow_probability where the scale and the shift are learned by a neural network.

        Parameters
        ----------
        x : jnp.Array
            Data point.
        output_units : int
            Argument not used in the network. Just for convenience with Tensorflow Probability format.
        y : jnp.Array
            Conditionning data vector.

        Returns
        -------
        tfb.Chain
            Bijector transforming a multidimensional Gaussian to a more complex distribution.
        """
        x = jnp.concatenate([x, y], axis=-1)
        for layer in self.linear_layers:
            x = self.activation(layer(x))

        # Shift and Scale parameters
        shift = self.shift_layer(x)
        scale = nnx.softplus(self.scale_layer(x)) + 1e-3

        return tfb.Chain([tfb.Shift(shift), tfb.Scale(scale)])


class ConditionalRealNVP(NDENetwork):
    """
    Base class for a Conditional RealNVP.

    A Normalizing Flow using RealNVP with a conditionning variable.

    Parameters
    ----------
    n_in : int
        Dimension of the input.
    n_cond : int
        Dimension of the conditionning variable.
    n_layers : int
        Number of layers.
    layers : list[int]
        List of hidden layers size.
    activation : Callable
        Activation function.
    rng : nnx.Rngs
        Random key
    """

    def __init__(
        self,
        n_in: int,
        n_cond: int,
        n_layers: int,
        layers: list[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.n_in = n_in
        self.n_cond = n_cond
        self.n_layers = n_layers
        self.layers = layers
        self.activation = activation

        self.coupling_layers = []
        for _ in range(self.n_layers):
            self.coupling_layers.append(
                AffineCoupling(
                    self.n_in // 2,
                    self.n_cond,
                    self.n_in // 2,
                    self.layers,
                    self.activation,
                    rngs=rngs,
                )
            )

    def __call__(self, y):
        """
        Build the bijector using tensorflow_probability.

        Parameters
        ----------
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        tfd.Distributions
            Normalizing Flow transporting a multidimensional Gaussian to a more complex distribution.
        """
        if self.n_in == 1:
            raise ValueError(
                "Flows can't be used to learn a one dimensional distribution. Consider using the `MixtureDensityNetwork`."
            )

        base_distribution = distrax.MultivariateNormalDiag(
            jnp.zeros(self.n_in), jnp.ones(self.n_in)
        )
        chain = distrax.Chain(
            [
                tfb.Permute(jnp.arange(self.n_in)[::-1])(
                    tfb.RealNVP(
                        self.n_in // 2,
                        bijector_fn=partial(self.coupling_layers[i], y=y),
                    )
                )
                for i in range(self.n_layers)
            ]
        )

        nvp = distrax.Transformed(base_distribution, bijector=chain)

        return nvp

    def sample(self, y, num_samples, key):
        """
        Sample from the distribution mapped by the real NVP.

        Parameters
        ----------
        y : jnp.Array
            Conditionning variable.
        num_samples : int
            Number of samples.
        key : jnp.Array
            Random key.

        Returns
        -------
        jnp.Array
            num_samples samples from the distribution.
        """
        y = y.squeeze()
        nvp = self.__call__(y)
        return nvp.sample(sample_shape=num_samples, seed=key)

    def log_prob(self, x, y):
        """
        Compute the log probability of the data point x conditioned by y from the normalizing flow.

        Parameters
        ----------
        x : jnp.Array
            Data point.
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        jnp.Array
            Log probability of the data point conditioned by y.
        """
        nvp = self.__call__(y)
        return nvp.log_prob(x)


# Reproduce implementation of MADE and MAFs from https://github.com/e-hulten/maf/blob/master/made.py


class MaskedLinear(nnx.Module):
    """
    Base class for a Masked Linear layer.

    Linear transformation with masked out elements.

    y = x.dot(mask*W.T)+b

    Parameters
    ----------
    n_out : int
        Output dimension.
    bias : bool
        Whether to include bias. Default True.
    mask : Any
        Mask to apply to the weights. Default None.
    rngs : nnx.Rngs
        Random key
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        mask: Any = None,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.mask = mask

        self.layer = nnx.Linear(self.n_in, self.n_out, use_bias=self.bias, rngs=rngs)

    def initialize_mask(self, mask: Any):
        """
        Set initialize mask.

        Parameters
        ----------
        mask : Any
            Boolean mask to apply to the weights.
        """
        self.mask = nnx.Variable(mask)

    def __call__(self, x):
        """
        Apply masked linear transformation.

        Parameters
        ----------
        x : jnp.Array
            Input vector.

        Returns
        -------
        jnp.Array
            Output vector.
        """
        w = nnx.state(self.layer)["kernel"].value
        x = jnp.dot(x, self.mask * w)
        if self.bias:
            b = nnx.state(self.layer)["bias"].value
            x += b
        return x


class ConditionalMADE(nnx.Module):
    """
    Base class for Conditional Masked Autoencoder Density Estimatior (MADE).

    MADE is a neural network that parameterizes the conditional distribution of a random variable using masked linear layers.

    Parameters
    ----------
    n_in : int
        Size of the input vector.
    hidden_dims : list[int]
        List of hidden dimensions.
    activation : Callable
        Activation function.
    n_cond : int
        Size of the conditionning variable. 0 if None.
    gaussian : bool
        Whether the output are mean and variance of a Gaussian conditional. Default True.
    random_order : bool
        Whether to use random order of the input for masking. Default False.
    seed : Optional[int]
        Random seed to label nodes. (Default: 42)
    """

    def __init__(
        self,
        n_in: int,  # Size of the input vector
        hidden_dims: list[int],  # List of hidden dimensions
        activation: Callable,  # Activation function
        n_cond: int = 0,  # Size of the conditionning variable. 0 if None
        gaussian: bool = True,  # Whether the outpur are mean and variance of a Gaussian conditional
        random_order: bool = False,  # Whether to use random order of the input for masking
        seed: Optional[int] = 42,  # Random seed to label nodes
        rngs: nnx.Rngs = nnx.Rngs(0),  # Random key
    ):
        self.n_in = n_in
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.n_cond = n_cond
        self.gaussian = gaussian
        self.random_order = random_order
        self.seed = seed

        np.random.seed(self.seed)  # Set the seed
        self.n_out = 2 * self.n_in if self.gaussian else self.n_in
        masks = {}
        mask_matrix = []
        layers = []

        dim_list = [self.n_in + self.n_cond, *self.hidden_dims, self.n_out]

        # Make layers and activation functions
        for in_features, out_features in zip(dim_list[:-2], dim_list[1:-1]):
            layers.append(MaskedLinear(in_features, out_features, rngs=rngs))
            layers.append(self.activation)
        # Last hidden layer to output layer
        layers.append(MaskedLinear(dim_list[-2], dim_list[-1], rngs=rngs))
        # Create masks
        self._create_masks(mask_matrix, masks, layers)
        # Create model
        self.layers = layers
        self.model = nnx.Sequential(*self.layers)

    def _create_masks(self, mask_matrix: list, masks: dict, layers: list):
        """Create masks for the model."""
        L = len(self.hidden_dims)  # Number of hidden layers
        D = self.n_in  # Number of input parameters
        C = self.n_cond  # Number of conditionning parameters

        # Whather to use random or natural order of the input
        masks[0] = np.random.permutation(D) if self.random_order else np.arange(D)

        # Set the connectivity number for the hidden layers
        for layer_index in range(L):
            low = masks[layer_index].min()
            size = self.hidden_dims[layer_index]
            if D > 1:
                masks[layer_index + 1] = np.random.randint(low, D - 1, size=size)
            else:
                masks[layer_index + 1] = np.zeros(size)
                # Order of the output layer is the same as the input layer
                masks[L + 1] = masks[0]

        # Create mask matric for input -> hidden_layer_1
        m = masks[0]
        m_next = masks[1]
        M = np.ones((len(m), len(m_next)))
        for j in range(len(m_next)):
            M[:, j] = (m <= m_next[j]).astype(int)
        M_cond = np.ones((C, len(m_next)))
        M = np.concatenate([M, M_cond], axis=0)
        mask_matrix.append(jnp.array(M))

        # Create mask matrix for hidden_layer_1 -> ... -> last_hidden_layers
        for i in range(1, len(masks) - 2):
            m = masks[i]
            m_next = masks[i + 1]
            # Initialise mask matrix
            M = np.zeros((len(m), len(m_next)))
            for j in range(len(m_next)):
                # Compare m_next[j] to each element of m
                M[:, j] = (m <= m_next[j]).astype(int)
            # append matrix to mask list
            mask_matrix.append(jnp.array(M))

        # Create mask matrix for last_hidden_layer -> output
        m = masks[len(masks) - 2]
        m_next = masks[len(masks) - 1]
        M = np.zeros((len(m), len(m_next)))
        for j in range(len(m)):
            # Compare m_next[j] to each element of m
            M[j, :] = (m[j] < m_next).astype(int)
        # append matrix to mask list
        mask_matrix.append(jnp.array(M))

        # If the output is Gaussian, double the number of output (mu, sigma)
        # Pairwise identical mask
        if self.gaussian:
            m = mask_matrix.pop(-1)
            mask_matrix.append(jnp.concatenate([m, m], axis=1))

        # Initialize the MaskedLinear layers with weights
        mask_iter = iter(mask_matrix)
        for module in layers:
            if isinstance(module, MaskedLinear):
                module.initialize_mask(next(mask_iter))

    def __call__(self, x, y=None):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : jnp.Array
            Input vector.
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        jnp.Array
            Output vector. If gaussian, the output is the mean and variance of the gaussian conditional. Otherwise, the output is the probability of the binary conditional.
        """
        if self.n_cond != 0:
            x = jnp.concatenate([x, y], axis=-1)
        if self.gaussian:
            return self.model(x)
        else:
            return jax.nn.sigmoid(self.model(x))


class MAFLayer(nnx.Module):
    """
    Base class for a Masked Autoregressive Flow layer.

    A single layer of a Masked Autoregressive Flow.

    Parameters
    ----------
    n_in : int
        Size of the input vector.
    n_cond : int
        Size of the conditionning variable.
    hidden_dims : list[int]
        List of hidden dimensions.
    reverse : bool
        Whether to reverse the order of the input.
    activation : Callable
        Activation function.
    seed : Optional[int]
        Random seed to label nodes. (Default: 42)
    rngs : nnx.Rngs
        Random key
    """

    def __init__(
        self,
        n_in: int,
        n_cond: int,
        hidden_dims: list[int],
        reverse: bool,
        activation: Callable,
        seed: Optional[int] = 42,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.n_in = n_in
        self.n_cond = n_cond
        self.hidden_dims = hidden_dims
        self.reverse = reverse
        self.activation = activation
        self.seed = seed

        self.conditional_made = ConditionalMADE(
            n_in=self.n_in,
            hidden_dims=self.hidden_dims,
            n_cond=self.n_cond,
            seed=self.seed,
            activation=self.activation,
            rngs=rngs,
        )

    def forward(self, x, y=None):
        """
        Forward pass of the model.

        Return vector u transformed by the flow and the log-determinant of the Jacobian of the flow.

        Parameters
        ----------
        x : jnp.Array
            Input vector.
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        jnp.Array
            Transformed vector.
        jnp.Array
            Log-determinant of the Jacobian.
        """
        out = self.__call__(x, y)
        mu, logp = jnp.split(out, 2, axis=-1)
        u = (x - mu) * jnp.exp(0.5 * logp)
        u = jnp.flip(u, axis=-1) if self.reverse else u
        log_det = 0.5 * jnp.sum(logp, axis=-1)
        return u, log_det

    def backward(self, u, y=None):
        """
        Backward pass of the model.

        Return vector x transformed by the inverse flow and the log-determinant of the Jacobian of the inverse flow.

        Parameters
        ----------
        u : jnp.Array
            Input vector.
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        jnp.Array
            Transformed vector.
        jnp.Array
            Log-determinant of the Jacobian.
        """
        u = jnp.flip(u, axis=-1) if self.reverse else u
        x = jnp.zeros_like(u)
        for dim in range(self.n_in):
            out = self.__call__(x, y)
            mu, logp = jnp.split(out, 2, axis=-1)
            mod_logp = jax.lax.clamp(-jnp.inf, -0.5 * logp, max=10.0)
            x = x.at[:, dim].set(mu[:, dim] + jnp.exp(mod_logp[:, dim]) * u[:, dim])
        log_det = jnp.sum(mod_logp, axis=-1)
        return x, log_det

    @nn.compact
    def __call__(self, x, y=None):
        """
        Forward pass of the model. Returns mean and variance of the gaussian conditionals.

        Parameters
        ----------
        x : jnp.Array
            Input vector.
        y : jnp.Array
            Conditionning variable.
        """
        x = self.conditional_made(x, y)
        return x


class ConditionalMAF(NDENetwork):
    """
    Base class of a Conditional Masked Autoregressive Flow.

    A Conditional Masked Autoregressive Flow to model the conditional distribution of a random variable. It is obtained by stacking `n_layers` MAF layers.

    Parameters
    ----------
    n_in : int
        Size of the input vector.
    n_cond : int
        Size of the conditionning variable.
    n_layers : int
        Number of layers (i.e. number of stacked MAFs).
    layers : list[int]
        List of hidden dimensions in each MAF.
    activation : Callable
        Activation function.
    use_reverse : bool
        Whether to reverse the order of the input between each MAF.
    seed : Optional[int]
        Random seed to label nodes. (Default: 42)
    rngs : nnx.Rngs
        Random key

    """

    def __init__(
        self,
        n_in: int,
        n_cond: int,
        n_layers: int,
        layers: list[int],
        activation: Callable,
        use_reverse: bool,
        seed: Optional[int] = 42,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.n_in = n_in
        self.n_cond = n_cond
        self.n_layers = n_layers
        self.layers = layers
        self.activation = activation
        self.use_reverse = use_reverse
        self.seed = seed

        # Sets the random seed
        np.random.seed(self.seed)
        if self.n_in == 1:
            raise ValueError(
                "Flows can't be used to learn a one dimensional distribution. Consider using the `MixtureDensityNetwork`."
            )
        layer_list = []
        for _ in range(self.n_layers):
            layer_list.append(
                MAFLayer(
                    n_in=self.n_in,
                    n_cond=self.n_cond,
                    hidden_dims=self.layers,
                    reverse=self.use_reverse,
                    seed=np.random.randint(0, 1000),
                    activation=self.activation,
                    rngs=rngs,
                )
            )
        self.layer_list = layer_list
        self.mean = nnx.Variable(jnp.zeros(self.n_in))
        self.cov = nnx.Variable(jnp.eye(self.n_in))

    def __call__(self, x, y=None):
        """
        Forward pass of the model.

        Returns mean and variance of the gaussian conditionals as well as the log-determinant of the Jacobian.

        Parameters
        ----------
        x : jnp.Array
            Input vector.
        y : jnp.Array
            Conditionning variable?=.

        Returns
        -------
        jnp.Array
            Transformed vector.
        jnp.Array
            Log-determinant of the Jacobian.
        """
        log_det_sum = jnp.zeros(x.shape[0])
        for layer in self.layer_list:
            x, log_det = layer.forward(x, y)
            log_det_sum += log_det
            # x = nn.BatchNorm(use_running_average=not train)(x)
        return x, log_det_sum

    def backward(self, u, y=None):
        """
        Backward pass of the model.

        Return vector x transformed by the inverse flow and the log-determinant of the Jacobian of the inverse flow.

        Parameters
        ----------
        u : jnp.Array
            Input vector.
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        x : jnp.Array
            Transformed vector.
        log_det_sum : jnp.Array
            Log-determinant of the Jacobian.
        """
        log_det_sum = jnp.zeros(u.shape[0])
        # backward pass
        for layer in reversed(self.layer_list):
            u, log_det = layer.backward(u, y)
            log_det_sum += log_det
        return u, log_det_sum

    def log_prob(self, x, y=None):
        """
        Compute the log-probability conditionned on some conditionning variable.

        Parameters
        ----------
        x : jnp.Array
            Input vector.
        y : jnp.Array
            Conditionning variable.

        Returns
        -------
        jnp.Array
            Log probability of the data point.
        """
        u, log_det_sum = self.__call__(x, y)
        log_pdf = multivariate_normal.logpdf(u, self.mean, self.cov)
        return log_pdf + log_det_sum

    def sample(self, y=None, num_samples=1, key=None):
        """
        Sample from the distribution emulated by the neural network.

        Parameters
        ----------
        y : jnp.Array
            Conditionning variable.
        num_samples : int
            Number of samples.
        key : jnp.Array
            Random key.

        Returns
        -------
        jnp.Array
            Samples from the distribution.
        """
        u = jax.random.multivariate_normal(
            key, self.mean, self.cov, shape=(num_samples,)
        )
        if y is not None:
            y = y * jnp.ones((num_samples, 1))
        x, _ = self.backward(u, y)
        return x


class NDE_Compressor(Compressor_w_NDE):
    """
    Base class for a normalizing flow with a compressor.

    WARNING: This class will likely be removed in the future as it is obsolete.
    A general class to implement a compressor followed by a normalizing flow implementing standard methods to compute the log-probability of the target distribution or sample from it.
    """

    compressor: nn.Module  # Compressor network
    nde: NDENetwork  # Normalizing Flow or Mixture Density network
    compressor_hparams: dict  # Hyperparameters of the Neural Density Estimator
    nde_hparams: dict  # Hyperparameters of the compressor

    def setup(self):
        """Set the compressor and the normalizing flow."""
        # Create models for the compressor and the NDE
        self.compressor_nn = self.compressor(**self.compressor_hparams)
        self.nde_nn = self.nde(**self.nde_hparams)

    def __call__(self, x, y, model="NPE"):
        """
        Perform a forward pass in the network and returns the log-probability of x given y.

        Parameters
        ----------
        x : jnp.Array
            Data point
        y : jnp.Array
            Conditionning variable

        Returns
        -------
        jnp.Array
            Log probability of the parameters y
        """
        assert model in ["NPE", "NLE"], "Model should be either 'NPE' or 'NLE'."
        if model == "NPE":
            z = self.compressor_nn(y)
            return self.nde_nn.log_prob(x, z)
        else:
            z = self.compressor_nn(x)
            return self.nde_nn.log_prob(z, y)

    def log_prob(self, x, y, model="NPE"):
        """
        Return the log-probability of the parameters y conditioned by the data point x.

        Parameters
        ----------
        x : jnp.Array
            Data point
        y : jnp.Array
            Conditionning variable

        Returns
        -------
        jnp.Array
            Log probability of the parameters y
        """
        return self.__call__(x, y, model)

    def log_prob_compressed(self, z, y, model="NPE"):
        """
        Return the log-probability of the compressed data z conditioned by the parameters y (if NPE).

        Parameters
        ----------
        z : jnp.Array
            Compressed data point
        y : jnp.Array
            Conditionning variable

        Returns
        -------
        jnp.Array
            Log probability of the parameters y
        """
        assert model in ["NPE", "NLE"], "Model should be either 'NPE' or 'NLE'."
        if model == "NPE":
            return self.nde_nn.log_prob(y, z)
        else:
            return self.nde_nn.log_prob(z, y)

    def sample(self, y, num_samples, key, model="NPE"):
        """
        Sample from the distribution conditioned by y.

        Parameters
        ----------
        y : jnp.Array
            Conditionning variable
        num_samples : int
            Number of samples
        key : jnp.Array
            Random key

        Returns
        -------
        jnp.Array
            num_samples samples from the distribution
        """
        assert model in ["NPE", "NLE"], "Model should be either 'NPE' or 'NLE'."
        if model == "NPE":
            z = self.compressor_nn(y)
            return self.nde_nn.sample(z, num_samples, key)
        else:
            return self.nde_nn.sample(y, num_samples, key)


class NDE_w_Standardization(NDENetwork):
    """
    Base class to implement normalizing flow with a standardization step.

    This class creates an NDE network where the input data is first standardized.
    It takes in input a neural density estimator, an embedding net and a transformation.
    The embedding net is used to embed the data point in a latent space where the NDE is applied. It allows to compress the data to lower dimensional space.
    The transformation is used to transform to standardize the variable learned by the normalizing flow for stability purpose.

    Parameters
    ----------
    nde : NDENetwork
        Neural Density Estimator used.
    embedding_net : nnx.Module
        Embedding net used for compression.
    transformation : distrax.Bijector
        Transformation used on the points in inference space before training the NDE.
    """

    def __init__(
        self,
        nde: NDENetwork,
        embedding_net: nnx.Module,
        shift_transformation: Array,
        scale_transformation: Array,
    ):
        self.nde = nde
        self.embedding_net = embedding_net
        self.shift_transformation = nnx.Variable(shift_transformation)
        self.scale_transformation = nnx.Variable(scale_transformation)

    def __call__(self, x, y, model="NPE"):
        """
        Return the log-probability of x given y for NPE and y given x for NLE.

        Parameters
        ----------
        x : jnp.Array
            Parameters
        y : jnp.Array
            Conditionning variable
        model : str
            Whether the network is trained using NPE or NLE. Default: NPE.

        Returns
        -------
        jnp.Array
            Log probability of the parameters y
        """
        transformation = self.get_transformation()
        assert model in ["NPE", "NLE"], "Model should be either 'NPE' or 'NLE'."
        if model == "NLE":
            x, y = y, x  # Learn the distribution p(y|x). Exchange the two.
        x, logprob_std = transformation.inverse_and_log_det(x)
        logprob_std = jnp.sum(logprob_std, axis=-1)
        z = self.embedding_net(y)
        log_prob = self.nde.log_prob(x, z)
        return log_prob + logprob_std

    def get_transformation(self):
        """Create the transformation object."""
        return distrax.ScalarAffine(
            shift=self.shift_transformation.value, scale=self.scale_transformation.value
        )

    def standardize(self, x):
        """Standardize the data point x."""
        transformation = self.get_transformation()
        return transformation.inverse(x)

    def unstandardize(self, x):
        """Unstandardize the data point x."""
        transformation = self.get_transformation()
        return transformation.forward(x)

    def embedding(self, x):
        """Embed the data point x."""
        return self.embedding_net(x)

    def log_prob(self, x, y=None, model="NPE"):
        """Return the log probability of the data point x conditioned by y."""
        return self.__call__(x, y, model)

    def sample(self, y, num_samples, key, model="NPE"):
        """Sample from the distribution conditioned by y."""
        assert model in ["NPE", "NLE"], "Model should be either 'NPE' or 'NLE'."
        if model == "NPE":
            z = self.embedding_net(y)
            samples = self.nde.sample(z, num_samples, key)
        else:
            samples = self.nde.sample(y, num_samples, key)
        transformation = self.get_transformation()
        samples = transformation.forward(samples)

        return samples
