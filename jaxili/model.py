from flax import linen as nn
import distrax
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
from typing import Any, Optional, Callable
from jax.scipy.stats import multivariate_normal
from functools import partial 

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

class NDENetwork(nn.Module):
    """
    A Normalizing Flow parent class to implement normalizing flows using
    neural networks.
    """
    def log_prob(self, x, y=None, **kwargs):
        """
        Log probability of the data point x conditioned by y.
        """
        raise NotImplementedError("log_prob method not implemented in your child class of NDENetwork")

    def sample(self, y, num_samples, key):
        """
        Sample from the distribution conditioned by y.
        """
        raise NotImplementedError("sample method not implemented in tour child class of NDENetwork")

class MixtureDensityNetwork(NDENetwork):
    """
    A Mixture of Gaussian Density modeled using neural networks.
    The weights of each gaussian component, the mean and the covariance are learned by the network.
    """
    n_data : int #Dimension of data vector
    n_components : int #number of mixture components
    layers : list #list of hidden layers size
    activation : Callable #activation function

    @nn.compact
    def __call__(self, y, **kwargs):
        """
        Builds a bijector that tranforms a multivariate Gaussian distribution
        into a Mixture of Gaussian distribution using a neural network.
        The weights, means and covariances are obtained from a conditioned variable y.

        Parameters
        ----------
        y : jnp.Array
            Conditionning variable

        Returns
        -------
        distribution : tfd.Distribution
            Mixture of Gaussian distribution
        """
        kernel_init = kwargs.get('kernel_init', nn.initializers.variance_scaling(scale=1., mode='fan_in', distribution='normal'))
        for size in self.layers:
            y = self.activation(nn.Dense(size, kernel_init=kernel_init)(y))
        final_size = self.n_components * (1 + self.n_data + self.n_data*(self.n_data+1)//2)
        y = nn.Dense(final_size, kernel_init=kernel_init)(y)
        logits = jax.nn.log_softmax(y[..., :self.n_components])
        locs = y[..., self.n_components:self.n_components*(self.n_data+1)]
        scale_tril = y[..., self.n_components*(self.n_data+1):]

        distribution = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(logits=logits),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=jnp.reshape(locs, (-1, self.n_components, self.n_data)),
                scale_tril=tfp.math.fill_triangular(jnp.reshape(scale_tril, (-1, self.n_components, self.n_data*(self.n_data+1)//2)))
            )
        )

        return distribution

    def log_prob(self, x, y, **kwargs):
        """
        Returns the log probability of the data point x conditioned by y.

        Parameters
        ----------
        x : jnp.Array
            Data point
        y : jnp.Array
            Conditionning variable

        Returns
        -------
        log_prob : jnp.Array
            Log probability of the data point
        """
        distribution = self.__call__(y, **kwargs)
        return distribution.log_prob(x)

    def sample(self, y, num_samples, key):
        """
        Samples from the distribution conditioned by y.

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
        samples : jnp.Array
            num_samples samples from the distribution
        """
        distribution = self.__call__(y)
        return distribution.sample(sample_shape=num_samples, seed=key).squeeze()

class AffineCoupling(nn.Module):
    y : Any #Conditionning variable
    layers : list #list of hidden layers size
    activation : callable #activation function

    @nn.compact
    def __call__(self, x, output_units, **kwargs):
        x = jnp.concatenate([x, self.y], axis=-1)
        for i, layer_size in enumerate(self.layers):
            x = self.activation(nn.Dense(layer_size, kernel_init=nn.initializers.truncated_normal(0.001))(x))
        
        #Shift and Scale parameters
        shift = nn.Dense(output_units, kernel_init=nn.initializers.truncated_normal(0.001))(x)
        scale = nn.softplus(nn.Dense(output_units, kernel_init=nn.initializers.truncated_normal(0.001))(x)) + 1e-3

        return tfb.Chain([tfb.Shift(shift), tfb.Scale(scale)])
    
class ConditionalRealNVP(NDENetwork):
    n_in : int #Dimension of the input
    n_layers : int #Number of layers
    layers : list[int] #list of hidden layers size
    activation : Callable #activation function

    @nn.compact
    def __call__(self, y, **kwargs):
        """
        Build the bijector using tensorflow_probability

        Parameters
        ----------
        y : jnp.Array
            Conditionning variable

        Returns
        -------
        nvp : tfd.Distributions
            Normalizing Flow transporting a multidimensional Gaussian
            to a more complex distribution.
        """
        bijector_fn  = partial(
            AffineCoupling,
            layers=self.layers,
            activation=self.activation
        )
        base_distribution = distrax.MultivariateNormalDiag(jnp.zeros(self.n_in), jnp.ones(self.n_in))
        chain = distrax.Chain(
            [
                tfb.Permute(jnp.arange(self.n_in)[::-1])(
                    tfb.RealNVP(
                        self.n_in//2, bijector_fn=bijector_fn(y, name="b%d" % i)
                    )
                )
            for i in range(self.n_layers)
            ]
        )

        nvp = distrax.Transformed(
            base_distribution,
            bijector=chain
        )

        return nvp
    
    def sample(self, y, num_samples, key, **kwargs):
        """
        Samples from the distribution mapped by the real NVP

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
        samples : jnp.Array
            num_samples samples from the distribution
        """
        nvp = self.__call__(y)
        return nvp.sample(sample_shape=num_samples, seed=key)

    def log_prob(self, x, y, **kwargs):
        """
        Computes the log probability of the data point x conditioned by y from the normalizing flow.

        Parameters
        ----------
        x : jnp.Array
            Data point
        y : jnp.Array
            Conditionning variable
        """
        nvp = self.__call__(y)
        return nvp.log_prob(x)


#Reproduce implementation of MADE and MAFs from https://github.com/e-hulten/maf/blob/master/made.py

class MaskedLinear(nn.Module): #Check if there is no issue when you jit a loss using such a network. Note: Mask will not change after initialisation.
    """
    Linear transformation with masked out elements.

    y = x.dot(mask*W.T)+b

    Parameters
    ----------
    n_out : int
        Output dimension
    bias : bool
        Whether to include bias. Default True.
    """
    n_out : int
    bias : bool = True
    mask: Any = None

    def initialize_mask(self, mask : Any):
        """Internal method to initialize mask"""
        self.mask = mask

    @nn.compact
    def __call__(self, x):
        """Apply masked linear transformation"""
        layer = nn.Dense(self.n_out, use_bias=self.bias, param_dtype=jnp.float64, kernel_init=nn.initializers.truncated_normal(0.01))
        is_initialized = self.has_variable('params', 'Dense_0')
        if is_initialized: 
            w = layer.variables['params']['kernel']
            b = layer.variables['params']['bias']
        else:
            return layer(x)
        return jnp.dot(x, self.mask*w)+b

class ConditionalMADE(nn.Module):
    n_in : int #Size of the input vector
    hidden_dims : list[int] #list of hidden dimensions
    activation : Callable #Activation function
    n_cond : int =0 #Size of the conditionning variable. 0 if None.
    gaussian : bool = True #whether the output are mean and variance of a Gaussian conditional
    random_order : bool = False #Whether to use random order of the input for masking
    seed : Optional[int] = None #Random seed to label nodes
    

    def setup(self):
        
        np.random.seed(self.seed)
        self.n_out = 2*self.n_in if self.gaussian else self.n_in
        masks = {}
        mask_matrix = []
        layers = []

        dim_list = [self.n_in+self.n_cond, *self.hidden_dims, self.n_out]

        #Make layers and activation functions
        for i in range(len(dim_list)-2):
            layers.append(MaskedLinear(dim_list[i+1]))
            layers.append(self.activation)
        #Last hidden layer to output layer
        layers.append(MaskedLinear(dim_list[-1]))
        #Create masks
        self._create_masks(mask_matrix, masks, layers)
        #Create model
        self.layers = layers
        self.model = nn.Sequential(self.layers)

    def _create_masks(self, mask_matrix: list, masks: dict, layers : list):
        """Create masks for the model"""
        L = len(self.hidden_dims) #Number of hidden layers
        D = self.n_in #Number of input parameters
        C = self.n_cond #Number of conditionning parameters

        #Whather to use random or natural order of the input
        masks[0] = np.random.permutation(D) if self.random_order else np.arange(D)

        #Set the connectivity number for the hidden layers
        for l in range(L):
            low = masks[l].min() #Get the lowest index in the previous layer
            size = self.hidden_dims[l] #The size of the current hidden layer
            masks[l+1] = np.random.randint(low, D-1, size=size)
        
        #Order of the output layer is the same as the input layer
        masks[L+1] = masks[0]

        #Create mask matric for input -> hidden_layer_1
        m = masks[0]
        m_next = masks[1]
        M = np.ones((len(m), len(m_next)))
        for j in range(len(m_next)):
            M[:, j] = (m <= m_next[j]).astype(int)
        M_cond = np.ones((C, len(m_next)))
        M = np.concatenate([M, M_cond], axis=0)
        mask_matrix.append(jnp.array(M))


        #Create mask matrix for hidden_layer_1 -> ... -> last_hidden_layers
        for i in range(1, len(masks)-2):
            m = masks[i]
            m_next = masks[i+1]
            #Initialise mask matrix
            M = np.zeros((len(m), len(m_next)))
            for j in range(len(m_next)):
                #Compare m_next[j] to each element of m
                M[:, j] = (m <= m_next[j]).astype(int)
            #append matrix to mask list
            mask_matrix.append(jnp.array(M))

        #Create mask matrix for last_hidden_layer -> output
        m = masks[len(masks)-2]
        m_next = masks[len(masks)-1]
        M = np.zeros((len(m), len(m_next)))
        for j in range(len(m)):
            #Compare m_next[j] to each element of m
            M[j, :] = (m[j] < m_next).astype(int)
        #append matrix to mask list
        mask_matrix.append(jnp.array(M))

        #If the output is Gaussian, double the number of output (mu, sigma)
        #Pairwise identical mask
        if self.gaussian:
            m = mask_matrix.pop(-1)
            mask_matrix.append(jnp.concatenate([m, m], axis=1))

        #Initialize the MaskedLinear layers with weights
        mask_iter = iter(mask_matrix)
        for module in layers:
            if isinstance(module, MaskedLinear):
                module.initialize_mask(next(mask_iter))

    def __call__(self, x, y=None):
        """
        Forward pass of the model

        Parameters
        ----------
        x : jnp.Array
            Input vector
        y : jnp.Array
            Conditionning variable
        """
        if self.n_cond != 0:
            x = jnp.concatenate([x, y], axis=-1)
        if self.gaussian:
            return self.model(x)
        else:
            return jax.nn.sigmoid(self.model(x))
        
class MAFLayer(nn.Module):
    n_in : int #Size of the input vector
    n_cond : int #Size of the conditionning variable
    hidden_dims : list[int] #list of hidden dimensions
    reverse : bool #Whether to reverse the order of the input
    activation : Callable #Activation function
    seed : Optional[int] = None #Random seed to label nodes

    def forward(self, x, y=None):
        out = self.__call__(x, y)
        mu, logp = jnp.split(out, 2, axis=-1)
        u = (x-mu)*jnp.exp(0.5*logp)
        u = jnp.flip(u, axis=-1) if self.reverse else u
        log_det = 0.5*jnp.sum(logp, axis=-1)
        return u, log_det
    
    def backward(self, u, y=None):
        u = jnp.flip(u, axis=-1) if self.reverse else u
        x = jnp.zeros_like(u)
        for dim in range(self.n_in):
            out = self.__call__(x,y)
            mu, logp = jnp.split(out, 2, axis=-1)
            mod_logp = jax.lax.clamp(-jnp.inf, -0.5*logp, max=10.)
            x = x.at[:,dim].set(mu[:,dim]+jnp.exp(mod_logp[:,dim])*u[:,dim])
        log_det = jnp.sum(mod_logp, axis=-1)
        return x, log_det
        

    @nn.compact
    def __call__(self, x, y=None):
        """
        Forward pass of the model. Returns mean and variance of the gaussian conditionals.

        Parameters
        ----------
        x : jnp.Array
            Input vector
        y : jnp.Array
            Conditionning variable
        """
        x = ConditionalMADE(n_in=self.n_in, hidden_dims=self.hidden_dims, n_cond=self.n_cond, seed=self.seed, activation=self.activation)(x, y)
        return x
        
class ConditionalMAF(NDENetwork):
    n_in : int #Size of the input vector
    n_cond : int #Size of the conditionning variable
    n_layers : int #Number of layers (i.e. number of stacked MADEs)
    layers : list[int] #list of hidden dimensionsin each MADE.
    activation : Callable #Activation function
    use_reverse : bool =True #Whether to reverse the order of the input between each MADE
    seed : Optional[int] = None #Random seed to label nodes
    

    def setup(self):
        np.random.seed(self.seed)
        layer_list = []
        for _ in range(self.n_layers):
            layer_list.append(
                MAFLayer(
                    n_in=self.n_in, n_cond=self.n_cond, hidden_dims=self.layers, reverse=self.use_reverse, seed=np.random.randint(0, 1000), activation=self.activation
                )
            )
        self.layer_list = layer_list
        self.mean = jnp.zeros(self.n_in)
        self.cov = jnp.eye(self.n_in)

    @nn.compact
    def __call__(self, x, y=None):
        """
        Forward pass of the model. Returns mean and variance of the gaussian conditionals
        as well as the log-determinant of the Jacobian

        Parameters
        ----------
        x : jnp.Array
            Input vector
        y : jnp.Array
            Conditionning variable
        """
        log_det_sum = jnp.zeros(x.shape[0])
        for layer in self.layer_list:
            x, log_det = layer.forward(x, y)
            log_det_sum += log_det
            #x = nn.BatchNorm(use_running_average=not train)(x)
        return x, log_det_sum
    
    def backward(self, u, y=None):
        log_det_sum = jnp.zeros(u.shape[0])
        #backward pass
        for layer in reversed(self.layer_list):
            u, log_det = layer.backward(u, y)
            log_det_sum += log_det
        return u, log_det_sum
    
    def log_prob(self, x, y=None):
        u, log_det_sum = self.__call__(x, y)
        log_pdf = multivariate_normal.logpdf(u, self.mean, self.cov)
        return log_pdf + log_det_sum
    
    def sample(self, y=None, num_samples=1, key=None):
        u = jax.random.multivariate_normal(key, self.mean, self.cov, shape=(num_samples,))
        if y is not None:
            y = y*jnp.ones((num_samples, 1))
        x, _ = self.backward(u, y)
        return x


        
