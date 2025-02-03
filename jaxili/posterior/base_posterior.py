"""
Base Posterior.

This module contains the base class for Neural Posteriors. Classes used to sample in NPE and NLE will inherit from this class.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional

from jaxtyping import Array

from jaxili.model import NDENetwork
from jaxili.train import TrainState


class NeuralPosterior:
    r"""
    Posterior $p(\theta|x)$ with `log_prob()` and `sample()` methods.

    The class wraps the trained neural network such that one can directly evaluate the log-probability and sample from the posterior.
    """

    def __init__(
        self,
        model: NDENetwork,
        state: TrainState,
        verbose: bool = False,
        x: Optional[Array] = None,
    ):
        """
        Initialize the Neural Posterior.

        Parameters
        ----------
        model : NDENetwork
            The neural network used to generate the posterior.
        state : dict
            The state of the neural network.
        verbose : bool
            Whether to print information. (Default: False)
        """
        self.model = model
        self.state = state
        self.verbose = verbose
        self.x = x

    @abstractmethod
    def sample(
        self,
        num_samples: int,
        key: Array,
        x: Array,
        mcmc_method: Optional[str] = None,
        mcmc_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Define abstract method to sample from the posterior. The sampling method depends on the methodology used."""
        pass

    @abstractmethod
    def unnormalized_log_prob(
        self,
        theta: Array,
    ):
        """Define abstract method to compute the unnormalized log-probability of a given parameter."""
        pass

    def set_default_x(self, x: Array):
        """Set the default data for the posterior."""
        self.x = x
