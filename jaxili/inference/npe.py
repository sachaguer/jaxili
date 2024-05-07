from typing import Any, Callable, Dict, Optional, Union
from jaxtyping import Array, Float, PyTree

from jaxili.utils import (
    check_density_estimator,
    validate_theta_x
)

import jax_dataloader as jdl

class NPE():

    def __init__(
            self,
            density_estimator: Union[str, Callable]='maf',
            logging_level: Union[int, str] = "WARNING",
            show_progress_bar: bool = True
    ):
        """
        Base class for Neural Posterior Estimation (NPE) methods.

        Parameters
        ----------
        density_estimator : Union[str, Callable], optional
            Density estimator to use. Options are 'maf' (Masked Autoregressive Flow, default), 'mdn' (Mixture Density Network) and 'realnvp').
        
        """
        check_density_estimator(density_estimator)

        self.density_estimator = density_estimator #Have to build the network here...

    
    def append_simulations(
            self,
            theta: Array,
            x: Array,        
    ):
        """
        Store parameters and simulation outputs to use them for later training.

        Data is stored in a Dataset object from `jax-dataloader`

        Parameter
        ---------
        theta : Array
            Parameters of the simulations.
        x : Array
            Simulation outputs.
        """

        theta, x = validate_theta_x(theta, x)

        jdl.Dataset

