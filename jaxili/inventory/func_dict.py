"""
Function dictionaries for JaxILI.

This script contains static dictionaries that map function names to their respective functions.
Those dictionaries are used to load neural network models using Callable as hyperparameters (e.g. activation functions)
from a JSON file where the name of the function is stored.
"""

import inspect

import jax
import jaxlib

from jaxili.loss import loss_nll_nle, loss_nll_npe, loss_mse, loss_mae
from jaxili.model import ConditionalMAF, ConditionalRealNVP, MixtureDensityNetwork


# Define a dictionary containing jax activation functions.
def is_jax_nn_activation_function(obj):
    """Return if an object is an activation function from jax.nn."""
    return (
        isinstance(obj, jaxlib.xla_extension.PjitFunction)
        or isinstance(obj, jax._src.custom_derivatives.custom_jvp)
        or inspect.isfunction(obj)
    )


jax_nn_dict = {
    name: func
    for name, func in inspect.getmembers(jax.nn, is_jax_nn_activation_function)
}

if "__getattr__" in jax_nn_dict.keys():
    jax_nn_dict.pop("__getattr__")

# Define a dictionary containing Neural Network from jaxili

jaxili_nn_dict = {
    "ConditionalMAF": ConditionalMAF,
    "MixtureDensityNetwork": MixtureDensityNetwork,
    "ConditionalRealNVP": ConditionalRealNVP,
}

# Define a dictionary containing the loss functions introduced in jaxili
jaxili_loss_dict = {
    "loss_nll_npe": loss_nll_npe,
    "loss_nll_nle": loss_nll_nle,
    "loss_mse": loss_mse,
    "loss_mae": loss_mae,
}
