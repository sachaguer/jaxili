"""
Inference.

This module contains the inference classes for the JaxILI library. JaxILI implements classes for Neural Posterior and Likelihood Estimations.
"""

from jaxili.inference.nle import NLE
from jaxili.inference.npe import NPE
from jaxili.inference.nre import NRE

__all__ = ["NPE", "NLE", "NRE"]
