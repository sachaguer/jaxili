"""
Posterior.

This module contains classes to sample from the posterior and evaluate the log probability of the normalizing flows.
"""

from jaxili.posterior.base_posterior import NeuralPosterior
from jaxili.posterior.direct_posterior import DirectPosterior
from jaxili.posterior.mcmc_posterior import MCMCPosterior
