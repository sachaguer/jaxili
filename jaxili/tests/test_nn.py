import jax
import jax.numpy as jnp

import numpy as np
import numpy.testing as npt

from jaxili.model import ConditionalMAF, ConditionalRealNVP, MixtureDensityNetwork

def test_conditional_maf():
    n_in = 3
    n_cond = 3
    n_layers = [3, 4]
    use_reverse = [True, False]

    for layer, reverse in zip(n_layers, use_reverse):
        maf = ConditionalMAF(n_in, n_cond, layer, [128, 128], use_reverse=reverse, seed=42, activation='silu')
        x = jnp.array(np.random.randn(10, n_in))
        cond = jnp.array(np.random.randn(10, n_cond))
        params = maf.init(jax.random.PRNGKey(0), x, cond)
        log_prob = maf.apply(params, x, cond, method='log_prob')
        assert log_prob.shape == (10,), f"The shape of the output of log_prob method is wrong."

        #test the forward and reverse modes
        u, log_det = maf.apply(params, x, cond)
        x_reconstructed, log_det_reconstructed = maf.apply(params, u, cond, method='backward')

        npt.assert_allclose(x, x_reconstructed, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(log_det, -log_det_reconstructed, rtol=1e-5, atol=1e-5)

def test_conditional_realnvp():
    pass

def test_mixture_density_network():
    pass
