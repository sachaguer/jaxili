import distrax
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from jaxili.model import *
from jaxili.compressor import *


def test_conditional_maf():
    n_in = 3
    n_cond = 5
    n_layers = [3, 4]
    use_reverse = [True, False]

    for layer, reverse in zip(n_layers, use_reverse):
        maf = ConditionalMAF(
            n_in,
            n_cond,
            layer,
            [128, 128],
            use_reverse=reverse,
            seed=42,
            activation=jax.nn.silu,
            rngs=nnx.Rngs(0)
        )
        x = jnp.array(np.random.randn(10, n_in))
        cond = jnp.array(np.random.randn(10, n_cond))
        log_prob = maf.log_prob(x, cond)
        assert log_prob.shape == (
            10,
        ), f"The shape of the output of log_prob method is wrong."

        # test the forward and reverse modes
        u, log_det = maf(x, cond)
        x_reconstructed, log_det_reconstructed = maf.backward(
            u, cond
        )

        npt.assert_allclose(x, x_reconstructed, rtol=1e-5, atol=1e-5)
        npt.assert_allclose(log_det, -log_det_reconstructed, rtol=1e-5, atol=1e-5)

        # Test the sampling
        samples = maf.sample(
            cond[0],
            num_samples=10_000,
            key=jax.random.PRNGKey(0),
        )
        assert samples.shape == (10_000, 3), f"The shape of the samples is wrong."

        # Test sampling with a different shape for the conditional
        samples = maf.sample(
            cond[0].reshape((-1, n_cond)),
            num_samples=10_000,
            key=jax.random.PRNGKey(0),
        )
        assert samples.shape == (10_000, 3), f"The shape of the samples is wrong."


def test_conditional_realnvp():
    n_in = 3
    n_cond = 5
    n_layers = 5
    layers = [50, 50, 50]
    activation = jax.nn.relu

    realnvp = ConditionalRealNVP(n_in, n_cond, n_layers, layers, activation, rngs=nnx.Rngs(0))

    x = jnp.array(np.random.randn(10, n_in))
    cond = jnp.array(np.random.randn(10, n_cond))

    log_prob = realnvp.log_prob(x, cond)
    assert log_prob.shape == (
        10,
    ), f"The shape of the output of log_prob method is wrong."

    # Test the sampling
    samples = realnvp.sample(
        cond[0], num_samples=10_000, key=jax.random.PRNGKey(0)
    )
    assert samples.shape == (10_000, n_in), f"The shape of the samples is wrong."

    # Test sampling with a different shape for the conditional
    samples = realnvp.sample(
        cond[0].reshape((-1, n_cond)),
        num_samples=10_000,
        key=jax.random.PRNGKey(0),
    )
    assert samples.shape == (10_000, n_in), f"The shape of the samples is wrong."


def test_mixture_density_network():
    n_in = 3
    n_cond = 5
    n_components = 5
    layers = [50, 50, 50]
    activation = jax.nn.relu

    mdn = MixtureDensityNetwork(n_in, n_cond, n_components, layers, activation, nnx.Rngs(0))

    x = jnp.array(np.random.randn(10, n_in))
    cond = jnp.array(np.random.randn(10, n_cond))

    log_prob = mdn.log_prob(x, cond)
    assert log_prob.shape == (
        10,
    ), f"The shape of the output of log_prob method is wrong."

    # Test the sampling
    samples = mdn.sample(
        cond[0], num_samples=10_000, key=jax.random.PRNGKey(0)
    )
    assert samples.shape == (10_000, n_in), f"The shape of the samples is wrong."

    # Test sampling with a different shape for the conditional
    samples = mdn.sample(
        cond[0].reshape((-1, n_cond)),
        num_samples=10_000,
        key=jax.random.PRNGKey(0),
    )
    assert samples.shape == (10_000, n_in), f"The shape of the samples is wrong."


def test_identity():
    identity = Identity()

    x = np.random.randn(10, 3)

    assert np.isclose(
        identity(x), x
    ).all(), "Identity function is not working."


def test_affine_transformation():
    n_dim = 2
    n_samples = 1000
    scale = 2
    shift = 1

    samples = jax.random.normal(jax.random.PRNGKey(0), shape=(n_samples, n_dim))
    shifted_samples = samples * scale + shift

    affine = distrax.ScalarAffine(scale=scale, shift=shift)

    test_forward = affine.forward(samples)
    test_inverse = affine.inverse(shifted_samples)

    npt.assert_allclose(test_forward, shifted_samples, rtol=1e-5, atol=1e-5)
    npt.assert_allclose(test_inverse, samples, rtol=1e-5, atol=1e-5)


def test_standardizer():
    n_dim = 3
    n_samples = 1000
    shift = np.random.randn(n_dim) * 2
    scale = np.random.randn(n_dim)

    samples = jax.random.normal(jax.random.PRNGKey(0), shape=(n_samples, n_dim))
    samples = samples * shift + scale

    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

    standardizer = Standardizer(mean, std)

    std_samples = (samples - mean) / std
    test_std_samples = standardizer(samples)

    npt.assert_allclose(std_samples, test_std_samples, rtol=1e-5, atol=1e-5)


def test_network_w_standardization():
    n_in = 2
    n_cond = 5
    n_layers = 3
    layers = [50, 50]
    activation = jax.nn.relu

    theta = jax.random.normal(jax.random.PRNGKey(0), shape=(10, n_in))
    x = jax.random.normal(jax.random.PRNGKey(0), shape=(10, n_cond))

    shift = np.random.randn(n_in) * 2
    scale = np.random.randn(n_in)

    shifted_theta = theta * scale + shift

    transformation = distrax.ScalarAffine(scale=scale, shift=shift)

    maf = ConditionalMAF(
        n_in=n_in,
        n_cond=n_cond,
        n_layers=n_layers,
        layers=layers,
        activation=activation,
        use_reverse=True,
        seed=42,
        rngs=nnx.Rngs(0)
    )

    net_w_standard = NDE_w_Standardization(
        nde=maf, embedding_net=Identity(), shift_transformation=transformation.shift, scale_transformation=transformation.scale
    )

    # Test the standardization
    test_theta = net_w_standard.standardize(shifted_theta)
    npt.assert_allclose(test_theta, theta, rtol=1e-5, atol=1e-5)

    # Test the unstandardization
    test_theta = net_w_standard.unstandardize(theta)
    npt.assert_allclose(test_theta, shifted_theta, rtol=1e-5, atol=1e-5)

    # Test the embedding
    test_embedding = net_w_standard.embedding(x)
    npt.assert_allclose(test_embedding, x, rtol=1e-5, atol=1e-5)

    # Test the log_prob
    log_prob = net_w_standard.log_prob(theta, x)
    assert log_prob.shape == (
        10,
    ), f"The shape of the output of log_prob method is wrong."

    # Test the sampling
    samples = net_w_standard.sample(
        x[0], num_samples=10_000, key=jax.random.PRNGKey(0)
    )
    assert samples.shape == (10_000, n_in), f"The shape of the samples is wrong."
