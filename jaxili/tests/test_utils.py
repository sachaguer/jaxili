import jax

from jaxili.utils import *


def test_check_density_estimator():

    check_density_estimator("maf")
    check_density_estimator("mdn")
    check_density_estimator("realnvp")

    try:
        check_density_estimator("blablabla")
    except ValueError:
        pass


def test_validate_theta_x():
    key = jax.random.PRNGKey(0)

    theta = jax.random.normal(key, (100, 2))
    x = jax.random.normal(key, (100, 2))

    validate_theta_x(theta, x)
    _, _, batch_size = validate_theta_x(np.array(theta), np.array(x))

    assert batch_size == 100, "Batch size is not correct."

    try:
        validate_theta_x(theta, x[1:])
    except AssertionError:
        pass


def test_check_hparams_maf():
    hparams = {
        "n_layers": 5,
        "layers": [50, 50, 50],
        "activation": jax.nn.relu,
        "use_reverse": True,
    }

    check_hparams_maf(hparams)

    try:
        check_hparams_maf({})
    except AssertionError:
        pass


def test_check_hparams_realnvp():
    hparams = {
        "n_layers": 5,
        "layers": [50, 50, 50],
        "activation": jax.nn.relu,
    }

    check_hparams_realnvp(hparams)

    try:
        check_hparams_realnvp({})
    except AssertionError:
        pass


def test_check_hparams_mdn():
    hparams = {"layers": [50, 50, 50], "activation": jax.nn.relu, "n_components": 5}

    check_hparams_mdn(hparams)

    try:
        check_hparams_mdn({})
    except AssertionError:
        pass
