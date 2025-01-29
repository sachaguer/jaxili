import warnings

import jax


def test_cuda():
    devices = jax.devices()
    if devices[0].platform != "gpu":
        warnings.warn("The test is not running on a GPU.")
