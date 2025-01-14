import jax


def test_cuda():
    devices = jax.devices()
    assert devices[0].platform == "gpu", "The test is not running on a GPU."
