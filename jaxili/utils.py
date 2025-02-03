import numpy as np
from typing import Sequence, Union, Any
import jax.numpy as jnp

import torch
import torch.utils.data as data

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    else:
        return np.array(batch)
    
def create_data_loader(*datasets : Sequence[data.Dataset],
                       train : Union[bool, Sequence[bool]]=True,
                       batch_size : int = 128,
                       num_workers : int = 4,
                       seed : int = 42):
    """
    Creates data loaders used in JAX for a set of datasets.

    Parameters
    ----------
    datasets : Datasets for which data loaders are created.
    train : Sequence indicating which datasets are used for training and which not.
    If single bool, the same value is used for all datasets.
    batch_size : Batch size to use in the data loaders.
    num_workers : Number of workers for each datasets.
    seed : Seed to initalize the workers and shuffling with
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=is_train,
                                 drop_last=is_train,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers,
                                 persistent_workers=is_train,
                                 generator=torch.Generator().manual_seed(seed))
        loaders.append(loader)
    return loaders

def check_density_estimator(estimator_arg: str):
    """
    Check density estimator argument to see if it belongs to the authorized network.

    Parameters
    ----------
    estimator_arg : str
        Density estimator argument to check.
    """
    if estimator_arg not in ['maf', 'mdn', 'realnvp']:
        raise ValueError(f"Invalid density estimator argument: {estimator_arg}. Options are 'maf', 'mdn' and 'realnvp'.")
    
def validate_theta_x(theta: Any, x: Any):
    """
    Checks if the passed $(\theta, x)$ pair is valid.

    We check that:
    - $\theta$ and $x$ are jax arrays
    - $\theta$ and $x$ have the same number of samples.
    - $\theta$ and $x$ have dtype=float32.

    Raises:
        AssertionError if $\theta$ and $x$ are not jax arrays, do not have the same batch size or are not dtype==np.float32.
    
    Parameters
    ----------
    theta : Any
        Parameters of the simulations.
    x : Any
        Simulation outputs.
    """

    assert isinstance(theta, jnp.ndarray), "theta should be a jax array."
    assert isinstance(x, jnp.ndarray), "x should be a jax array."
    assert theta.shape[0] == x.shape[0], (
        f"Number of parameter sets ({theta.shape[0]}) and number of simulation outputs ({x.shape[0]}) should be the same."
    )

    assert theta.dtype == jnp.float32, "theta should have dtype float32."
    assert x.dtype == jnp.float32, "x should have dtype float32."

    batch_size = theta.shape[0]
    
    return theta, x, batch_size
