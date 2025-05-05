import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np

from jaxili.loss import loss_mse
import flax.linen as nn
from jaxili.compressor import Compressor
from jaxili.compressor import MLPCompressor

master_key = jax.random.PRNGKey(0)
num_samples = 10_000
n_dim = 3
n_reals = 11
def simulator(theta, rng_key):
    batch_size = theta.shape[0]
    return (theta[..., None] + jax.random.normal(rng_key, shape=(batch_size, n_dim, n_reals))*0.1).reshape(batch_size, n_dim*n_reals)

train_set_size = 10_000
master_key, theta_key = jax.random.split(master_key)
theta_train = jax.random.uniform(theta_key, shape=(train_set_size, n_dim), minval=jnp.array([-2., -2., -2.]), maxval=jnp.array([2., 2., 2.]))
master_key, simkey = jax.random.split(master_key)
x_train = simulator(theta_train, simkey)

model_class=MLPCompressor

model_hparams={
    'hidden_size': [8, 4],
    'activation': nn.relu,
    'output_size': n_dim,
}

def test_init():
    compressor = Compressor(
        model_class=model_class,
        model_hparams=model_hparams,
    )
    assert (
        compressor._model_class == model_class
    ), "The model class is not MLPCompressor."
    assert (
        compressor._model_hparams == model_hparams
    ), "The model hyperparameters are not correctly initialized."
    assert (
        compressor._logging_level == "WARNING"
    ), "The logging level is not 'WARNING'."
    assert compressor.verbose == True, "The verbose attribute is not True."
    assert (
        compressor._loss_fn == loss_mse
    ), "The loss function is not correctly initialized."

def test_append_simulations():
    compressor = Compressor(
        model_class=model_class,
        model_hparams=model_hparams,
    )
    compressor = compressor.append_simulations(theta_train, x_train)
    assert (
        compressor._dim_params == theta_train.shape[1]
    ), "The number of parameters is wrong."
    assert (
        compressor._dim_sim == x_train.shape[1]
    ), "The number of the conditionning variable is wrong."
    assert compressor._num_sims == train_set_size, "The number of simulations is wrong."
    assert compressor._train_dataset is not None, "The training dataset is None."
    assert compressor._val_dataset is not None, "The validation dataset is None."
    assert compressor._test_dataset is not None, "The test dataset is None."


    # Test only adding train and validation split
    compressor = Compressor(
        model_class=model_class,
        model_hparams=model_hparams,
    )
    compressor = compressor.append_simulations(
        theta_train, x_train, train_test_split=[0.7, 0.3]
    )
    assert (
        compressor._dim_params == theta_train.shape[1]
    ), "The number of parameters is wrong."
    assert (
        compressor._dim_sim == x_train.shape[1]
    ), "The number of the conditionning variable is wrong."
    assert compressor._num_sims == train_set_size, "The number of simulations is wrong."
    assert compressor._train_dataset is not None, "The training dataset is None."
    assert compressor._val_dataset is not None, "The validation dataset is None."
    assert compressor._test_dataset is None, "The test dataset is not None."

def test_dataloaders():
    compressor = Compressor(
        model_class=model_class,
        model_hparams=model_hparams,
    )
    compressor = compressor.append_simulations(theta_train, x_train)
    batch_size = 64
    compressor._create_data_loader(batch_size=batch_size)
    assert compressor._train_loader is not None, "The training loader is None."
    assert compressor._val_loader is not None, "The validation loader is None."
    assert compressor._test_loader is not None, "The test loader is None."
    test_train = next(iter(compressor._train_loader))
    test_val = next(iter(compressor._val_loader))
    test_test = next(iter(compressor._test_loader))
    assert test_train[0].shape[0] == batch_size, "The training batch size is wrong."
    assert test_val[0].shape[0] == batch_size, "The validation batch size is wrong."
    assert test_test[0].shape[0] == batch_size, "The test batch size is wrong."
    
    # Test only adding train and validation split
    compressor = Compressor(
        model_class=model_class,
        model_hparams=model_hparams,
    )
    compressor.append_simulations(theta_train, x_train, train_test_split=[0.7, 0.3])
    compressor._create_data_loader(batch_size=batch_size)
    assert compressor._train_loader is not None, "The training loader is None."
    assert compressor._val_loader is not None, "The validation loader is None."
    assert compressor._test_loader is None, "The test loader is not None."
    test_train = next(iter(compressor._train_loader))
    test_val = next(iter(compressor._val_loader))
    assert test_train[0].shape[0] == batch_size, "The training batch size is wrong."
    assert test_val[0].shape[0] == batch_size, "The validation batch size is wrong."

def test_build_neural_network():
    compressor = Compressor(
        model_class=model_class,
        model_hparams=model_hparams,
    )
    compressor.append_simulations(theta_train, x_train)
    model = compressor._build_neural_network()
    assert model is not None, "The model is None."
    assert isinstance(model, model_class), "The model is not of the correct class."
    params = model.init(
        master_key,
        x_train,
    )
    t_train = model.apply(
        params,
        x_train,
    )
    assert t_train is not None, "The training output is None."
    assert t_train.shape == theta_train.shape, "The training output shape is wrong."

def test_training():
    learning_rate = 5e-4
    gradient_clip = 5.0
    warmup = 0.1
    weight_decay = 0.0
    batch_size = 64
    checkpoint_path = "~/test"
    checkpoint_path = os.path.expanduser(checkpoint_path)

    compressor = Compressor(
        model_class=model_class,
        model_hparams=model_hparams,
    )
    compressor.append_simulations(theta_train, x_train)
    metrics, compression_function = compressor.train(
        training_batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_clip=gradient_clip,
        warmup=warmup,
        weight_decay=weight_decay,
        checkpoint_path=checkpoint_path,
        z_score_x=True,
    )
    assert metrics is not None, "The metrics are None."
    assert compression_function is not None, "The compression function is None."

    t_train = compression_function(x_train)
    assert t_train is not None, "The training output is None."
    assert t_train.shape == theta_train.shape, "The training output shape is wrong."

    # Test if the checkpoints have been saved
    assert os.path.exists(
        os.path.join(checkpoint_path)
    ), "The checkpoint log dir does not exist. Check ~/test."
    assert os.path.exists(
        os.path.join(checkpoint_path, "MLPCompressor/version_0")
    ), "The checkpoint dir does not exist. Check ~/test/."
    assert os.path.exists(
        os.path.join(checkpoint_path, "MLPCompressor/version_0/metrics")
    ), "The metrics dir does not exist. Check ~/test/MLPCompressor/version_0."
    assert os.path.exists(
        os.path.join(checkpoint_path, "MLPCompressor/version_0/hparams.json")
    ), "The hparams JSON file does not exist. Check ~/test/MLPCompressor/version_0."

    shutil.rmtree(checkpoint_path)
