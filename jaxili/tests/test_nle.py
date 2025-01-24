import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from jaxili.inference.nle import NLE
from jaxili.inference.nle import default_maf_hparams
from jaxili.model import ConditionalMAF

import sbibm

task = sbibm.get_task("slcp")
simulator = task.get_simulator()
prior = task.get_prior()
simulator = task.get_simulator()

train_set_size = 10_000
theta_train = prior(num_samples=train_set_size)
x_train = simulator(theta_train)

theta_train, x_train = np.array(theta_train), np.array(x_train)

def test_init():
    inference = NLE()

    assert (
        inference._model_class == ConditionalMAF
    ), "The model class is not ConditionalMAF."
    assert inference._logging_level == "WARNING", "The logging level is not 'WARNING'."
    assert inference.verbose == True, "The verbose attribute is not True."
    assert (
        inference._model_hparams is default_maf_hparams
    ), "The model hyperparameters are not correctly initialized."


def test_append_simulations():

    inference = NLE()
    inference = inference.append_simulations(theta_train, x_train)
    assert (
        inference._dim_params == x_train.shape[1]
    ), "The number of parameters is wrong."
    assert inference._dim_cond == theta_train.shape[1], "The number of the conditionning variable is wrong."
    assert inference._num_sims == train_set_size, "The number of simulations is wrong."

    assert inference._train_dataset is not None, "The training dataset is None."
    assert inference._val_dataset is not None, "The validation dataset is None."
    assert inference._test_dataset is not None, "The test dataset is None."

    # Test only adding train and validation split
    inference = NLE()
    inference = inference.append_simulations(
        theta_train, x_train, train_test_split=[0.7, 0.3]
    )
    assert (
        inference._dim_params == x_train.shape[1]
    ), "The number of parameters is wrong."
    assert inference._dim_cond == theta_train.shape[1], "The number of the conditionning variable is wrong."
    assert inference._num_sims == train_set_size, "The number of simulations is wrong."

    assert inference._train_dataset is not None, "The training dataset is None."
    assert inference._val_dataset is not None, "The validation dataset is None."
    assert inference._test_dataset is None, "The test dataset is not None."


def test_dataloaders():

    inference = NLE()
    inference = inference.append_simulations(theta_train, x_train)

    batch_size = 64

    inference._create_data_loader(batch_size=batch_size)

    assert inference._train_loader is not None, "The training loader is None."
    assert inference._val_loader is not None, "The validation loader is None."
    assert inference._test_loader is not None, "The test loader is None."

    test_train = next(iter(inference._train_loader))
    test_val = next(iter(inference._val_loader))
    test_test = next(iter(inference._test_loader))

    assert test_train[0].shape[0] == batch_size, "The training batch size is wrong."
    assert test_val[0].shape[0] == batch_size, "The validation batch size is wrong."
    assert test_test[0].shape[0] == batch_size, "The test batch size is wrong."

    # Test only adding train and validation split
    inference = NLE()
    inference.append_simulations(theta_train, x_train, train_test_split=[0.7, 0.3])

    inference._create_data_loader(batch_size=batch_size)

    assert inference._train_loader is not None, "The training loader is None."
    assert inference._val_loader is not None, "The validation loader is None."
    assert inference._test_loader is None, "The test loader is not None."

    test_train = next(iter(inference._train_loader))
    test_val = next(iter(inference._val_loader))

    assert test_train[0].shape[0] == batch_size, "The training batch size is wrong."
    assert test_val[0].shape[0] == batch_size, "The validation batch size is wrong."


def test_build_neural_network():
    inference = NLE()
    inference.append_simulations(theta_train, x_train)

    model = inference._build_neural_network()

    assert model is not None, "The model is None."
    assert inference._transformation is not None, "The transformation is None."
    assert inference._embedding_net is not None, "The embedding net is None."

    shift = jnp.mean(inference._train_dataset.theta, axis=0)
    scale = jnp.std(inference._train_dataset.theta, axis=0)

    standardized_theta = (inference._train_dataset.theta - shift) / scale

    params = model.init(jax.random.PRNGKey(0), x_train, theta_train)

    test_theta = model.apply(params, inference._train_dataset.theta, method="embedding")

    npt.assert_allclose(standardized_theta, test_theta, rtol=1e-5, atol=1e-5)

    test_x = model.apply(params, inference._train_dataset.x, method="standardize")
    shift_x = jnp.mean(inference._train_dataset.x, axis=0)
    scale_x = jnp.std(inference._train_dataset.x, axis=0)
    standardized_x = (inference._train_dataset.x - shift_x) / scale_x
    npt.assert_allclose(standardized_x, test_x, rtol=1e-5, atol=1e-5)

    log_prob = model.apply(params, inference._train_dataset.x, inference._train_dataset.theta, method="log_prob")
    assert log_prob.shape[0] == len(inference._train_dataset), "The shape of the output of log_prob method is wrong."

    samples = model.apply(params, inference._train_dataset.theta[0], num_samples=10_000, key=jax.random.PRNGKey(0), method="sample")

    assert samples.shape == (10_000, inference._dim_params), "The shape of the samples is wrong."

def test_training():
    #Test if the training pipeline runs without troubleshot
    learning_rate = 5e-4
    gradient_clip = 5.0
    warmup = 0.1
    weight_decay = 0.0
    batch_size = 64
    checkpoint_path = "~/test"
    checkpoint_path = os.path.expanduser(checkpoint_path)

    inference = NLE()

    inference.append_simulations(theta_train, x_train)

    metrics, density_estimator = inference.train(
        training_batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_clip=gradient_clip,
        warmup=warmup,
        weight_decay=weight_decay,
        checkpoint_path=checkpoint_path,
        z_score_x=True,
    )

    assert metrics is not None, "The metrics are None."
    assert density_estimator is not None, "The density estimator is None."

    #Test if the density estimator can return a log_prob
    log_prob = density_estimator.log_prob(x_train[0:10], theta_train[0:10])
    assert log_prob.shape == (10,), "The shape of the output of log_prob method is wrong."

    #Test if the density estimator can return samples
    samples = density_estimator.sample(theta_train[0], num_samples=10_000, key=jax.random.PRNGKey(0))
    assert samples.shape == (10_000, inference._dim_params), "The shape of the samples is wrong."

    #Test if the checkpoints have been saved
    assert os.path.exists(os.path.join(checkpoint_path)), "The checkpoint log dir does not exist. Check ~/test."
    assert os.path.exists(os.path.join(checkpoint_path, "NDE_w_Standardization/version_0")), "The checkpoint dir does not exist. Check ~/test/."
    #assert os.path.exists(os.path.join(checkpoint_path, "NDE_w_Standardization/version_0/")), "The checkpoint dir does not exist. Check ~/test/NDE_w_Standardization/version_0."
    assert os.path.exists(os.path.join(checkpoint_path, "NDE_w_Standardization/version_0/metrics")), "The metrics dir does not exist. Check ~/test/NDE_w_Standardization/version_0."
    assert os.path.exists(os.path.join(checkpoint_path, "NDE_w_Standardization/version_0/hparams.json")), "The hparams JSON file does not exist. Check ~/test/NDE_w_Standardization/version_0."

    shutil.rmtree(checkpoint_path)