import os
import shutil

import jax
import jax.numpy as jnp
import jax_dataloader as jdl

from jaxili.utils import create_data_loader
from jaxili.train import TrainerModule
from jaxili.model import ConditionalMAF
from jaxili.loss import loss_nll_npe
from jaxili.inference import NPE

import numpy.testing as npt

# Let's use the simulator used in the examples
n_dim = 3


def simulator(theta, rng_key):
    batch_size = theta.shape[0]
    return theta + jax.random.normal(rng_key, shape=(batch_size, n_dim)) * 0.1


master_key = jax.random.PRNGKey(0)
num_samples = 10_000

theta_key, master_key = jax.random.split(master_key)

# Draw the parameters from the prior
theta = jax.random.uniform(
    theta_key,
    shape=(num_samples, n_dim),
    minval=jnp.array([-2.0, -2.0, -2.0]),
    maxval=jnp.array([2.0, 2.0, 2.0]),
)

sim_key, master_key = jax.random.split(master_key)
x = simulator(theta, sim_key)

train_set = jdl.ArrayDataset(theta[:7000], x[:7000])
val_set = jdl.ArrayDataset(theta[7000:9000], x[7000:9000])
test_set = jdl.ArrayDataset(theta[:9000], x[:9000])

train_loader, val_loader, test_loader = create_data_loader(
    train_set, val_set, test_set, train=[True, False, False], batch_size=128
)

CHECKPOINT_PATH = "~/test"
CHECKPOINT_PATH = os.path.expanduser(CHECKPOINT_PATH)

loss_fn = loss_nll_npe

model_hparams_maf = {
    "n_in": n_dim,
    "n_cond": n_dim,
    "n_layers": 5,
    "layers": [50, 50],
    "activation": jax.nn.relu,
    "use_reverse": True,
    "seed": 42,
}

optimizer_hparams = {  # hyperparameters of the optimizer for training
    "lr": 5e-4,
    "optimizer_name": "adam",
}

logger_params = {"base_log_dir": CHECKPOINT_PATH}

check_val_every_epoch = 1

debug = False

nde_class = "NPE"

# Let's first create an observation
obs_key, master_key = jax.random.split(master_key)
fiducial = jnp.array([[0.5, 0.5, 0.5]])
obs = simulator(fiducial, obs_key)


def test_checkpoint_maf():
    # First train network
    trainer_maf_npe = TrainerModule(
        model_class=ConditionalMAF,
        model_hparams=model_hparams_maf,
        optimizer_hparams=optimizer_hparams,
        loss_fn=loss_fn,
        exmp_input=next(iter(train_loader)),
        logger_params=logger_params,
        debug=debug,
        check_val_every_epoch=check_val_every_epoch,
        nde_class=nde_class,
    )

    # Train the Neural Density Estimator
    _ = trainer_maf_npe.train_model(
        train_loader, val_loader, test_loader=test_loader, num_epochs=500, patience=20
    )

    model_maf_npe = trainer_maf_npe.model

    sample_key = jax.random.PRNGKey(0)
    samples_maf_npe = model_maf_npe.sample(obs, num_samples=10000, key=sample_key)

    # Reload from the checkpoints
    trainer_maf_npe = TrainerModule.load_from_checkpoints(
        model_class=ConditionalMAF,
        checkpoint=CHECKPOINT_PATH + "/ConditionalMAF/version_0/",
    )

    model_maf_npe = trainer_maf_npe.model

    samples_maf_npe_ckpt = model_maf_npe.sample(obs, num_samples=10000, key=sample_key)

    # Assert that samples_maf_npe and samples_maf_npe_ckpt are equal
    npt.assert_array_equal(samples_maf_npe, samples_maf_npe_ckpt)

    shutil.rmtree(CHECKPOINT_PATH)


def test_checkpoints_npe():
    inference = NPE()

    inference = inference.append_simulations(theta, x)

    metrics, density_estimator = inference.train(
        checkpoint_path=CHECKPOINT_PATH, num_epochs=500
    )

    posterior = inference.build_posterior()

    sample_key = jax.random.PRNGKey(0)
    samples = posterior.sample(x=obs, num_samples=10000, key=sample_key)

    inference = NPE.load_from_checkpoints(
        checkpoint=CHECKPOINT_PATH + "/NDE_w_Standardization/version_0/",
    )

    posterior = inference.build_posterior()

    samples_ckpt = posterior.sample(x=obs, num_samples=10000, key=sample_key)

    npt.assert_allclose(samples, samples_ckpt, rtol=1e-05, atol=1e-08)

    shutil.rmtree(CHECKPOINT_PATH)
