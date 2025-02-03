# JaxILI

[![CI Test](https://github.com/sachaguer/jaxili/actions/workflows/ci.yml/badge.svg)]() [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![PyPI](https://img.shields.io/pypi/v/jaxili)](https://pypi.org/project/jaxili/) [![PyPI - License](https://img.shields.io/pypi/l/jaxili)]

This is a package to run Neural Density Estimation using Jax. The training is performed using `optax` (documentation available [here](https://optax.readthedocs.io/en/latest/)) and the neural network are created using `flax` (see [documentation](https://flax.readthedocs.io/en/latest/)).

The code is meant to provide tools to train Normalizing Flows easily to perform Implicit Likelihood Inference.

## Installation

Install `jaxili` using `PyPI`:

```bash
$ pip install jaxili
```

## First example: performing Neural Posterior Estimation

```python
from jaxili.inference import NPE
```

First fetch the data you want to train on:

```python
theta, x = ... #Theta corresponds to the parameter to be infered and x to the simulator output given theta.
```

Then create an inference object, add the simulations and train:

```python
inference = NPE()
inference.append_simulations(theta, x)

learning_rate = ... #Choose your learning rate
num_epochs = ... #Choose the number of epochs
batch_size = ... #Choose the batch size
checkpoint_path = ... #Choose the checkpoint path
checkpoint_path = os.path.abspath(checkpoint_path) #Beware, this should be an absolute path.

metrics, density_estimator = inference.train(
    training_batch_size=batch_size,
    learning_rate=learning_rate,
    checkpoint_path=checkpoint_path,
    num_epochs=num_epochs
)
```

You can then fetch the posterior to sample from it.

```python
posterior = inference.build_posterior()

observation = ... #The observation should have the shape [1, data vector size].
samples = posterior.sample(x=observation, num_samples=..., key=...) #You have to give a PRNGKey and specify the number of samples.
```


## Training a conditional MAF

If you want to control the architecture of the network you can use the following code to train e.g. a Masked Autoregressive Flow (MAF).

```python
import jax
import jax.numpy

from jaxili.utils import create_data_loader  #To create data loaders
from jaxili.train import TrainerModule #To perform the training
from jaxili.model import ConditionalMAF #The model used to learn the target distribution
from jaxili.loss import loss_nll_npe #Losses to train NFs with different configurations are provided
```

Given a train, validation and test set, one can create associated data loaders to perform the training.

```python
train_loader, val_loader, test_loader = create_data_loader(
    train_set, val_set, test_set,
    train = [True, False, False],
    batch_size=128
)
```

You can then specify hyperparameters for your training

```python
CHECKPOINT_PATH = ... #Path to save the weights of your neural network

loss_fn = loss_nll_npe

model_hparams_maf = {
    'n_in': dim_theta,
    'n_cond': dim_obs,
    'n_layers': 5,
    'layers': [50, 50],
    'activation': jax.nn.relu,
    'use_reverse': True,
    'seed' : 42
}

optimizer_hparams = { #hyperparameters of the optimizer for training
    'lr': 5e-4,
    'optimizer_name': 'adam'
}

logger_params = {
    'base_log_dir': CHECKPOINT_PATH
}

check_val_every_epoch = 1

debug = False

nde_class= "NPE"
```

A `TrainerModule` object can then be created to train the Neural Network:

```python
trainer_maf_npe = TrainerModule(
    model_class=ConditionalMAF,
    model_hparams=model_hparams_maf,
    optimizer_hparams=optimizer_hparams,
    loss_fn=loss_fn,
    exmp_input=next(iter(train_loader)),
    logger_params=logger_params,
    debug=debug,
    check_val_every_epoch=check_val_every_epoch,
    nde_class=nde_class    
)

#Train the Neural Density Estimator
metrics_maf_npe = trainer_maf_npe.train_model(
    train_loader, val_loader, test_loader=test_loader, num_epochs=500, patience=20
)
```
The trained model can then be used to sample from or compute the log-probability of the learned distribution:

```python
model_maf_npe = trainer_maf_npe.bind_model()

key, jax.random.PRNGKey(0)
samples_maf_npe = model_maf_npe.sample(
    observation, num_samples=10000, key=key
)
log_prob = model_maf_npe.apply(params, samples_maf_npe, observation, method="log_prob")
```