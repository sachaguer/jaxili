# Release notes

These notes relate the changes between different `JaxILI` releases.

## 0.1.3

- Fix the save of checkpoints with the new `orbax` API.
- Add functions to load the checkpoints when using the inference class `NPE` or `NLE`.
- Fixed bugs related to issues when data points have zero or close to zero variance.
- Moved classes `Identity` and `Standardizer` to `compressor.py`.

## 0.1.2

- Remove `torch.utils.data.DataLoader` causing kernel crash. Replaced with DataLoaders from `jax_dataloader`.
- Add example notebooks to the API documentation.
- Dependencies of `JaxILI` have been updated to avoid conflicts between `jax` and `tensorflow` with `nvidia` packages.

## 0.1.1

- Update of the API documentation.

## 0.1

- Added different classes to have user-friendly calls to the models and trainer implemented in the pre-release.
  - Classes in `jaxili.inference` allow to create an object to train a model for Neural Posterior Estimation (NPE) or Neural Likelihood Estimation (NLE).
  - Classes in `jaxili.posterior` provide the abstract class to sample from the posterior and evaluate the log-posterior of the trained model. Currently, the `DirectPosterior` for NPE is implemented and the `MCMCPosterior` for NLE uses `numpyro` to run gradient-based MCMC algorithms.
- Future updates will add other samplers and add utilities to perform stacking of posteriors.

## 0.0.1 (Pre-release)

- First prerelease of `JaxILI`.
- The package contains the module to create and train Normalizing Flows (e.g. [Masked Autoregressive Flows](https://arxiv.org/abs/1705.07057) and [RealNVP](https://arxiv.org/abs/1605.08803)). They are contained in `jaxili.model` and can be trained with the `TrainerModule` in `jaxili.train`.
- `JaxILI` provides parent classes to use other types of Neural Density Estimator. Likewise, a class can inherit from the `TrainerModule` to define different loss functions, traning loop, etc...
