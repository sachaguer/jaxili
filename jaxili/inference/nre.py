"""
NRE.

This module provides the Neural Ratio Estimation (NRE) class to train a neural network to estimate the likelihood ratio.
"""

import os
import json
import re
import warnings
import copy
from typing import Any, Callable, Dict, Iterable, Optional, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
import numpy as np
from jaxtyping import Array, Float, PyTree

import jaxili
from jaxili.loss import loss_bce_nre # Use NRE loss
from jaxili.model import RatioEstimatorMLP, NDE_w_Standardization # Use NRE model
from jaxili.compressor import Identity, Standardizer
# NRE doesn't build a posterior object, it returns a ratio function
# from jaxili.posterior import DirectPosterior # Remove posterior import
from jaxili.train import TrainerModule
from jaxili.utils import *
from jaxili.utils import (
    # check_density_estimator, # Not applicable to NRE classifier
    create_data_loader,
    validate_theta_x,
)
# TODO: Update func_dict imports if necessary (maybe add NRE loss/model there?)
from jaxili.inventory.func_dict import jaxili_loss_dict, jax_nn_dict, jaxili_nn_dict

# Default hyperparameters for the RatioEstimatorMLP
default_mlp_hparams = {
    "layers": [50, 50],
    "activation": jax.nn.relu,
}

# Define the wrapper here to be accessible by class methods like load_from_checkpoints
# Also cleaner than defining inline multiple times
class NREModelWrapper(nn.Module):
    theta_compressor: nn.Module
    x_compressor: nn.Module
    embedding_net: nn.Module
    classifier_class: nn.Module
    classifier_hparams: dict

    def setup(self):
        self.classifier = self.classifier_class(**self.classifier_hparams)

    def __call__(self, theta, x):
        theta_std = self.theta_compressor(theta)
        x_std = self.x_compressor(x)
        x_embedded = self.embedding_net(x_std)
        classifier_input = jnp.concatenate([theta_std, x_embedded], axis=-1)
        logits = self.classifier(classifier_input)
        return logits

    def log_ratio(self, theta, x):
        logits = self.__call__(theta, x)
        # The logit output by the classifier trained with BCE loss is log(r(x,theta))
        return logits

class NRE:
    """
    NRE.

    Base class for Neural Ratio Estimation (NRE) methods.
    Default configuration uses a simple Multi-Layer Perceptron (MLP) to learn the likelihood ratio.

    Examples
    --------
    >>> from jaxili.inference import NRE
    >>> inference = NRE()
    >>> theta, x = ...  # Load parameters and simulation outputs
    >>> inference.append_simulations(theta, x)
    >>> metrics, ratio_estimator = inference.train() # Train your ratio estimator
    >>> log_ratio = ratio_estimator.log_ratio(theta_observed, x_observed)
    """

    def __init__(
        self,
        model_class: nn.Module = RatioEstimatorMLP, # Default to MLP
        logging_level: Union[int, str] = "WARNING",
        verbose: bool = True,
        model_hparams: Optional[Dict[str, Any]] = None, # Use None to merge with default later
        loss_fn: Callable = loss_bce_nre, # Default to BCE loss
    ):
        """
        Initialize class for Neural Ratio Estimation (NRE) methods.

        Parameters
        ----------
        model_class : flax.linen.Module
            Class of the neural network to use for ratio estimation. Default: RatioEstimatorMLP.
        model_hparams : Dict[str, Any], optional
            Hyperparameters to use for the model. If None, uses default MLP hparams.
        logging_level: Union[int, str], optional
            Logging level to use. Default is "WARNING".
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        loss_fn : Callable
            Loss function to use for training. Default: loss_bce_nre.
        """
        self._model_class_uninit = model_class
        # Merge provided hparams with defaults
        _hparams = default_mlp_hparams.copy()
        if model_hparams is not None:
            _hparams.update(model_hparams)
        self._model_hparams = _hparams
        self._logging_level = logging_level
        # Ensure the provided loss function is compatible (expects key)
        # TODO: Add check for loss function signature? Or rely on TrainerModule error?
        self._loss_fn = loss_fn
        self.verbose = verbose
        # Initialize placeholder compressors/embedding net
        self._theta_compressor = None
        self._x_compressor = None
        self._embedding_net = None

    def set_model_hparams(self, hparams):
        """
        Set the hyperparameters of the model.

        Parameters
        ----------
        hparams : Dict[str, Any]
            Hyperparameters to use for the model.
        """
        # Reset hparams completely (don't merge with default again)
        self._model_hparams = hparams

    def set_loss_fn(self, loss_fn):
        """
        Set the loss function to use for training.

        Parameters
        ----------
        loss_fn : Callable
            Loss function to use for training.
        """
        self._loss_fn = loss_fn

    def set_dataset(self, dataset, type):
        """
        Set the dataset to use for training, validation or testing.

        Parameters
        ----------
        dataset : data.Dataset
            Dataset to use.
        type : str
            Type of the dataset. Can be 'train', 'val' or 'test'.
        """
        assert type in [
            "train",
            "val",
            "test",
        ], "Type should be 'train', 'val' or 'test'."

        if type == "train":
            self._train_dataset = dataset
        elif type == "val":
            self._val_dataset = dataset
        elif type == "test":
            self._test_dataset = dataset

    def set_dataloader(self, dataloader, type):
        """
        Set the dataloader to use for training, validation or testing.

        Parameters
        ----------
        dataloader : data.DataLoader
            dataloader to use.
        type : str
            Type of the dataloader. Can be 'train', 'val' or 'test'.
        """
        assert type in [
            "train",
            "val",
            "test",
        ], "Type should be 'train', 'val' or 'test'."

        if type == "train":
            self._train_dataloader = dataloader
        elif type == "val":
            self._val_dataloader = dataloader
        elif type == "test":
            self._test_dataloader = dataloader

    def append_simulations(
        self,
        theta: Array,
        x: Array,
        train_test_split: Iterable[float] = [0.7, 0.2, 0.1],
        key: Optional[PyTree] = None,
    ):
        """
        Store parameters and simulation outputs to use them for later training.

        Data is stored in a Dataset object from `jax-dataloader`

        Parameters
        ----------
        theta : Array
            Parameters of the simulations.
        x : Array
            Simulation outputs.
        train_test_split : Iterable[float], optional
            Fractions to split the dataset into training, validation and test sets.
            Should be of length 2 or 3. A length 2 list will not generate a test set. Default is [0.7, 0.2, 0.1].
        key : PyTree, optional
            Key to use for the random permutation of the dataset. Default is None.
        """
        theta, x, num_sims = validate_theta_x(theta, x)
        if self.verbose:
            print(f"[!] Inputs are valid.")
            print(f"[!] Appending {num_sims} simulations to the dataset.")

        self._dim_params = theta.shape[1]
        self._dim_cond = x.shape[1]
        self._num_sims = num_sims

        # Reset fitted status of compressors if appending new data
        if hasattr(self, "_theta_compressor") and self._theta_compressor is not None:
             self._theta_compressor = None # Mark for rebuild/refit
        if hasattr(self, "_x_compressor") and self._x_compressor is not None:
             self._x_compressor = None

        is_test_set = len(train_test_split) == 3
        if is_test_set:
            train_fraction, val_fraction, test_fraction = train_test_split
            assert np.isclose(
                train_fraction + val_fraction + test_fraction, 1.0
            ), "The sum of the split fractions should be 1."
        elif len(train_test_split) == 2:
            train_fraction, val_fraction = train_test_split
            assert np.isclose(
                train_fraction + val_fraction, 1.0
            ), "The sum of the split fractions should be 1."
        else:
            raise ValueError("train_test_split should have 2 or 3 elements.")

        if key is None:
            key = jr.PRNGKey(np.random.randint(0, 1000))
        index_permutation = jr.permutation(key, num_sims)

        train_idx = index_permutation[: int(train_fraction * num_sims)]
        val_idx = index_permutation[
            int(train_fraction * num_sims) : int(
                (train_fraction + val_fraction) * num_sims
            )
        ]
        if is_test_set:
            test_idx = index_permutation[
                int((train_fraction + val_fraction) * num_sims) :
            ]

        # Store raw data, compressors fit later in _build_neural_network
        self.set_dataset(jdl.ArrayDataset(theta[train_idx], x[train_idx]), type="train")
        self.set_dataset(jdl.ArrayDataset(theta[val_idx], x[val_idx]), type="val")
        self.set_dataset(
            jdl.ArrayDataset(theta[test_idx], x[test_idx]) if is_test_set else None,
            type="test",
        )

        if self.verbose:
            print(f"[!] Dataset split into training, validation and test sets.")
            print(f"[!] Training set: {len(train_idx)} simulations.")
            print(f"[!] Validation set: {len(val_idx)} simulations.")
            if is_test_set:
                print(f"[!] Test set: {len(test_idx)} simulations.")
        return self

    def _create_data_loader(self, **kwargs):
        """
        Create DataLoaders for the training, validation and test datasets.
        Can only be executed after appending simulations.

        Parameters
        ----------
        batch_size : int
            Batch size for the training dataloader.
        batch_size_val : int, optional
            Batch size for the validation dataloader. If None, use batch_size. Default is None.
        batch_size_test : int, optional
            Batch size for the test dataloader. If None, use batch_size. Default is None.
        shuffle : bool, optional
            Whether to shuffle the training dataset. Default is True.
        key : PyTree, optional
            Key to use for shuffling the training dataset. Default is None.
        """
        batch_size = kwargs.pop("batch_size", 128)
        batch_size_val = kwargs.pop("batch_size_val", batch_size)
        batch_size_test = kwargs.pop("batch_size_test", batch_size)
        shuffle = kwargs.pop("shuffle", True)
        key = kwargs.pop("key", None)

        if key is None:
            key = jr.PRNGKey(np.random.randint(0, 1000))

        train_loader, val_loader, test_loader = create_data_loader(
            self._train_dataset,
            self._val_dataset,
            self._test_dataset,
            train=[True, False, False],
            batch_size=[batch_size, batch_size_val, batch_size_test],
            shuffle=shuffle,
            rng=key,
            **kwargs,
        )

        self.set_dataloader(train_loader, type="train")
        self.set_dataloader(val_loader, type="val")
        if test_loader:
            self.set_dataloader(test_loader, type="test")

        if self.verbose:
            print(f"[!] DataLoaders created.")
            print(f"[!] Training DataLoader with batch size {batch_size}.")
            print(f"[!] Validation DataLoader with batch size {batch_size_val}.")
            if test_loader:
                print(f"[!] Test DataLoader with batch size {batch_size_test}.")
        return train_loader, val_loader, test_loader

    def _build_neural_network(
        self,
        z_score_theta: bool = True,
        z_score_x: bool = True,
        embedding_net: nn.Module = Identity,
        embedding_hparams: Optional[dict] = None,
        **kwargs,
    ):
        """
        Build the neural network components for NRE.

        Initializes or fits standardization layers and the embedding network.
        Does not return the full model, but sets up the components.

        Parameters
        ----------
        z_score_theta : bool, optional
            Whether to z-score parameters (theta). Default is True.
        z_score_x : bool, optional
            Whether to z-score conditions (x). Default is True.
        embedding_net : nn.Module, optional
            Embedding network class for conditions (x). Default is Identity.
        embedding_hparams : dict, optional
            Hyperparameters for the embedding network. Default is None.
        """
        if self.verbose:
            print(f"[!] Configuring Neural Network components.")

        # Initialize compressors only if not already done or if settings change
        if self._theta_compressor is None or kwargs.get("force_rebuild", False):
            self._theta_compressor = Standardizer() if z_score_theta else Identity()
            if isinstance(self._theta_compressor, Standardizer):
                if hasattr(self, "_train_dataset"):
                    self._theta_compressor.fit(self._train_dataset.data[0]) # Theta is first element
                    if self.verbose:
                         print("[!] Fitted theta standardizer.")
                else:
                    warnings.warn("Theta compressor cannot be fitted without training data.")

        if self._x_compressor is None or kwargs.get("force_rebuild", False):
            self._x_compressor = Standardizer() if z_score_x else Identity()
            if isinstance(self._x_compressor, Standardizer):
                if hasattr(self, "_train_dataset"):
                    self._x_compressor.fit(self._train_dataset.data[1]) # x is second element
                    if self.verbose:
                         print("[!] Fitted x standardizer.")
                else:
                    warnings.warn("X compressor cannot be fitted without training data.")

        if self._embedding_net is None or kwargs.get("force_rebuild", False):
            self._embedding_net = embedding_net(**(embedding_hparams or {}))
            if self.verbose and not isinstance(self._embedding_net, Identity):
                 print(f"[!] Using embedding net: {self._embedding_net.__class__.__name__}")

        # Determine embedding output dimension if needed (though MLP usually infers)
        # Note: RatioEstimatorMLP takes concatenated input, shape inference handles it.

        if self.verbose:
            if isinstance(self._theta_compressor, Standardizer):
                print("[!] Parameters (theta) will be z-scored.")
            if isinstance(self._x_compressor, Standardizer):
                 print("[!] Conditions (x) will be z-scored.")

        # Model structure is now defined via NREModelWrapper using these components
        # Actual instantiation happens in create_trainer
        return self

    def create_trainer(
        self,
        optimizer_hparams: Dict[str, Any],
        seed: int = 42,
        logger_params: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        check_val_every_epoch: int = 1,
        **kwargs,
    ):
        """
        Create a trainer module for the NRE network.

        Parameters
        ----------
        optimizer_hparams : Dict[str, Any]
            Hyperparameters for the optimizer (e.g., learning rate, optimizer name).
        seed : int, optional
            Random seed for reproducibility. Default is 42.
        logger_params : Dict[str, Any], optional
            Parameters for the logger (e.g., base log directory). Default is None.
        debug : bool, optional
            Whether to run in debug mode (more logging). Default is False.
        check_val_every_epoch : int, optional
            Frequency of validation checks. Default is 1.

        Returns
        -------
        TrainerModule
            The initialized trainer module.
        """
        # Ensure components are built/configured
        assert self._theta_compressor is not None, "Network components not built. Call _build_neural_network first."
        assert self._x_compressor is not None, "Network components not built. Call _build_neural_network first."
        assert self._embedding_net is not None, "Network components not built. Call _build_neural_network first."

        # Get an example input for initialization
        if not hasattr(self, "_train_dataloader"):
            self._create_data_loader(batch_size=kwargs.get("training_batch_size", 128))
        exmp_input = next(iter(self._train_dataloader))

        # Package hyperparameters for the NREModelWrapper
        model_wrapper_hparams = {
            'theta_compressor': self._theta_compressor, # Pass initialized components
            'x_compressor': self._x_compressor,
            'embedding_net': self._embedding_net,
            'classifier_class': self._model_class_uninit, # Pass the class
            'classifier_hparams': self._model_hparams, # Pass its hparams
        }

        if self.verbose:
            print(f"[!] Creating Trainer for {self._model_class_uninit.__name__}.")

        # TrainerModule needs slight modification or a custom training step
        # to handle the PRNG key required by the new loss_bce_nre
        # Let's assume TrainerModule can handle passing RNGs to the loss function
        # during its train/eval steps (or we need to adapt TrainerModule)

        self.trainer = TrainerModule(
            model_class=NREModelWrapper, # Pass the wrapper class
            model_hparams=model_wrapper_hparams, # Pass composed hparams
            optimizer_hparams=optimizer_hparams,
            loss_fn=self._loss_fn,
            exmp_input=exmp_input,
            logger_params=logger_params,
            debug=debug,
            check_val_every_epoch=check_val_every_epoch,
            nde_class="NRE",
            # Pass seed to trainer for potential use in loss function key generation
            seed=seed,
        )
        return self.trainer

    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        patience: int = 20,
        num_epochs: int = 2**31 - 1,
        check_val_every_epoch: int = 1,
        optimizer_name: str = 'adam',
        seed: int = 42, # Add seed for reproducibility in training and loss
        checkpoint_path: Optional[str] = None,
        z_score_theta: bool = True,
        z_score_x: bool = True,
        embedding_net: nn.Module = Identity,
        embedding_hparams: Optional[dict] = None,
        **kwargs,
    ):
        """
        Train the Neural Ratio Estimator.

        Parameters
        ----------
        training_batch_size : int, optional
            Batch size for training. Default is 50.
        learning_rate : float, optional
            Learning rate for the optimizer. Default is 5e-4.
        patience : int, optional
            Number of epochs to wait for improvement before early stopping. Default is 20.
        num_epochs : int, optional
            Maximum number of epochs to train. Default is effectively infinity.
        check_val_every_epoch : int, optional
            Frequency of validation checks. Default is 1.
        optimizer_name : str, optional
            Name of the optimizer to use (e.g., 'adam', 'sgd'). Default is 'adam'.
        seed : int, optional
            Random seed for initializing trainer and potentially loss shuffling. Default is 42.
        checkpoint_path : str, optional
            Path to save model checkpoints. Default is None (no saving).
        z_score_theta : bool, optional
            Whether to z-score parameters (theta). Default is True.
        z_score_x : bool, optional
            Whether to z-score conditions (x). Default is True.
        embedding_net : nn.Module, optional
            Embedding network class for conditions (x). Default is Identity.
        embedding_hparams : dict, optional
            Hyperparameters for the embedding network. Default is None.

        Returns
        -------
        metrics : dict
            Dictionary containing training metrics.
        ratio_estimator : NREModelWrapper
            The trained ratio estimator model instance (bound with parameters).
        """
        assert hasattr(self, "_train_dataset"), "Training dataset not set. Use append_simulations()."
        assert hasattr(self, "_val_dataset"), "Validation dataset not set. Use append_simulations()."

        # Create dataloaders
        train_loader, val_loader, test_loader = self._create_data_loader(
            batch_size=training_batch_size, key=jr.PRNGKey(seed), **kwargs
        )

        # Build/configure the neural network components
        self._build_neural_network(
            z_score_theta=z_score_theta,
            z_score_x=z_score_x,
            embedding_net=embedding_net,
            embedding_hparams=embedding_hparams,
            force_rebuild=kwargs.get("force_rebuild", False)
        )

        # Create the trainer
        optimizer_hparams = {"lr": learning_rate, "optimizer_name": optimizer_name}
        logger_params = {"base_log_dir": checkpoint_path} if checkpoint_path else None

        trainer = self.create_trainer(
            optimizer_hparams=optimizer_hparams,
            logger_params=logger_params,
            check_val_every_epoch=check_val_every_epoch,
            seed=seed,
            **kwargs,
        )

        if self.verbose:
            print(f"[!] Starting training...")
            print(f"    Model: {self._model_class_uninit.__name__}, Loss: {self._loss_fn.__name__}")
            print(f"    Optimizer: {optimizer_name}, LR: {learning_rate}")
            print(f"    Batch Size: {training_batch_size}, Seed: {seed}")
            print(f"    Epochs: {num_epochs}, Patience: {patience}")
            if checkpoint_path:
                print(f"    Checkpoints: {checkpoint_path}")

        # Train the model
        # Assumes trainer.train_model handles passing keys to loss if needed
        metrics = trainer.train_model(
            train_loader,
            val_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            patience=patience,
        )

        # Bind the trained parameters to the model structure
        self.ratio_estimator = trainer.bind_model()

        if self.verbose:
            print(f"[!] Training finished.")
            if checkpoint_path:
                 print(f"[!] Best model saved in {trainer.logger.log_dir if trainer.logger else checkpoint_path}")

        return metrics, self.ratio_estimator

    def build_ratio_estimator(self):
        """
        Return the trained ratio estimator model instance.

        This instance has the `log_ratio` method available.

        Returns
        -------
        NREModelWrapper
            The trained ratio estimator model instance (bound with parameters).
        """
        assert hasattr(self, "ratio_estimator"), "Model not trained yet. Call train() first."
        return self.ratio_estimator

    @classmethod
    def load_from_checkpoints(
        cls,
        checkpoint: str,
        exmp_input: Any, # Example (theta, x) tuple
        # Allow specifying components used during save, otherwise try defaults/load
        model_class: Optional[nn.Module] = None,
        model_hparams: Optional[dict] = None,
        loss_fn: Optional[Callable] = None,
        embedding_net_class: Optional[nn.Module] = None,
        embedding_hparams: Optional[dict] = None,
        z_score_theta: Optional[bool] = None,
        z_score_x: Optional[bool] = None,
    ) -> Any:
        """
        Load a trained NRE model from saved checkpoints.

        Parameters
        ----------
        checkpoint : str
            Path to the checkpoint directory (containing hparams.json and checkpoints subdir).
        exmp_input : Any
             Example (theta, x) tuple to infer input shapes and initialize model.
        model_class : nn.Module, optional
            The classifier class used during training. If None, attempts to infer or uses RatioEstimatorMLP.
        model_hparams : dict, optional
            Hyperparameters for the classifier. If None, attempts to load from checkpoint or uses defaults.
        loss_fn : Callable, optional
            Loss function used during training. If None, assumes loss_bce_nre.
        embedding_net_class : nn.Module, optional
            The embedding network class used. If None, assumes Identity.
        embedding_hparams : dict, optional
            Hyperparameters for the embedding network. Default is None.
        z_score_theta : bool, optional
            Whether theta was z-scored. If None, attempts to load from checkpoint or assumes True.
        z_score_x : bool, optional
            Whether x was z-scored. If None, attempts to load from checkpoint or assumes True.

        Returns
        -------
        NRE
            An NRE instance with the loaded model.
        """
        print(f"[!] Loading NRE model from checkpoint: {checkpoint}")
        hparam_file = os.path.join(checkpoint, "hparams.json")
        ckpt_dir = os.path.join(checkpoint, "checkpoints")
        if not os.path.exists(hparam_file):
            raise FileNotFoundError(f"Hyperparameter file not found: {hparam_file}")
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_dir}")

        with open(hparam_file, "r") as f:
            saved_hparams = json.load(f)

        # --- Determine configuration --- 
        # Priority: Argument > Saved Hparam > Default
        _loss_fn = loss_fn or jaxili_loss_dict.get(saved_hparams.get("loss_fn"), loss_bce_nre)
        _opt_hparams = saved_hparams.get("optimizer_hparams", {"lr": 1e-4, "optimizer_name": "adam"})
        _seed = saved_hparams.get("seed", 42)

        # Model components require careful reconstruction based on saved hparams
        _wrapper_hparams = saved_hparams.get("model_hparams", {})

        # Classifier
        _classifier_hparams = model_hparams or _wrapper_hparams.get("classifier_hparams") or default_mlp_hparams
        _classifier_class_name = _wrapper_hparams.get("classifier_class") # Assumes class name string was saved
        if model_class is None:
            if _classifier_class_name:
                 # Attempt to find class by name (requires registration or specific imports)
                 # For simplicity, assume known classes or require user to provide it
                 if _classifier_class_name == "RatioEstimatorMLP":
                     _model_class = RatioEstimatorMLP
                 # Add other known models here... 
                 else:
                     # Try finding in jaxili.model or flax.linen? Risky.
                     # raise ValueError(f"Unknown classifier class '{_classifier_class_name}' in checkpoint. Please provide `model_class`.")
                     print(f"Warning: Unknown classifier class '{_classifier_class_name}' in checkpoint. Using default RatioEstimatorMLP.")
                     _model_class = RatioEstimatorMLP
            else:
                _model_class = RatioEstimatorMLP # Fallback to default
        else:
            _model_class = model_class

        # Embedding Net
        _embedding_hparams = embedding_hparams or _wrapper_hparams.get("embedding_hparams")
        _embedding_class_name = _wrapper_hparams.get("embedding_net_class") # Assume saved name
        if embedding_net_class is None:
            if _embedding_class_name:
                if _embedding_class_name == "Identity":
                    _embedding_net_class = Identity
                # Add other known embedding nets... 
                else:
                     print(f"Warning: Unknown embedding net class '{_embedding_class_name}'. Using default Identity.")
                     _embedding_net_class = Identity
            else:
                _embedding_net_class = Identity # Fallback to default
        else:
            _embedding_net_class = embedding_net_class

        # Z-scoring (assuming saved as simple boolean flags in wrapper_hparams or top level)
        _z_score_theta = z_score_theta if z_score_theta is not None else _wrapper_hparams.get("z_score_theta", saved_hparams.get("z_score_theta", True))
        _z_score_x = z_score_x if z_score_x is not None else _wrapper_hparams.get("z_score_x", saved_hparams.get("z_score_x", True))

        # --- Create and configure NRE instance --- 
        inference = cls(model_class=_model_class, model_hparams=_classifier_hparams, loss_fn=_loss_fn)

        exmp_theta, exmp_x = exmp_input
        inference._dim_params = exmp_theta.shape[-1]
        inference._dim_cond = exmp_x.shape[-1]

        # Build components (standardizers need fitting if loading state)
        inference._build_neural_network(
            z_score_theta=_z_score_theta,
            z_score_x=_z_score_x,
            embedding_net=_embedding_net_class,
            embedding_hparams=_embedding_hparams,
            force_rebuild=True # Ensure components match loaded config
        )

        # Load standardizer states if they were used and saved
        # Assumes standardizer state (mean_, std_) might be saved in checkpoint files
        # or maybe need separate saving logic. TrainerModule doesn't handle this.
        # For now, we assume fit happens on provided exmp_input or requires manual loading.
        # Let's fit on exmp_input if standardizers are used
        if isinstance(inference._theta_compressor, Standardizer):
            inference._theta_compressor.fit(exmp_theta) # Fit on example data for shape
            # Ideally, load mean/std from checkpoint here if saved
        if isinstance(inference._x_compressor, Standardizer):
            inference._x_compressor.fit(exmp_x)
            # Ideally, load mean/std from checkpoint here if saved

        # Recreate the exact model wrapper hparams for the Trainer
        model_wrapper_hparams = {
            'theta_compressor': inference._theta_compressor, # Instance from build_nn
            'x_compressor': inference._x_compressor,
            'embedding_net': inference._embedding_net,
            'classifier_class': _model_class, # The class
            'classifier_hparams': _classifier_hparams,
            # Pass z_score info if needed by wrapper/trainer logic (e.g. for saving)
            'z_score_theta': _z_score_theta,
            'z_score_x': _z_score_x,
            'embedding_net_class': _embedding_net_class.__name__, # Save class names
             'embedding_hparams': _embedding_hparams,
        }

        # Create trainer instance to load state
        inference.trainer = TrainerModule(
            model_class=NREModelWrapper,
            model_hparams=model_wrapper_hparams,
            loss_fn=_loss_fn,
            optimizer_hparams=_opt_hparams,
            exmp_input=exmp_input, # Crucial for shape inference before loading
            logger_params={"base_log_dir": checkpoint}, # Point to loaded dir
            nde_class="NRE",
            seed=_seed,
        )

        # Load the saved model state into the trainer
        inference.trainer.load_model(ckpt_path=ckpt_dir)

        # Bind the loaded model state
        inference.ratio_estimator = inference.trainer.bind_model()
        print(f"[!] NRE model loaded successfully from {checkpoint}.")

        return inference 