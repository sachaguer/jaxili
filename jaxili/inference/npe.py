"""
NPE.

This module provides the Neural Posterior Estimation (NPE) class to train a neural density estimator to perform NPE.
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
import flax.nnx as nnx
import numpy as np
from jaxtyping import Array, Float, PyTree

import jaxili
from jaxili.loss import loss_nll_npe
from jaxili.model import (
    ConditionalMAF,
    ConditionalRealNVP,
    MixtureDensityNetwork,
    NDE_w_Standardization,
)
from jaxili.compressor import Identity, Standardizer
from jaxili.posterior import DirectPosterior
from jaxili.train import TrainerModule
from jaxili.utils import *
from jaxili.utils import (
    check_density_estimator,
    create_data_loader,
    validate_theta_x,
)
from jaxili.inventory.func_dict import jaxili_loss_dict, jax_nn_dict, jaxili_nn_dict

import datasets as hf_datasets

default_maf_hparams = {
    "n_layers": 5,
    "layers": [50, 50],
    "activation": jax.nn.relu,
    "use_reverse": True,
    "seed": 42,
    "rngs": nnx.Rngs(0),
}


class NPE:
    """
    NPE.

    Base class for Neural Posterior Estimation (NPE) methods.
    Default configuration used a `ConditionalMAF` to learn the posterior function.

    Examples
    --------
    >>> from jaxili.inference import NPE
    >>> inference = NPE()
    >>> theta, x = ...  # Load parameters and simulation outputs
    >>> inference.append_simulations(theta, x) #Push your simulations in the trainer
    >>> inference.train() #Train your density estimator
    """

    def __init__(
        self,
        model_class: jaxili.model.NDENetwork = ConditionalMAF,
        logging_level: Union[int, str] = "WARNING",
        verbose: bool = True,
        model_hparams: Optional[Dict[str, Any]] = default_maf_hparams,
        loss_fn: Callable = loss_nll_npe,
    ):
        """
        Initialize class for Neural Posterior Estimation (NPE) methods.

        Parameters
        ----------
        model_class : jaxili.model.NDENetwork
            Class of the neural density estimator to use. Default: ConditionalMAF.
        model_hparams : Dict[str, Any]
            Hyperparameters to use for the model.
        logging_level: Union[int, str], optional
            Logging level to use. Default is "WARNING".
        show_progress_bar : bool, optional
            Whether to show a progress bar during training. Default is True.
        """
        self._model_class = model_class
        self._model_hparams = model_hparams
        self._logging_level = logging_level
        self._loss_fn = loss_fn
        self.verbose = verbose

    def set_model_hparams(self, hparams):
        """
        Set the hyperparameters of the model.

        Parameters
        ----------
        hparams : Dict[str, Any]
            Hyperparameters to use for the model.
        """
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
        # Verify theta and x typing and size of the dataset
        theta, x, num_sims = validate_theta_x(theta, x)
        if self.verbose:
            print(f"[!] Inputs are valid.")
            print(f"[!] Appending {num_sims} simulations to the dataset.")

        self._dim_params = theta.shape[1]
        self._dim_cond = x.shape[1]
        self._num_sims = num_sims

        # Split the dataset into training, validation and test sets
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

    def append_simulations_huggingface(
        self,
        hf_dataset: hf_datasets.Dataset,
        train_test_split: Iterable[float] = [0.7, 0.2, 0.1],
        key: Optional[PyTree] = None,
    ):
        """
        Store parameters and simulation outputs to use them for later training.

        Data is stored in a Dataset object from `jax-dataloader`

        Parameters
        ----------
        hf_dataset : hf_datasets.Dataset
            HuggingFace dataset to split and add to the inference object.
        train_test_split : Iterable[float], optional
            Fractions to split the dataset into training, validation and test sets.
            Should be of length 2 or 3. A length 2 list will not generate a test set. Default is [0.7, 0.2, 0.1].
        key : PyTree, optional
            Key to use for the random permutation of the dataset. Default is None.
        """
        # check if the hugging face dataset has the correct form
        if ("x" not in hf_dataset.features.keys()) or (
            "theta" not in hf_dataset.features.keys()
        ):
            raise ValueError(
                "The hugging face dataset should have columns 'theta' and 'x'"
            )

        theta, x = hf_dataset[0]["theta"], hf_dataset[0]["x"]
        # theta, x, _ = validate_theta_x(theta, x)
        num_sims = hf_dataset.num_rows
        if self.verbose:
            print(f"[!] Inputs are valid.")
            print(f"[!] Appending {num_sims} simulations to the dataset.")
        self._dim_params = len(theta)
        self._dim_cond = len(x)
        self._num_sims = num_sims

        # Split the dataset into training, validation and test sets
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

        if not is_test_set:
            test_fraction = 0.0
        hf_dataset = hf_dataset.train_test_split(test_size=val_fraction + test_fraction)
        if is_test_set:
            temp_dataset = hf_dataset["test"].train_test_split(
                test_size=val_fraction / (val_fraction + test_fraction)
            )
            hf_dataset["val"] = temp_dataset["train"]
            hf_dataset["test"] = temp_dataset["test"]
            del temp_dataset
        self.set_dataset(hf_dataset["train"], type="train")
        self.set_dataset(hf_dataset["test"], type="val")
        self.set_dataset(
            hf_dataset["val"] if is_test_set else None,
            type="test",
        )

        if self.verbose:
            print(f"[!] Dataset split into training, validation and test sets.")
            print(f"[!] Training set: {hf_dataset["train"].num_rows} simulations.")
            print(f"[!] Validation set: {hf_dataset["test"].num_rows} simulations.")
            if is_test_set:
                print(f"[!] Test set: {hf_dataset["val"].num_rows} simulations.")
        return self

    def _create_data_loader(self, **kwargs):
        """
        Create DataLoaders for the training, validation and test datasets. Can only be executed after appending simulations.

        Parameters
        ----------
        batch_size : int
            Batch size to use for the DataLoader. Default is 128.
        """
        try:
            self._train_dataset
        except AttributeError:
            raise ValueError(
                "No training dataset found. Please append simulations first."
            )
        try:
            self._val_dataset
        except AttributeError:
            raise ValueError(
                "No validation dataset found. Please append simulations first."
            )

        train = [True, False] if self._test_dataset is None else [True, False, False]
        batch_size = kwargs.get("batch_size", 128)
        if self.verbose:
            print(f"[!] Creating DataLoaders with batch_size {batch_size}.")
        if self._test_dataset is None:
            self._train_loader, self._val_loader = create_data_loader(
                self._train_dataset, self._val_dataset, train=train, **kwargs
            )
            self._test_loader = None
        else:
            self._train_loader, self._val_loader, self._test_loader = (
                create_data_loader(
                    self._train_dataset,
                    self._val_dataset,
                    self._test_dataset,
                    train=train,
                    batch_size=batch_size,
                )
            )

    def _build_neural_network(
        self,
        z_score_theta: bool = True,
        z_score_x: bool = True,
        embedding_net: nn.Module = Identity,
        embedding_hparams: dict = None,
        **kwargs,
    ):
        """
        Build the neural network for the density estimator.

        Parameters
        ----------
        z_score_theta : bool, optional
            Whether to z-score the parameters. Default is True.
        z_score_x : bool, optional
            Whether to z-score the simulation outputs. Default is True.
        embedding_net : nn.Module, optional
            Neural network to use for embedding. Default is nn.Identity().
        """
        if self.verbose:
            print("[!] Building the neural network.")

        # Check if the model class and hparams are correct
        if self._model_class == ConditionalMAF:
            check_hparams_maf(self._model_hparams)
        elif self._model_class == ConditionalRealNVP:
            check_hparams_realnvp(self._model_hparams)
        elif self._model_class == MixtureDensityNetwork:
            check_hparams_mdn(self._model_hparams)
        else:
            warnings.warn(
                f"Model class {self.model_class} is not a base class of JaxILI.\n Check that the hyperparameters of your network are consistent.",
                Warning,
            )

        try:
            self._train_dataset
        except AttributeError:
            raise ValueError(
                "No training dataset found. Please append simulations first."
            )

        # Check if z-score is required for theta.
        shift = jnp.zeros(self._dim_params)
        scale = jnp.ones(self._dim_params)

        if z_score_theta:
            if isinstance(self._train_dataset, jdl.ArrayDataset):
                train_dataset_theta = self._train_dataset[:][0]
            elif isinstance(self._train_dataset, hf_datasets.Dataset):
                train_dataset_theta = jnp.array(self._train_dataset["theta"])
            else:
                raise TypeError(
                    "Only JAX or HuggingFace datasets are currently supported."
                )

            shift = jnp.mean(train_dataset_theta, axis=0)
            scale = jnp.std(train_dataset_theta, axis=0)
            del train_dataset_theta
            min_std = kwargs.get("min_std", 1e-14)
            scale = scale.at[scale < min_std].set(min_std)

        self._transformation_hparams = {"shift": shift, "scale": scale}

        self._transformation = distrax.ScalarAffine(scale=scale, shift=shift)

        # Check if z-score is required for x.
        if z_score_x:
            if isinstance(self._train_dataset, jdl.ArrayDataset):
                train_dataset_x = self._train_dataset[:][1]
            elif isinstance(self._train_dataset, hf_datasets.Dataset):
                train_dataset_x = jnp.array(self._train_dataset["x"])
            else:
                raise TypeError(
                    "Only JAX or HuggingFace datasets are currently supported."
                )
            shift = jnp.mean(train_dataset_x, axis=0)
            scale = jnp.std(train_dataset_x, axis=0)
            min_std = kwargs.get("min_std", 1e-14)
            del train_dataset_x
            scale = scale.at[scale < min_std].set(min_std)
            standardizer = Standardizer(shift, scale)
            self._standardizer_hparams = {}
            self._standardizer_hparams["shift"] = shift
            self._standardizer_hparams["scale"] = scale
        else:
            standardizer = Identity()

        if embedding_net == Identity:
            embedding_net = Identity()
        else:
            if embedding_hparams is None:
                warnings.warn(
                    "An embedding net has been specified but not its hyperparameters. Creating an embedding of the instance `Identity` instead."
                )
                embedding_net = Identity()
            else:
                embedding_net = embedding_net(**embedding_hparams)

        self._embedding_net = nnx.Sequential(standardizer, embedding_net)

        if isinstance(embedding_net, Identity):
            n_cond = self._dim_cond
        else:
            n_cond = embedding_net.output_size

        self._model_hparams["n_in"] = self._dim_params
        self._model_hparams["n_cond"] = n_cond
        self._nde = self._model_class(**self._model_hparams)

        model = NDE_w_Standardization(
            nde=self._nde,
            embedding_net=self._embedding_net,
            shift_transformation=self._transformation_hparams["shift"],
            scale_transformation=self._transformation_hparams["scale"],
        )

        return model

    def create_trainer(
        self,
        optimizer_hparams: Dict[str, Any],
        seed: int = 42,
        logger_params: Dict[str, Any] = None,
        debug: bool = False,
        check_val_every_epoch: int = 1,
        **kwargs,
    ):
        """
        Create a TrainerModule for the density estimator.

        Parameters
        ----------
        optimizer_hparams : Dict[str, Any]
            Hyperparameters to use for the optimizer.
        loss_fn : Callable
            Loss function to use for training.
        seed : int, optional
            Seed to use for the trainer. Default is 42.
        logger_params : Dict[str, Any], optional
            Parameters to use for the logger. Default is None.
        debug : bool, optional
            Whether to use debug mode. Default is False.
        check_val_every_epoch : int, optional
            Frequency at which to check the validation loss. Default is 1.
        """
        z_score_theta = kwargs.get("z_score_theta", True)
        z_score_x = kwargs.get("z_score_x", True)
        embedding_net = kwargs.get("embedding_net", Identity)
        embedding_net_hparams = kwargs.get("embedding_hparams", None)
        try:
            self._nde
        except AttributeError:
            _ = self._build_neural_network(
                z_score_theta=z_score_theta,
                z_score_x=z_score_x,
                embedding_net=embedding_net,
                embedding_hparams=embedding_net_hparams,
            )

        nde_w_std_hparams = {
            "nde": self._nde,
            "embedding_net": self._embedding_net,
            "shift_transformation": self._transformation_hparams["shift"],
            "scale_transformation": self._transformation_hparams["scale"],
        }

        if self.verbose:
            print("[!] Creating the Trainer module.")

        self.trainer = TrainerModule(
            model_class=NDE_w_Standardization,
            model_hparams=nde_w_std_hparams,
            optimizer_hparams=optimizer_hparams,
            loss_fn=self._loss_fn,
            seed=seed,
            logger_params=logger_params,
            enable_progress_bar=self.verbose,
            debug=debug,
            check_val_every_epoch=check_val_every_epoch,
        )
        self.trainer.config.update({"nde_class": self._model_class.__name__})
        self.trainer.config.update({"nde_hparams": copy.deepcopy(self._model_hparams)})
        # Check if there is an activation function to rename
        if "activation" in self._model_hparams.keys():
            self.trainer.config["nde_hparams"]["activation"] = self.trainer.config[
                "nde_hparams"
            ]["activation"].__name__
        self.trainer.config.update(
            {"transformation_hparams": copy.deepcopy(self._transformation_hparams)}
        )
        if z_score_x:
            self.trainer.config.update(
                {"standardizer_hparams": copy.deepcopy(self._standardizer_hparams)}
            )
        self.trainer.config.update({"embedding_class": embedding_net.__name__})
        if embedding_net_hparams is not None:
            self.trainer.config.update(
                {"embedding_hparams": copy.deepcopy(embedding_net_hparams)}
            )
            # Check if there is an activation function to rename
            if "activation" in embedding_net_hparams.keys():
                self.trainer.config["embedding_hparams"]["activation"] = (
                    self.trainer.config["embedding_hparams"]["activation"].__name__
                )
        self.trainer.write_config(self.trainer.log_dir)

    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        patience: int = 20,
        num_epochs: int = 2**31 - 1,
        check_val_every_epoch: int = 1,
        **kwargs,
    ):
        r"""
        Train the density estimator to approximate the distribution $p(\theta|x)$.

        Parameters
        ----------
        training_batch_size : int, optional
            Batch size to use during training. Default is 50.
        learning_rate: float, optional
            Learning rate to use during training. Default is 5e-4.
        patience: int, optional
            Number of epochs to wait before early stopping. Default is 20.
        num_epochs: int, optional
            Maximum number of epochs to train. Default is 2**31 - 1.
        check_val_every_epoch: int, optional
            Frequency at which to check the validation loss. Default is 1.
        **kwargs : dict, optional
            Additional keyword arguments for training customization:

            - optimizer_name (str): Name of the optimizer to use (default: 'adam').
            - gradient_clip (float): Value for gradient clipping (default: 5.0).
            - warmup (float): Warmup proportion for learning rate scheduling (default: 0.1).
            - weight_decay (float): Weight decay (L2 regularization) (default: 0.0).
            - checkpoint_path (str): Directory to save training checkpoints (default: 'checkpoints/').
            - log_dir (str or None): Directory for logging (default: None).
            - logger_type (str): Type of logger to use (default: 'TensorBoard').
            - seed (int): Random seed for reproducibility (default: 42).
            - debug (bool): Whether to run in debug mode (default: False).
            - min_delta (float): Minimum change in validation loss to qualify as improvement (default: 1e-3).


        Returns
        -------
        metrics : Dict[str, float]
            Dictionary containing the training, validation and test losses.
        density_estimator : nn.Module
            The trained density estimator.
        """
        try:
            self._train_dataset
        except AttributeError:
            raise ValueError(
                "No training dataset found. Please append simulations first."
            )

        # Create the dataloaders to perform the training
        try:
            batch_size = self._train_loader.dataloader.batch_size
            if batch_size != training_batch_size:
                self._create_data_loader(batch_size=training_batch_size)
        except AttributeError:
            self._create_data_loader(batch_size=training_batch_size)

        try:
            metrics = self.trainer.train_model(
                self._train_loader,
                self._val_loader,
                test_loader=self._test_loader,
                num_epochs=num_epochs,
                patience=patience,
                **kwargs,
            )
        except AttributeError:
            test_optimizer_hparams = kwargs.get("optimizer_hparams", None)
            if test_optimizer_hparams is not None:
                warnings.warn(
                    "The optimizer hyperparameters specified will not be taken into account. Please refer to the documentation to modify it. Falling back to default optimizer hyperparameters."
                )
            optimizer_hparams = {
                "lr": learning_rate,
                "optimizer_name": kwargs.get("optimizer_name", "adam"),
                "gradient_clip": kwargs.get("gradient_clip", 5.0),
                "warmup": kwargs.get("warmup", 0.1),
                "weight_decay": kwargs.get("weight_decay", 0.0),
            }

            logger_params = {
                "base_log_dir": kwargs.get("checkpoint_path", "checkpoints/"),
                "log_dir": kwargs.get("log_dir", None),
                "logger_type": kwargs.get("logger_type", "TensorBoard"),
            }

            self.create_trainer(
                optimizer_hparams=optimizer_hparams,
                seed=kwargs.get("seed", 42),
                logger_params=logger_params,
                debug=kwargs.get("debug", False),
                check_val_every_epoch=check_val_every_epoch,
                **kwargs,
            )

            if self.verbose:
                print("[!] Training the density estimator.")
            metrics = self.trainer.train_model(
                self._train_loader,
                self._val_loader,
                test_loader=self._test_loader,
                num_epochs=num_epochs,
                patience=patience,
                min_delta=kwargs.get("min_delta", 1e-3),
            )

        if self.verbose:
            print(f"[!] Training loss: {metrics['train/loss']}")
            print(f"[!] Validation loss: {metrics['val/loss']}")
            if self._test_loader is not None:
                print(f"[!] Test loss: {metrics['test/loss']}")

        density_estimator = self.trainer.model
        return metrics, density_estimator

    def build_posterior(
        self, verbose: Optional[bool] = None, x: Optional[Array] = None
    ):
        r"""
        Build the posterior distribution $p(\theta|x)$ using the trained density estimator.

        Parameters
        ----------
        x : Array, optional
            The data used to condition the posterior. Default is None.

        Returns
        -------
        posterior : NeuralPosterior
            The posterior distribution allowing to sample and evaluate the unnormalized log-probability.
        """
        try:
            self.trainer
        except AttributeError:
            raise ValueError("No trainer found. You must first create a trainer.")

        if verbose is None:
            verbose = self.verbose
        posterior = DirectPosterior(model=self.trainer.model, verbose=verbose, x=x)

        if self.verbose:
            print(
                r"[!] Posterior $p(\theta| x)$ built. The class DirectPosterior is used to sample and evaluate the log probability."
            )

        return posterior

    @classmethod
    def load_from_checkpoints(
        cls, checkpoint: str, embedding_net_class=Identity
    ) -> Any:
        """
        Create a NPE object where the TrainerModule is loading the already existing weights for the neural network.

        Parameters
        ----------
        nde_class: NDENetwork
            Class used to create the neural density estimator.
        checkpoint: str
            Folder in which the checkpoint and hyperparameter file is stored.
        embedding_net_class: nn.Module
            Class used to create the embedding net. (Default: Identity)

        Returns
        -------
        A NPE object containing a model with the pre-trained weights loaded.
        """
        hparams_file = os.path.join(checkpoint, "hparams.json")
        assert os.path.isfile(hparams_file), "Could not find hparams file."
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        assert (
            hparams["model_class"] == NDE_w_Standardization.__name__
        ), "The model has not been trained with NDE_w_Standardization. Check the checkpoint path is correct."
        hparams.pop("model_class")

        embedding_class = hparams["embedding_class"]
        assert (
            embedding_class == embedding_net_class.__name__
        ), "The embedding class does not match. Check that you are using the correct architecture."

        # Check if the loss function is correct.
        assert (
            hparams["loss_fn"] in jaxili_loss_dict
        ), "Unknown loss function. Check that the loss function you used comes from `jax.nn`."
        hparams["loss_fn"] = jaxili_loss_dict[hparams["loss_fn"]]
        # Create the NDE
        # Extract the nde string
        nde_class_name = hparams["nde_class"]
        hparams.pop("nde_class")

        nde_class = jaxili_nn_dict[nde_class_name]
        nde_hparams = hparams["nde_hparams"]
        if "activation" in nde_hparams.keys():
            nde_hparams["activation"] = jax_nn_dict[nde_hparams["activation"]]
        nde_hparams["rngs"] = nnx.Rngs(0)

        # Create object from the class NPE
        inference = cls(
            model_class=nde_class, model_hparams=nde_hparams, loss_fn=hparams["loss_fn"]
        )

        # Create the NDE
        inference._nde = nde_class(**nde_hparams)

        # Regenerate the embedding net
        standardizer_hparams = hparams.get("standardizer_hparams", None)
        if standardizer_hparams is None:
            standardizer = Identity()
        else:
            shift_str = standardizer_hparams["shift"]
            shift_list = [float(x) for x in shift_str.strip("[]").split()]

            scale_str = standardizer_hparams["scale"]
            scale_list = [float(x) for x in scale_str.strip("[]").split()]

            standardizer = Standardizer(
                mean=np.array(shift_list), std=np.array(scale_list)
            )

        if embedding_class != "Identity":
            if "embedding_hparams" not in hparams.keys():
                raise ValueError(
                    "The embedding net hyperparameters can't be find. Check that you are using the correct checkpoint path."
                )
            if "activation" in hparams["embedding_hparams"].keys():
                hparams["embedding_hparams"]["activation"] = jax_nn_dict[
                    hparams["embedding_hparams"]["activation"]
                ]
            embedding_net = embedding_net_class(**hparams["embedding_hparams"])
        else:
            embedding_net = Identity()

        inference._embedding_net = nnx.Sequential(standardizer, embedding_net)

        # Regenerate the transformation of the parameters
        shift_str = hparams["transformation_hparams"]["shift"]
        shift_list = [float(x) for x in shift_str.strip("[]").split()]

        scale_str = hparams["transformation_hparams"]["scale"]
        scale_list = [float(x) for x in scale_str.strip("[]").split()]

        inference._transformation = distrax.ScalarAffine(
            np.array(shift_list), np.array(scale_list)
        )

        model_hparams = {
            "nde": inference._nde,
            "embedding_net": inference._embedding_net,
            "shift_transformation": inference._transformation.shift,
            "scale_transformation": inference._transformation.scale,
        }

        if not hparams["logger_params"]:
            hparams["logger_params"] = dict()
        hparams["logger_params"]["log_dir"] = checkpoint
        hparams.pop("model_hparams")

        inference.trainer = TrainerModule(
            model_class=NDE_w_Standardization,
            model_hparams=model_hparams,
            **hparams,
        )

        inference.trainer.load_model()

        return inference
