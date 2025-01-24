import warnings
from typing import Any, Callable, Dict, Optional, Union, Iterable
from jaxtyping import Array, Float, PyTree

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import distrax

import jaxili
from jaxili.utils import (
    check_density_estimator,
    validate_theta_x,
    create_data_loader,
    numpy_collate,
)
from jaxili.model import (
    ConditionalMAF,
    ConditionalRealNVP,
    MixtureDensityNetwork,
    Identity,
    Standardizer,
    NDE_w_Standardization,
)
from jaxili.train import TrainerModule
from jaxili.loss import loss_nll_nle
from jaxili.utils import *
from jaxili.inference.npe import NDEDataset

import torch.utils.data as data

default_maf_hparams = {
    "n_layers": 5,
    "layers": [50, 50],
    "activation": jax.nn.relu,
    "use_reverse": True,
    "seed": 42
}


class NLE:

    def __init__(
        self,
        model_class: jaxili.model.NDENetwork = ConditionalMAF,
        logging_level: Union[int, str] = "WARNING",
        verbose: bool = True,
        model_hparams: Optional[Dict[str, Any]] = default_maf_hparams,
        loss_fn: Callable = loss_nll_nle,
    ):
        """
        Base class for Neural Likelihood Estimation (NLE) methods.

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
        Sets the dataset to use for training, validation or testing.

        Parameters
        ----------
        dataset : data.Dataset
            Dataset to use.
        type : str
            Type of the dataset. Can be 'train', 'val' or 'test'.
        """
        assert type in ["train", "val", "test"], "Type should be 'train', 'val' or 'test'."

        if type == "train":
            self._train_dataset = dataset
        elif type == "val":
            self._val_dataset = dataset
        elif type == "test":
            self._test_dataset = dataset
    
    def set_dataloader(self, dataloader, type):
        """
        Sets the dataloader to use for training, validation or testing.

        Parameters
        ----------
        dataloader : data.DataLoader
            dataloader to use.
        type : str
            Type of the dataloader. Can be 'train', 'val' or 'test'.
        """
        assert type in ["train", "val", "test"], "Type should be 'train', 'val' or 'test'."

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

        Parameter
        ---------
        theta : Array
            Parameters of the simulations.
        x : Array
            Simulation outputs.
        train_test_split : Iterable[float], optional
            Fractions to split the dataset into training, validation and test sets.
            Should be of length 2 or 3. A length 2 list will not generate a test set. Default is [0.7, 0.2, 0.1].
        """
        # Verify theta and x typing and size of the dataset
        theta, x, num_sims = validate_theta_x(theta, x)
        if self.verbose:
            print(f"[!] Inputs are valid.")
            print(f"[!] Appending {num_sims} simulations to the dataset.")

        self._dim_params = x.shape[1] #The distribution of the data is learned.
        self._dim_cond = theta.shape[1] #Conditionned on the parameters
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
        self.set_dataset(NDEDataset(theta[train_idx], x[train_idx]), type="train")
        self.set_dataset(NDEDataset(theta[val_idx], x[val_idx]), type="val")
        self.set_dataset(
            NDEDataset(theta[test_idx], x[test_idx]) if is_test_set else None,
            type="test"
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
        Creates DataLoaders for the training, validation and test datasets. Can only be executed after appending simulations.

        Parameters
        ----------
        batch_size : int
            Batch size to use for the DataLoader. Default is 128.
        num_workers : int
            Number of workers to use for the DataLoader. Default is 4.
        seed : int
            Seed to use for the DataLoader. Default is 42.
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
                    **kwargs,
                )
            )

    def _build_neural_network(
        self,
        z_score_theta: bool = True,
        z_score_x: bool = True,
        embedding_net: nn.Module = Identity(),
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
            warnings.warn(f"Model class {self.model_class} is not a base class of JaxILI.\n Check that the hyperparameters of your network are consistent.", Warning)

        try:
            self._train_dataset
        except AttributeError:
            raise ValueError(
                "No training dataset found. Please append simulations first."
            )

        # Check if z-score is required for theta.
        if z_score_theta:
            shift = jnp.mean(self._train_dataset.theta, axis=0)
            scale = jnp.std(self._train_dataset.theta, axis=0)
            standardizer = Standardizer(shift, scale)
        else:
            standardizer = Identity()

        self._embedding_net = nn.Sequential([standardizer, embedding_net])

        # Check if z-score is required for x.
        shift = jnp.zeros(self._dim_params)
        scale = jnp.ones(self._dim_params)
        
        if z_score_x:
            shift = jnp.mean(self._train_dataset.x, axis=0)
            scale = jnp.std(self._train_dataset.x, axis=0)
            
        self._transformation = distrax.ScalarAffine(scale=scale, shift=shift)

        self._model_hparams["n_in"] = self._dim_params
        self._model_hparams["n_cond"] = self._dim_cond
        self._nde = self._model_class(**self._model_hparams)

        model = NDE_w_Standardization(
            nde=self._nde,
            embedding_net=self._embedding_net,
            transformation=self._transformation,
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
        exmp_input : Any
            Example input to use for the model.
        seed : int, optional
            Seed to use for the trainer. Default is 42.
        logger_params : Dict[str, Any], optional
            Parameters to use for the logger. Default is None.
        debug : bool, optional
            Whether to use debug mode. Default is False.
        check_val_every_epoch : int, optional
            Frequency at which to check the validation loss. Default is 1.
        """

        try:
            self._nde
        except AttributeError:
            z_score_theta = kwargs.get("z_score_theta", True)
            z_score_x = kwargs.get("z_score_x", True)
            embedding_net = kwargs.get("embedding_net", Identity())
            _ = self._build_neural_network(
                z_score_theta=z_score_theta,
                z_score_x=z_score_x,
                embedding_net=embedding_net,
            )

        nde_w_std_hparams = {
            "nde": self._nde,
            "embedding_net": self._embedding_net,
            "transformation": self._transformation
        }

        exmp_input = (jnp.zeros((1, self._dim_cond)), jnp.zeros((1, self._dim_params))) #Example will be reversed in the trainer.

        if self.verbose:
            print("[!] Creating the Trainer module.")

        self.trainer = TrainerModule(
            model_class=NDE_w_Standardization,
            model_hparams=nde_w_std_hparams,
            optimizer_hparams=optimizer_hparams,
            loss_fn=self._loss_fn,
            exmp_input=exmp_input,
            seed=seed,
            logger_params=logger_params,
            enable_progress_bar=self.verbose,
            debug=debug,
            check_val_every_epoch=check_val_every_epoch,
            nde_class="NLE"
        )

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
        """

        try:
            self._train_dataset
        except AttributeError:
            raise ValueError(
                "No training dataset found. Please append simulations first."
            )

        # Create the dataloaders to perform the training
        try:
            self._train_loader
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

            _ = self.create_trainer(
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

        density_estimator = self.trainer.bind_model()
        return metrics, density_estimator
