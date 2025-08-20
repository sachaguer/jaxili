"""
Compressor.

This module contains classes that implement compressors used in JaxILI.
"""

from typing import Any, Callable, Dict, Optional, Union, Iterable
from jaxtyping import Array, Float, PyTree
from jaxili.loss import loss_mse
from jaxili.utils import *
from jaxili.utils import check_density_estimator, create_data_loader, validate_theta_x
from jaxili.train import TrainerModule
from jaxili.inventory.func_dict import jax_nn_dict, jaxili_nn_dict
import jax.random as jr
import jax.numpy as jnp
import jax
import flax.linen as nn
import flax.nnx as nnx
import numpyro.distributions as dist

from jaxili.posterior.mcmc_posterior import nuts_numpyro_kwargs_default

import os
import json
from functools import partial
import datasets as hf_datasets


default_maf_hparams = {
    "n_layers": 5,
    "layers": [50, 50],
    "activation": jax.nn.relu,
    "use_reverse": True,
    "seed": 42,
}


class TrainerCompressor(TrainerModule):
    """
    A module to perform the train a compressor. It inherits from the `TrainerModule` and is adapted for neural network performing compression.

    This module contains the training loop, evaluation, logging, and checkpointing. It can also be used to load a model from a checkpoint.
    """
    def __init__(self, model_class: nn.Module, **kwargs):
        """
        Initialize a basic Trainer module summarizing most training functionalities like logging, model initialization, training loop, etc...

        Attributes
        ----------
        model_class : nn.Module
            The class of the model that should be trained.
        model_hparams : Dict[str, Any]
            A dictionnary of the hyperparameters of the model. Is used as input to the model when it is created.
        optimizer_hparams : Dict[str, Any]
            A dictionnary of the hyperparameters of the optimizer. Used during initialization of the optimizer.
        exmp_input : Any
            Input to the model for initialisation and tabulate.
        seed : int
            Seed to initialise PRNG.
        logger_params : Dict[str, Any]
            A dictionary containing the specifications of the logger.
        enable_progress_bar : bool
            Whether to enable the progress bar. Default is True.
        debug : bool
            If True, no jitting is applied. Can be helpful for debugging. Default is False.
        check_val_every_epoch : int
            How often to check the validation set. Default is 1.
        """
        super().__init__(model_class, **kwargs)

    def handle_hf_dataset(self, batch: Dict) -> Union[jnp.ndarray, jnp.ndarray]:
        """
        Handle the dataset and return the input and target.

        Parameters
        ----------
        batch : dict
            The batch of data.

        Returns
        -------
        x : jnp.ndarray
            The input data.
        theta : jnp.ndarray
            The target data.
        """
        return batch["theta"], batch["x"]

    @classmethod
    def load_from_checkpoints(
        cls,
        model_class: nn.Module,
        checkpoint: str,
        loss_function: Callable,
    ) -> Any:
        """
        Create a Trainer object with same hyperparameters and loaded model from a checkpoint directory.

        Parameters
        ----------
        model_class : nn.Module
            The class of the model that should be loaded.
        checkpoint : str
            Folder in which the checkpoint and hyperparameter file is stored
        exmp_input : Any
            An input to the model with which the shapes are inferred.
        loss_function : Callable
            The loss function used for training.

        Returns
        -------
        A Trainer object with model loaded from the checkpoint folder.
        """
        hparams_file = os.path.join(checkpoint, "hparams.json")
        assert os.path.isfile(hparams_file), "Could not find hparams file."
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        assert (
            hparams["model_class"] == model_class.__name__
        ), "Model class does not match. Check that you are using the correct architecture."
        hparams.pop("model_class")
        hparams.pop("loss_fn")
        if "activation" in hparams["model_hparams"].keys():
            hparams["model_hparams"]["activation"] = jax_nn_dict[
                hparams["model_hparams"]["activation"]
            ]
        if "nde" in hparams["model_hparams"].keys():
            hparams["model_hparams"]["nde"] = jaxili_nn_dict[
                hparams["model_hparams"]["nde"]
            ]
        if not hparams["logger_params"]:
            hparams["logger_params"] = dict()
        hparams["logger_params"]["log_dir"] = checkpoint
        trainer = cls(model_class=model_class, loss_fn=loss_function, **hparams)
        trainer.load_model()
        return trainer


class Identity(nnx.Module):
    """Identity transformation."""

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the identity transformation.

        Parameters
        ----------
        x : jnp.Array
            Input data.

        Returns
        -------
        jnp.Array
            Output data.
        """
        return x


class Standardizer(nnx.Module):
    r"""
    Standardizer class making a Z-score standardization given some mean and standard deviation.

    The transformation applied to the input vector $x$ is:

    $$
    x \longrightarrow \frac{x- \langle x \rangle}{\sigma_x}.
    $$
    """

    def __init__(self, mean: Array, std: Array):
        """
        Z-score standardizer.

        Parameters
        ----------
        mean : jnp.Array
            Mean to be substracted during the standardization.
        std : jnp.Array
            Standard deviation used during the standardization
        """
        self.mean = nnx.Variable(mean)
        self.std = nnx.Variable(std)

    def __call__(self, x):
        """
        Forward pass of the standardizer transformation. The standardization uses the z-score.

        Parameters
        ----------
        x : jnp.Array
            Input data.

        Returns
        -------
        jnp.Array
            Standardized data.
        """
        return (x - self.mean) / self.std


class MLPCompressor(nnx.Module):
    """
    Base class of a MLP Compressor.

    Defines a MLP compressor to send the summary statistic to the same dimension than the parameters.

    Parameters
    ----------
    hidden_size : list
        List with the size of the hidden layers.
    activation : Callable
        Activation function. Preferably from `jax.nn` or `jax.nn.activation`.
    output_size : int
        Size of the output layer.
    rngs :
        RNG keys
    """

    def __init__(
        self,
        hidden_size: list,
        activation: Callable,
        input_size: int,
        output_size: int,
        rngs: Array,
    ):
        self.hidden_size = hidden_size
        self.activation = activation
        self.output_size = output_size

        self.layers = []
        # Create first layer
        if len(hidden_size) > 0:
            self.layers.append(
                nnx.Linear(input_size, hidden_size[0], rngs=rngs)
            )  # Append the first layer.
            for in_features, out_features in zip(hidden_size[:-1], hidden_size[1:]):
                self.layers.append(
                    nnx.Linear(in_features, out_features, rngs=rngs)
                )  # Append intermediate layers.
            self.layers.append(
                nnx.Linear(out_features, output_size, rngs=rngs)
            )  # Append last layer.
        else:
            self.layers.append(
                nnx.Linear(input_size, output_size, rngs=rngs)
            )  # Only one layer is appened.

    def __call__(self, x):
        """
        Forward pass of the MLP Compressor.

        Parameters
        ----------
        x : jnp.array
            Input data.

        Returns
        -------
        jnp.array
            Compressed data.
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class CNN2DCompressor(nnx.Module):
    """
    Base class of a CNN2D Compressor.

    Defines a 2 dimensional Convolutional Neural Network to compress the data to the same dimension as the parameters.

    Parameters
    ----------
    output_size : int
        Size of the output layer
    activation : Callable
        Activation function. Preferably from `jax.nn` or `jax.nn.activation`.
    """

    def __init__(
        self, input_size: int, output_size: int, activation: Callable, rngs: nnx.Rngs
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Create the convolutional layers
        self.conv_1 = nnx.Conv(
            self.input_size, 32, kernel_size=(3, 3), strides=2, rngs=rngs
        )
        self.conv_2 = nnx.Conv(32, 64, kernel_size=(3, 3), strides=2, rngs=rngs)
        self.conv_3 = nnx.Conv(64, 128, kernel_size=(3, 3), strides=2, rngs=rngs)

        # Create the pooling layer
        self.avg_pool = partial(
            nnx.avg_pool, window_shape=(16, 16), strides=(8, 8), padding="SAME"
        )

        # Create linear layers
        self.linear_1 = nnx.Linear(
            8192, 64, rngs=rngs
        )  # The size of the flattened array is hardcoded. A solution must be found to account for images of any size
        self.linear_2 = nnx.Linear(64, self.output_size, rngs=rngs)

    def __call__(self, inputs):
        """
        Forward pass of the CNN2D Compressor.

        Parameters
        ----------
        inputs : jnp.array
            Input data.

        Returns
        -------
        jnp.array
            Compressed data.
        """
        net_x = self.activation(self.conv_1(inputs))
        net_x = self.activation(self.conv_2(net_x))
        net_x = self.activation(self.conv_3(net_x))
        net_x = self.avg_pool(net_x)

        # Flatten the tensor
        net_x = net_x.reshape((net_x.shape[0], -1))
        net_x = self.activation(self.linear_1(net_x))
        net_x = self.activation(self.linear_2(net_x))
        return net_x.squeeze()


class Compressor:
    """
    Mean Squared Error (MSE) compressor.

    This class implements a compressor that computes the mean squared error between the input and the target.

    Parameters
    ----------
    model : nn.Module
        The model to be compressed.
    """

    def __init__(
        self,
        model_class: nn.Module = Identity,
        logging_level: Union[int, str] = "WARNING",
        verbose: bool = True,
        model_hparams: Optional[Dict[str, Any]] = default_maf_hparams,
        loss_fn: Callable = loss_mse,
    ):
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

        self._dim_sim = x.shape[1]
        self._dim_params = theta.shape[1]
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

        theta, x = hf_dataset["theta"][0], hf_dataset["x"][0]

        num_sims = hf_dataset.num_rows
        if self.verbose:
            print(f"[!] Inputs are valid.")
            print(f"[!] Appending {num_sims} simulations to the dataset.")
        self._dim_sim = len(x)
        self._dim_params = len(theta)
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
            print(f"[!] Training set: {hf_dataset['train'].num_rows} simulations.")
            print(f"[!] Validation set: {hf_dataset['test'].num_rows} simulations.")
            if is_test_set:
                print(f"[!] Test set: {hf_dataset['val'].num_rows} simulations.")
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
    ):
        """
        Build the neural network for the density estimator.

        Parameters
        ----------
        z_score_theta : bool, optional
            Whether to z-score the parameters. Default is True.
        z_score_x : bool, optional
            Whether to z-score the simulation outputs. Default is True.
        """
        if self.verbose:
            print("[!] Building the neural network.")

        try:
            self._train_dataset
        except AttributeError:
            raise ValueError(
                "No training dataset found. Please append simulations first."
            )
        self._compressor = self._model_class(**self._model_hparams)
        model = self._compressor
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
        Create a `TrainerModule` for the compressor.

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
        try:
            self._compressor
        except AttributeError:
            _ = self._build_neural_network()

        exmp_input = (
            jnp.zeros((1, self._dim_sim)),
            jnp.zeros((1, self._dim_params)),
        )

        if self.verbose:
            print("[!] Creating the Trainer module.")

        self.trainer = TrainerCompressor(
            model_class=self._model_class,
            model_hparams=self._model_hparams,
            optimizer_hparams=optimizer_hparams,
            loss_fn=self._loss_fn,
            seed=seed,
            logger_params=logger_params,
            enable_progress_bar=self.verbose,
            debug=debug,
            check_val_every_epoch=check_val_every_epoch,
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
        Train the compressor.

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
            self._train_loader
        except AttributeError:
            self._create_data_loader(batch_size=training_batch_size)

        try:
            metrics = self.trainer.train_model(
                self._train_loader,
                self._val_loader,
                test_loader=self._test_loader,
                num_epochs=num_epochs,
                min_delta=kwargs.get("min_delta", 1e-3),
                patience=patience,
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
                print("[!] Training the compressor.")
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

        compressor = self.trainer.model
        return metrics, compressor
