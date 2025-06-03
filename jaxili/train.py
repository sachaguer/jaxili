"""Train.

This module implements an object to perform the training of Normalizing Flows and more generally of neural networks.
"""

import json
import os
import time
import warnings
from collections import defaultdict
from copy import copy, deepcopy
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import linen as nn
import flax.nnx as nnx
from flax.training import checkpoints, orbax_utils, train_state
from flax.training.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from tqdm import tqdm

from jaxili.model import NDENetwork
from jaxili.inventory.func_dict import jax_nn_dict, jaxili_loss_dict, jaxili_nn_dict
from jaxili.utils import handle_non_serializable

import datasets as hf_datasets


class TrainerModule:
    """
    A module to perform the training of Normalizing Flows.

    This module contains the training loop, evaluation, logging, and checkpointing. It can also be used to load a model from a checkpoint.
    """

    def __init__(
        self,
        model_class: NDENetwork,
        model_hparams: Dict[str, Any],
        optimizer_hparams: Dict[str, Any],
        loss_fn: Callable,
        seed: int = 42,
        logger_params: Dict[str, Any] = None,
        enable_progress_bar: bool = True,
        debug: bool = False,
        check_val_every_epoch: int = 1,
        nde_class: str = "NPE",
        **kwargs,
    ):
        """
        Initialize a basic Trainer module summarizing most training functionalities like logging, model initialization, training loop, etc...

        Attributes
        ----------
        model_class : jaxili.model.NDENetwork
            The class of the model that should be trained.
        model_hparams : Dict[str, Any]
            A dictionnary of the hyperparameters of the model. Is used as input to the model when it is created.
        optimizer_hparams : Dict[str, Any]
            A dictionnary of the hyperparameters of the optimizer. Used during initialization of the optimizer.
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
        nde_class : str
            The class of the Neural Density Estimator. Default is "NPE". Only "NPE" and "NLE" are allowed.
        """
        super().__init__()
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.loss_fn = loss_fn
        self.optimizer_hparams = optimizer_hparams
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.key_rng = jax.random.PRNGKey(seed)
        self.check_val_every_epoch = check_val_every_epoch
        self.nde_class = nde_class
        assert (
            nde_class == "NPE" or nde_class == "NLE"
        ), "Choose a valid class of Neural Density Estimator. (NPE or NLE)"
        self.generate_config(logger_params)
        self.config.update(kwargs)
        # Create a model. Contraty to Flax linen, parameters are created at the same time.
        self.model = self.model_class(**self.model_hparams)
        nnx.display(self.model)
        # Init trainer parts
        self.init_logger(logger_params)
        self.create_jitted_functions()
        # Initialize checkpointer
        self.init_checkpointer()

    def init_logger(self, logger_params: Optional[Dict] = None):
        """
        Initialize the logger and created a logging directory.

        Parameters
        ----------
        logger_params : Dict[str, Any]
            A dictionary containing the specifications of the logger.
        """
        if logger_params is None:
            logger_params = dict()
        # Determine logging directory
        log_dir = logger_params.get("log_dir", None)
        if log_dir is None:
            base_log_dir = logger_params.get("base_log_dir", "checkpoints/")
            # Prepare logging
            log_dir = os.path.join(base_log_dir, self.config["model_class"])
            if "logger_name" in logger_params:
                log_dir = os.path.join(log_dir, logger_params["logger_name"])
            version = None
        else:
            version = ""
        # Create logger object
        logger_type = logger_params.get("logger_type", "TensorBoard").lower()
        if logger_type == "tensorboard":
            self.logger = TensorBoardLogger(save_dir=log_dir, version=version, name="")
        elif logger_type == "wandb":
            self.logger = WandbLogger(save_dir=log_dir, version=version, name="")
        else:
            assert False, f'Unknown logger type "{logger_type}"'
        # Save hyperparameters
        log_dir = self.logger.log_dir
        if not os.path.isfile(os.path.join(log_dir, "hparams.json")):
            os.makedirs(os.path.join(log_dir, "metrics/"), exist_ok=True)
            try:
                self.write_config(log_dir)
            except:
                warnings.warn("Could not save hyperparameters.", Warning)
        self.log_dir = log_dir

    def write_config(self, log_dir):
        """Write the config of the trainer in a JSON file."""
        with open(os.path.join(log_dir, "hparams.json"), "w") as f:
            json.dump(self.config, f, indent=4, default=handle_non_serializable)

    def init_checkpointer(self):
        """Initialize the checkpointer to save the model."""
        options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
        self.checkpoint_manager = ocp.CheckpointManager(self.log_dir, options=options)

    def generate_config(self, logger_params):
        """Generate a configuration dictionary for the trainer."""
        self.config = {
            "model_class": self.model_class.__name__,
            "model_hparams": deepcopy(self.model_hparams),
            "loss_fn": self.loss_fn.__name__,
            "optimizer_hparams": self.optimizer_hparams,
            "logger_params": logger_params,
            "enable_progress_bar": self.enable_progress_bar,
            "debug": self.debug,
            "check_val_every_epoch": self.check_val_every_epoch,
            "seed": self.seed,
            "nde_class": self.nde_class,
        }

        if "activation" in self.model_hparams.keys():
            self.config["model_hparams"]["activation"] = self.model_hparams[
                "activation"
            ].__name__

    def init_optimizer(self, num_epochs: int, num_steps_per_epoch: int):
        """
        Initialize the optimizer and learning rate scheduler.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train.
        num_steps_per_epoch : int
            Number of steps per epoch.
        """
        hparams = copy(self.optimizer_hparams)

        # Initialize optimizer
        optimizer_name = hparams.pop("optimizer_name", "adam")
        if optimizer_name.lower() == "adam":
            opt_class = optax.adam
        elif optimizer_name.lower() == "sgd":
            opt_class = optax.sgd
        elif optimizer_name.lower() == "adamw":
            opt_class = optax.adamw
        else:
            assert False, f'Unknown optimizer "{optimizer_name}"'
        # Initialize learning rate scheduler
        # A cosine decay scheduler is used, but others are also possible
        lr = hparams.pop("lr", 1e-3)
        warmup = hparams.pop(
            "warmup", num_steps_per_epoch
        )  # By default linear warmup during the first epoch
        decay_steps = hparams.pop(
            "decay_steps", int(num_epochs // 2 * num_steps_per_epoch)
        )
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.01 * lr,
            peak_value=lr,
            warmup_steps=warmup,
            decay_steps=decay_steps,
            end_value=0.01 * lr,
        )
        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip_by_global_norm(hparams.pop("gradient_clip", 5.0))]
        if opt_class == optax.sgd and "weight_decay" in hparams:
            transf.append(optax.add_decayed_weights(hparams.pop("weight_decay", 0.0)))
        hparams.pop(
            "weight_decay", None
        )  # removes weight decay if the opt_class is not sgd.
        self.optimizer = nnx.Optimizer(
            self.model, optax.chain(*transf, opt_class(lr_schedule, **hparams))
        )

    def create_jitted_functions(self):
        """
        Create jitted versions of the training, validation and evaluation functions.

        If self.debug is True, no jitting is applied.
        """
        train_step, eval_step = self.create_functions()
        if self.debug:  # Skip jitting
            print("Skipping jitting due to debug=True")
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            self.train_step = nnx.jit(train_step)
            self.eval_step = nnx.jit(eval_step)

    def create_functions(
        self,
    ) -> Tuple[
        Callable[[Any, Any, Any], Tuple[Dict]],
        Callable[[Any, Any], Tuple[Dict]],
    ]:
        """
        Create and returns functions for the training and evaluation step.

        The functions take as input the training state and a batch from the train/val/test loader.
        Both functions are expected to return a dictionary of logging metrics, and the training function a new train state. This function can be overwritten by a subclass. The train_step and eval_step functions here are examples for the signature of the functions.
        """

        def train_step(model: Any, optimizer: Any, batch: Any):
            loss, grads = nnx.value_and_grad(self.loss_fn)(model, batch)
            optimizer.update(grads)
            metrics = {"loss": loss}
            return metrics

        def eval_step(model: Any, batch: Any):
            loss = self.loss_fn(model, batch)
            metrics = {"loss": loss}
            return metrics

        return train_step, eval_step

    def train_model(
        self,
        train_loader: Iterator,
        val_loader: Iterator,
        test_loader: Optional[Iterator] = None,
        num_epochs: int = 500,
        min_delta: float = 1e-3,
        patience: int = 20,
        load_best: bool = True,
    ) -> Dict[str, Any]:
        """
        Start a training loop for the given number of epochs.

        Parameters
        ----------
        train_loader : Iterator
            An iterator over the training data.
        val_loader : Iterator
            An iterator over the validation data.
        test_loader : Iterator
            If given, best model will be evaluated on the test set.
        num_epochs : int
            Number of epochs for which to train the model.
        min_delta : float
            Minimum change in the monitored metric to qualify as an improvement.
        patience : int
            Number of epochs with no improvement after which training will be stopped. Default is 20.
        load_best : bool
            If True, loads the best epoch on the validation set. (Default: True)

        Returns
        -------
        Dict[str, Any]
            A dictionary of the train, validation and evt. test metrics for the best model on the validation set.
        """
        # Create optimizer and the scheduler for the given numer of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        # Prepare training loop
        self.on_training_start()
        best_eval_metrics = None
        best_epoch = None
        early_stop = EarlyStopping(min_delta, patience)
        pbar = self.tracker(range(1, num_epochs + 1), desc="Epochs")
        for epoch_idx in pbar:
            train_metrics = self.train_epoch(train_loader)
            self.logger.log_metrics(train_metrics, step=epoch_idx)
            self.on_training_epoch_end(epoch_idx)
            # Validation every N epochs
            if epoch_idx % self.check_val_every_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix="val/")
                self.on_validation_epoch_end(epoch_idx, eval_metrics, val_loader)
                self.logger.log_metrics(eval_metrics, step=epoch_idx)
                self.save_metrics(f"eval_epoch_{str(epoch_idx).zfill(3)}", eval_metrics)
                early_stop = early_stop.update(eval_metrics["val/loss"])
                # Save best model
                if early_stop.has_improved:
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    best_epoch = epoch_idx
                    self.save_model(step=epoch_idx)
                    self.save_metrics("best_eval", best_eval_metrics)

                if early_stop.should_stop:
                    print(f"Neural network training stopped after {epoch_idx} epochs.")
                    print(
                        f"Early stopping with best validation metric: {early_stop.best_metric}"
                    )
                    print(f"Best model saved at epoch {best_epoch}")
                    print(
                        f"Early stopping parameters: min_delta={min_delta}, patience={patience}"
                    )
                    break
                if self.enable_progress_bar:
                    pbar.set_description(
                        f"Epochs: Val loss {eval_metrics['val/loss']:.5g}/ Best val loss {early_stop.best_metric:.5g}"
                    )
        # Test best model if possible
        if test_loader is not None:
            if load_best:
                self.load_model()
            test_metrics = self.eval_model(test_loader, log_prefix="test/")
            self.logger.log_metrics(test_metrics, step=epoch_idx)
            self.save_metrics("test", test_metrics)
            best_eval_metrics.update(test_metrics)
        # Close logger
        self.logger.finalize("success")
        return best_eval_metrics

    def train_epoch(self, train_loader: Iterator) -> Dict[str, Any]:
        """
        Train the model for one epoch.

        Parameters
        ----------
        train_loader : Iterator
            An iterator over the training data.

        Returns
        -------
        Dict[str, Any]
            A dictionary of the average training metrics over all batches for logging
        """
        # Train model for one epoch, and log avg loss and accuracy
        hf_dataset = False
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        if isinstance(train_loader.dataloader.dataset, hf_datasets.Dataset):
            if self.check_hf_dataset(train_loader):
                raise ValueError(
                    "The Dataset should contain the keys 'theta' and 'x'. Make sure that the dataset has the correct format."
                )
            hf_dataset = True
        for batch in train_loader:
            if hf_dataset:
                batch = self.handle_hf_dataset(batch)
            step_metrics = self.train_step(self.model, self.optimizer, batch)
            for key in step_metrics:
                metrics["train/" + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics["epoch_time"] = time.time() - start_time
        return metrics

    def eval_model(
        self, data_loader: Iterator, log_prefix: Optional[str] = ""
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.

        Parameters
        ----------
        data_loader : Iterator
            An iterator over the data.
        log_prefix : str
            A prefix to add to all metrics.

        Returns
        -------
        Dict[str, Any]
            A dictionary of the evaluation metrics, averaged over data points in the dataset
        """
        hf_dataset = False
        # Test model on all element of the dataloader and return avg loss
        metrics = defaultdict(float)
        num_elements = 0
        if isinstance(data_loader.dataloader.dataset, hf_datasets.Dataset):
            if self.check_hf_dataset(data_loader):
                raise ValueError(
                    "The Dataset should contain the keys 'theta' and 'x'. Make sure that the dataset has the correct format."
                )
            hf_dataset = True
        for batch in data_loader:
            if hf_dataset:
                batch = self.handle_hf_dataset(batch)
            step_metrics = self.eval_step(self.model, batch)
            batch_size = (
                batch[0].shape[0]
                if isinstance(batch, (list, tuple))
                else batch.shape[0]
            )
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size
            num_elements += batch_size
        metrics = {
            (log_prefix + key): (metrics[key] / num_elements).item() for key in metrics
        }
        return metrics

    def is_new_model_better(
        self, new_metrics: Dict[str, Any], old_metrics: Dict[str, Any]
    ) -> bool:
        """
        Compare two sets of evaluation metrics to decide whether the new model is better than the previous ones or not.

        Parameters
        ----------
        new_metrics : Iterator
            A dictionary of the evaluation metrics of the new model
        old_metrics : Iterator
            A dictionary of the evaluation metrics of the previously best model.

        Returns
        -------
        bool
            True if the new model is better, False otherwise.
        """
        if old_metrics is None:
            return True
        for key, is_larger in [
            ("val/val_metric", False),
            ("val/acc", True),
            ("val/loss", False),
        ]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f"No known metrics to log on: {new_metrics}"

    def tracker(self, iterator: Iterator, **kwargs) -> Iterator:
        """
        Wrap an iterator in a progress bar track (tqdm) if the progress bar is enabled.

        Parameters
        ----------
        iterator : Iterator
            Iterator to wrap in tqdm.
        kwargs : Any
            additional arguments to tqdm.

        Returns
        -------
        Iterator
            Wrapped iterator if progress bar is enabled, otherwise same iterator than input.
        """
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """
        Save a dictionary of metrics to file.

        This can be used as a textual representation of the validation performance for checking in the terminal.

        Parameters
        ----------
        filename : str
            Name of the metrics file without folders and postfix
        metrics : Dict[str, Any]
            A dictionary of the metrics to save
        """
        with open(os.path.join(self.log_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def check_hf_dataset(self, loader: hf_datasets.Dataset):
        """
        Check if the hugging face dataset contains the correct features.

        Parameters
        ----------
        loader : hf_datasets.Dataset
            The HuggingFace data loader that needs to be checked.
        """
        dataset_keys = loader.dataloader.dataset.features.keys()
        return ("theta" not in dataset_keys) or ("x" not in dataset_keys)

    def handle_hf_dataset(self, batch):
        """Modify the batch obtained from HuggingFace dataset to feed it correctly to the training loop."""
        return batch["theta"], batch["x"]

    def on_training_start(self):
        """
        Perform any necessary operations before the training starts.

        Method called before training is started. Can be used for additional initialization operations etc.
        """
        pass

    def on_training_epoch_end(self, epoch_idx: int):
        """
        Perform any necessary operations at the end of each training epoch.

        Method called at the end of each training epoch. Can be used for additional logging or similar.

        Parameters
        ----------
        epoch_idx : int
            Index of the epoch that just finished.
        """
        pass

    def on_validation_epoch_end(
        self, epoch_idx: int, eval_metrics: Dict[str, Any], val_loader: Iterator
    ):
        """
        Perform any necessary operations at the end of each validation epoch.

        Method called at the end of each validation epoch. Can be used for additional logging and evaluation.

        Parameters
        ----------
        epoch_idx : int
            Index of the epoch that just finished.
        eval_metrics : Dict[str, Any]
            A dictionary of the evaluation metrics.
        val_loader : Iterator
            DataLoader of the validation set to support additional evaluation.
        """
        pass

    def save_model(self, step: int = 0):
        """
        Save current training state at certain training iteration.

        Only the model parameters and batch statistics are saved to reduce memory footprint. To allow the training to be continued from a checkpoint, this method can be extended to include the optimizer state as well.

        Parameters
        ----------
        step : int
            Index of the step to save the model at, e.g. epoch.
        """
        _, state = nnx.split(self.model)
        self.checkpoint_manager.save(step, args=ocp.args.StandardSave(state))
        self.checkpoint_manager.wait_until_finished()

    def load_model(self):
        """Load model and batch statistics from the logging directory."""
        step = self.checkpoint_manager.latest_step()
        state_dict = self.checkpoint_manager.restore(step)
        graphdef, _ = nnx.split(self.model)
        self.model = nnx.merge(graphdef, state_dict)

    def bind_model(self):
        """
        Return a model with parameters bound to it. Enables an easier inference access.

        Returns
        -------
        The model with parameters and evt. batch statistics bound to it.
        """
        warnings.warn(
            "This function is deprecated since the transition form Flax linen to Flax NNX. The model is bind by default now.",
            DeprecationWarning,
        )

    @classmethod
    def load_from_checkpoints(cls, model_class: NDENetwork, checkpoint: str) -> Any:
        """
        Create a Trainer object with same hyperparameters and loaded model from a checkpoint directory.

        Parameters
        ----------
        model_class : jaxili.model.NDENetwork
            The class of the model that should be loaded.
        checkpoint : str
            Folder in which the checkpoint and hyperparameter file is stored

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
        # Check if an activation function is used as a hyperparameter if the neural network.
        assert (
            hparams["loss_fn"] in jaxili_loss_dict
        ), "Unknown loss function. Check that the loss function you used comes from `jax.nn`."
        hparams["loss_fn"] = jaxili_loss_dict[hparams["loss_fn"]]
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
        trainer = cls(model_class=model_class, **hparams)
        trainer.load_model()
        return trainer
