from typing import Any, Callable, Dict, Optional, Union
from jaxtyping import Array, Float, PyTree
import jax
import jax.numpy as jnp
import jax.random as jr

import jaxili
from jaxili.utils import (
    check_density_estimator,
    validate_theta_x
)
from jaxili.train import TrainerModule
from jaxili.loss import loss_nll_npe

import jax_dataloader as jdl

class NPE():

    def __init__(
            self,
            model_class: jaxili.model.NDENetwork,
            model_hparams: Dict[str, Any],
            logging_level: Union[int, str] = "WARNING",
            show_progress_bar: bool = True
    ):
        """
        Base class for Neural Posterior Estimation (NPE) methods.

        Parameters
        ----------
        model_class : jaxili.model.NDENetwork
            Class of the neural density estimator to use.
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
        self.show_progress_bar = show_progress_bar        

    
    def append_simulations(
            self,
            theta: Array,
            x: Array,        
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
        """
        #Verify theta and x typing and batch_size
        theta, x, batch_size = validate_theta_x(theta, x)

        self._dim_params = theta.shape[1]
        self._dim_outputs = x.shape[1]
        self._batch_size = batch_size

        #Create a Dataset
        self._dataset = jdl.ArrayDataset(theta, x)

        return self
    
    def create_trainer(self,
                       optimizer_hparams: Dict[str, Any],
                       exmp_input: Any,
                       seed: int = 42,
                       logger_params : Dict[str, Any] = None,
                       debug : bool = False,
                       check_val_every_epoch : int = 1):
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

        self.trainer = TrainerModule(
            model_class = self._model_class,
            model_hparams = self._model_hparams,
            optimizer_hparams = optimizer_hparams,
            loss_fn = loss_nll_npe,

        )
    
    def train(
            self,
            training_batch_size: int = 50,
            learning_rate: float =  5e-4,
            validation_size: float = 0.1,
            patience: int = 20,
            n_epoch: int = 2**31 -1,
            model_hparams: Optional[Dict] = None,
            key: Optional[Array] = None,
            checkpoint_path: Optional[str] = '.'

    ):
        r"""
        Train the density estimator to approximate the distribution $p(\theta|x)$.

        Parameters
        ----------
        training_batch_size : int, optional
            Batch size to use during training. Default is 50.
        learning_rate: float, optional
            Learning rate to use during training. Default is 5e-4.
        validation_size: float, optional
            Fraction of the dataset to use for validation. Default is 0.1.
        patience: int, optional
            Number of epochs to wait before early stopping. Default is 20.
        n_epoch: int, optional
            Maximum number of epochs to train. Default is 2**31 - 1.
        model_hparams: Optional[Dict], optional
            Hyperparameters to use for the model. Default is None.
        key: Optional[Array], optional
            Random key to use for training. Default is None.
        """

        #Create a training and validation DataLoader using the dataset.
        assert (self._dataset is not None), "No dataset found. Please append simulations first."

        if key is None:
            key = jr.PRNGKey(0)
        key, subkey = jr.split(key)

        permutation = jr.permutation(subkey, self._batch_size)

        index_max = int((1 - validation_size) * self._batch_size)

        train_idx = permutation[:index_max]
        val_idx = permutation[index_max:]

        train_ds = jdl.ArrayDataset(*self._dataset[train_idx])
        val_ds = jdl.ArrayDataset(*self._dataset[val_idx])

        train_loader = jdl.DataLoader(
            train_ds,
            'jax',
            batch_size=training_batch_size,
            shuffle=True,
            drop_last=False
        )
        val_loader =  jdl.DataLoader(
            val_ds,
            'jax',
            batch_size=training_batch_size,
            shuffle=False,
            drop_last=False
        )

        #Build the trainer with correct architecture
        self.trainer = TrainerModule(
            
        )


