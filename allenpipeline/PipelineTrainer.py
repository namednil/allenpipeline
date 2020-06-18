import datetime
import logging
import math
import os
import re
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from allennlp.common.util import int_to_device
from comet_ml import Experiment

from allenpipeline import BatchDecoder, DatasetWriter, BaseEvaluationCommand, OrderedDatasetReader
from allenpipeline.Decoder import split_up
from allenpipeline.annotate import Annotator
from allenpipeline.callback import Callbacks, CallbackName

try:
    from apex import amp
except ImportError:
    amp = None
import torch
import torch.distributed as dist
import torch.optim.lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.data import DataLoader, DatasetReader
from allennlp.data.dataloader import TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter

from allennlp.training.trainer import Trainer, GradientDescentTrainer, BatchCallback, EpochCallback


logger = logging.getLogger(__name__)

@Trainer.register("pipeline", constructor="from_partial_objects")
class PipelineTrainer(GradientDescentTrainer):
    """
    A trainer for doing supervised learning with gradient descent. It just takes a labeled dataset
    and a `DataLoader`, and uses the supplied `Optimizer` to learn the weights for your model over
    some fixed number of epochs. You can also pass in a validation dataloader and enable early
    stopping. There are many other bells and whistles as well.

    Registered as a `Trainer` with the name "gradient_descent" (and is also the default `Trainer`).
    The constructor that is registered is `from_partial_objects` - see the arguments to that
    function for the exact keys that should be used, if you are using a configuration file.  They
    largely match the arguments to `__init__`, and we don't repeat their docstrings in
    `from_partial_objects`.

    [0]: https://tinyurl.com/y5mv44fw
    [1]: https://nvidia.github.io/apex/amp.html#opt-levels-and-properties

    # Parameters

    model : `Model`, required.
        An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
        their `forward` method returns a dictionary with a "loss" key, containing a
        scalar tensor representing the loss function to be optimized.

        If you are training your model using GPUs, your model should already be
        on the correct device. (If you are using our `train` command this will be
        handled for you.)

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    optimizer : `torch.nn.Optimizer`, required.
        An instance of a Pytorch Optimizer, instantiated with the parameters of the
        model to be optimized.

    data_loader : `DataLoader`, required.
        A pytorch `DataLoader` containing your `Dataset`, yielding padded indexed batches.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    patience : `Optional[int] > 0`, optional (default=`None`)
        Number of epochs to be patient before early stopping: the training is stopped
        after `patience` epochs with no improvement. If given, it must be `> 0`.
        If None, early stopping is disabled.

    validation_metric : `str`, optional (default=`"loss"`)
        Validation metric to measure for whether to stop training using patience
        and whether to serialize an `is_best` model each epoch. The metric name
        must be prepended with either "+" or "-", which specifies whether the metric
        is an increasing or decreasing function.

    validation_data_loader : `DataLoader`, optional (default=`None`)
        A `DataLoader` to use for the validation set.  If `None`, then
        use the training `DataLoader` with the validation data.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    num_epochs : `int`, optional (default = `20`)
        Number of training epochs.

    serialization_dir : `str`, optional (default=`None`)
        Path to directory for saving and loading model files. Models will not be saved if
        this parameter is not passed.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    checkpointer : `Checkpointer`, optional (default=`None`)
        A `Checkpointer` is responsible for periodically saving model weights.  If none is given
        here, we will construct one with default parameters.

    cuda_device : `int`, optional (default = `-1`)
        An integer specifying the CUDA device(s) to use for this process. If -1, the CPU is used.
        Data parallelism is controlled at the allennlp train level, so each trainer will have a single
        GPU.

    grad_norm : `float`, optional, (default = `None`).
        If provided, gradient norms will be rescaled to have a maximum of this value.

    grad_clipping : `float`, optional (default = `None`).
        If provided, gradients will be clipped `during the backward pass` to have an (absolute)
        maximum of this value.  If you are getting `NaNs` in your gradients during training
        that are not solved by using `grad_norm`, you may need this.

    learning_rate_scheduler : `LearningRateScheduler`, optional (default = `None`)
        If specified, the learning rate will be decayed with respect to
        this schedule at the end of each epoch (or batch, if the scheduler implements
        the `step_batch` method). If you use `torch.optim.lr_scheduler.ReduceLROnPlateau`,
        this will use the `validation_metric` provided to determine if learning has plateaued.
        To support updating the learning rate on every batch, this can optionally implement
        `step_batch(batch_num_total)` which updates the learning rate given the batch number.

    momentum_scheduler : `MomentumScheduler`, optional (default = `None`)
        If specified, the momentum will be updated at the end of each batch or epoch
        according to the schedule.

    tensorboard_writer : `TensorboardWriter`, optional
        If this is not provided, we will construct a `TensorboardWriter` with default
        parameters and use that.

    moving_average : `MovingAverage`, optional, (default = `None`)
        If provided, we will maintain moving averages for all parameters. During training, we
        employ a shadow variable for each parameter, which maintains the moving average. During
        evaluation, we backup the original parameters and assign the moving averages to corresponding
        parameters. Be careful that when saving the checkpoint, we will save the moving averages of
        parameters. This is necessary because we want the saved model to perform as well as the validated
        model if we load it later. But this may cause problems if you restart the training from checkpoint.

    batch_callbacks : `List[BatchCallback]`, optional (default = `None`)
        A list of callbacks that will be called at the end of every batch, during both train and
        validation.

    epoch_callbacks : `List[EpochCallback]`, optional (default = `None`)
        A list of callbacks that will be called at the end of every epoch, and at the start of
        training (with epoch = -1).

    distributed : `bool`, optional, (default = `False`)
        If set, PyTorch's `DistributedDataParallel` is used to train the model in multiple GPUs. This also
        requires `world_size` to be greater than 1.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately (you need a top-level "distributed" key, next to
        the "trainer" entry, that specifies a list of "cuda_devices").

    local_rank : `int`, optional, (default = `0`)
        This is the unique identifier of the `Trainer` in a distributed process group. The GPU device id is
        used as the rank.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    world_size : `int`, (default = `1`)
        The number of `Trainer` workers participating in the distributed training.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "trainer", it gets constructed separately.

    num_gradient_accumulation_steps : `int`, optional, (default = `1`)
        Gradients are accumulated for the given number of steps before doing an optimizer step. This can
        be useful to accommodate batches that are larger than the RAM size. Refer [Thomas Wolf's
        post][0] for details on Gradient Accumulation.

    opt_level : `str`, optional, (default = `None`)
        Each opt_level establishes a set of properties that govern Amp’s implementation of pure or mixed
        precision training. Must be a choice of `"O0"`, `"O1"`, `"O2"`, or `"O3"`.
        See [the Apex documentation][1] for
        more details. If `None`, Amp is not used. Defaults to `None`.

    """

    def __init__(
            self,
            model: Model,
            optimizer: torch.optim.Optimizer,
            data_loader: torch.utils.data.DataLoader,
            patience: Optional[int] = None,
            validation_metric: str = "-loss",
            validation_data_loader: torch.utils.data.DataLoader = None,
            num_epochs: int = 20,
            serialization_dir: Optional[str] = None,
            checkpointer: Checkpointer = None,
            cuda_device: int = -1,
            grad_norm: Optional[float] = None,
            grad_clipping: Optional[float] = None,
            learning_rate_scheduler: Optional[LearningRateScheduler] = None,
            momentum_scheduler: Optional[MomentumScheduler] = None,
            tensorboard_writer: TensorboardWriter = None,
            moving_average: Optional[MovingAverage] = None,
            batch_callbacks: List[BatchCallback] = None,
            epoch_callbacks: List[EpochCallback] = None,
            distributed: bool = False,
            local_rank: int = 0,
            world_size: int = 1,
            num_gradient_accumulation_steps: int = 1,
            opt_level: Optional[str] = None,

            epochs_before_validate: int = 0,
            annotator : Optional[Annotator] = None,
            external_callbacks: Optional[Callbacks] = None,
            decoder: Optional[BatchDecoder] = None,
            dataset_writer: Optional[DatasetWriter] = None,
            validation_command: Optional[BaseEvaluationCommand] = None,

    ) -> None:
        super().__init__(model,
            optimizer,
            data_loader,
            patience,
            validation_metric,
            validation_data_loader,
            num_epochs,
            serialization_dir,
            checkpointer,
            cuda_device,
            grad_norm,
            grad_clipping,
            learning_rate_scheduler,
            momentum_scheduler,
            tensorboard_writer,
            moving_average,
            batch_callbacks,
            epoch_callbacks,
            distributed,
            local_rank,
            world_size,
            num_gradient_accumulation_steps,
            opt_level)

        self.decoder = decoder
        self.annotator = annotator
        self.external_callbacks = external_callbacks
        self.dataset_writer = dataset_writer
        self.validation_command = validation_command
        self.epochs_before_validate = epochs_before_validate


    def _validation_loss(self, epoch: int) -> Tuple[float, float, int, List[Dict[str, torch.Tensor]]]:
        """
        Computes the validation loss. Returns it and the number of batches.
        Also returns list of predictions.
        """
        logger.info("Validating")

        self._pytorch_model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_data_loader is not None:
            validation_data_loader = self._validation_data_loader
        else:
            raise ConfigurationError(
                "Validation results cannot be calculated without a validation_data_loader"
            )

        val_generator_tqdm = Tqdm.tqdm(validation_data_loader)
        batches_this_epoch = 0
        val_loss = 0
        val_reg_loss = 0
        done_early = False
        preds = []
        for batch in val_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing validation early! "
                        "This implies that there is an imbalance in your validation "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            batch_outputs = self.batch_outputs(batch, for_training=False)
            loss = batch_outputs.get("loss")
            reg_loss = batch_outputs.get("reg_loss")
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()
                if reg_loss is not None:
                    val_reg_loss += reg_loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(
                self.model,
                val_loss,
                val_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

            if self.dataset_writer:
                output_dict = self.model.make_output_human_readable(batch_outputs)
                output_dict = split_up(output_dict, batch["order_metadata"])
                preds.extend(output_dict)

            for callback in self._batch_callbacks:
                callback(
                    self,
                    [batch],
                    [batch_outputs],
                    epoch,
                    batches_this_epoch,
                    is_training=False,
                    is_master=self._master,
                )

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (validation)."
            )
            # Indicate that we're done so that any workers that have remaining data stop validation early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_loss, val_reg_loss, batches_this_epoch, preds

    def train(self, experiment : Optional[Experiment] = None) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        self.experiment = experiment

        logger.info("Beginning training.")

        self.val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        self.metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        self.metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            self.metrics["best_validation_" + key] = value

        for callback in self._epoch_callbacks:
            callback(self, metrics={}, epoch=-1, is_master=self._master)

        for epoch in range(epoch_counter, self._num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if experiment:
                with experiment.train():
                    experiment.log_metrics({k : v for k,v in train_metrics.items() if np.isscalar(v)}, step=epoch)

            # get peak of memory usage
            for key, value in train_metrics.items():
                if key.startswith("gpu_") and key.endswith("_memory_MB"):
                    self.metrics["peak_" + key] = max(self.metrics.get("peak_" + key, 0), value)
                elif key.startswith("worker_") and key.endswith("_memory_MB"):
                    self.metrics["peak_" + key] = max(self.metrics.get("peak_" + key, 0), value)

            if self._validation_data_loader is not None and epoch >= self.epochs_before_validate:
                with torch.no_grad():
                    try:
                        if self.external_callbacks:
                            self.external_callbacks.call_if_registered(CallbackName.BEFORE_VALIDATION, annotator=self.annotator, model=self.model, trainer=self, experiment=experiment)

                        # We have a validation set, so compute all the metrics on it.
                        val_loss, val_reg_loss, num_batches, preds = self._validation_loss(epoch)

                        # It is safe again to wait till the validation is done. This is
                        # important to get the metrics right.
                        if self._distributed:
                            dist.barrier()

                        self.val_metrics = training_util.get_metrics(
                            self.model,
                            val_loss,
                            val_reg_loss,
                            num_batches,
                            reset=True,
                            world_size=self._world_size,
                            cuda_device=self.cuda_device,
                        )

                        if self.dataset_writer:
                            if self.decoder:
                                preds = self.decoder.decode_batch(self.model.vocab, preds)
                            filename = self._serialization_dir+f"/pred_epoch_{epoch}.txt"
                            with open(filename,"w") as f:
                                self.dataset_writer.write_to_file(self.model.vocab, OrderedDatasetReader.restore_order(preds), f)

                            if self.validation_command:
                                self.val_metrics.update(self.validation_command.evaluate(filename))

                        if self.external_callbacks:
                            self.external_callbacks.call_if_registered(CallbackName.AFTER_VALIDATION, annotator=self.annotator, model=self.model, trainer=self, experiment=experiment)


                        # Check validation metric for early stopping
                        this_epoch_val_metric = self.val_metrics[self._validation_metric]
                        self._metric_tracker.add_metric(this_epoch_val_metric)

                        if self._metric_tracker.should_stop_early():
                            logger.info("Ran out of patience.  Stopping training.")
                            break

                    except Exception as ex:
                        print("An exception occured:")
                        print(ex)
                        self._checkpointer.save_checkpoint("validation-failed", trainer=self)
                        raise

            if self._master:
                self._tensorboard.log_metrics(
                    train_metrics, val_metrics=self.val_metrics, log_to_console=True, epoch=epoch + 1
                )  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            self.metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            self.metrics["training_start_epoch"] = epoch_counter
            self.metrics["training_epochs"] = epochs_trained
            self.metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                self.metrics["training_" + key] = value
            for key, value in self.val_metrics.items():
                self.metrics["validation_" + key] = value

            if experiment:
                with experiment.validate():
                    experiment.log_metrics({k : v for k,v in self.metrics.items() if np.isscalar(v)}, step=epoch)

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                self.metrics["best_epoch"] = epoch
                for key, value in self.val_metrics.items():
                    self.metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = self.val_metrics

            if self._serialization_dir and self._master:
                common_util.dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), self.metrics
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric)

            if self._master:
                self._checkpointer.save_checkpoint(
                    epoch, self, is_best_so_far=self._metric_tracker.is_best_so_far()
                )

            # Wait for the master to finish saving the checkpoint
            if self._distributed:
                dist.barrier()

            for callback in self._epoch_callbacks:
                callback(self, metrics=self.metrics, epoch=epoch, is_master=self._master)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                        (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        self._tensorboard.close()

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        if self.external_callbacks:
            self.external_callbacks.call_if_registered(CallbackName.AFTER_TRAINING, annotator=self.annotator, model=self.model, trainer=self, experiment=experiment)

        return self.metrics



    @classmethod
    def from_partial_objects(
            cls,
            model: Model,
            serialization_dir: str,
            data_loader: DataLoader,
            validation_data_loader: DataLoader = None,
            local_rank: int = 0,
            patience: int = None,
            validation_metric: str = "-loss",
            num_epochs: int = 20,
            cuda_device: int = -1,
            grad_norm: float = None,
            grad_clipping: float = None,
            distributed: bool = None,
            world_size: int = 1,
            num_gradient_accumulation_steps: int = 1,
            opt_level: Optional[str] = None,
            no_grad: List[str] = None,
            optimizer: Lazy[Optimizer] = None,
            learning_rate_scheduler: Lazy[LearningRateScheduler] = None,
            momentum_scheduler: Lazy[MomentumScheduler] = None,
            tensorboard_writer: Lazy[TensorboardWriter] = None,
            moving_average: Lazy[MovingAverage] = None,
            checkpointer: Lazy[Checkpointer] = None,
            batch_callbacks: List[BatchCallback] = None,
            epoch_callbacks: List[EpochCallback] = None,

            epochs_before_validate: int = 0,
            annotator : Optional[Annotator] = None,
            decoder: Optional[BatchDecoder] = None,
            dataset_writer: Optional[DatasetWriter] = None,
            validation_command: Optional[BaseEvaluationCommand] = None,
            external_callbacks: Optional[Callbacks] = None
    ) -> "Trainer":
        """
        This method exists so that we can have a documented method to construct this class using
        `FromParams`. If you are not using `FromParams` or config files, you can safely ignore this
        method.

        The reason we can't just use `__init__` with `FromParams` here is because there are
        sequential dependencies to this class's arguments.  Anything that has a `Lazy[]` type
        annotation needs something from one of the non-`Lazy` arguments.  The `Optimizer` needs to
        have the parameters from the `Model` before it's constructed, and the `Schedulers` need to
        have the `Optimizer`. Because of this, the typical way we construct things `FromParams`
        doesn't work, so we use `Lazy` to allow for constructing the objects sequentially.

        If you're not using `FromParams`, you can just construct these arguments in the right order
        yourself in your code and call the constructor directly.
        """

        check_for_gpu(cuda_device)
        if cuda_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(cuda_device)

        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        common_util.log_frozen_and_tunable_parameter_names(model)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer_ = optimizer.construct(model_parameters=parameters)
        if not optimizer_:
            optimizer_ = Optimizer.default(parameters)

        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)
        except TypeError:
            batches_per_epoch = None

        moving_average_ = moving_average.construct(parameters=parameters)
        learning_rate_scheduler_ = learning_rate_scheduler.construct(
            optimizer=optimizer_, num_epochs=num_epochs, num_steps_per_epoch=batches_per_epoch
        )
        momentum_scheduler_ = momentum_scheduler.construct(optimizer=optimizer_)

        checkpointer_ = checkpointer.construct() or Checkpointer(serialization_dir)
        tensorboard_writer_ = tensorboard_writer.construct() or TensorboardWriter(serialization_dir)

        return cls(
            model,
            optimizer_,
            data_loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler_,
            momentum_scheduler=momentum_scheduler_,
            tensorboard_writer=tensorboard_writer_,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            batch_callbacks=batch_callbacks,
            epoch_callbacks=epoch_callbacks,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            opt_level=opt_level,

            epochs_before_validate=epochs_before_validate,
            annotator=annotator,
            decoder=decoder,
            dataset_writer=dataset_writer,
            validation_command=validation_command,
            external_callbacks=external_callbacks

        )






