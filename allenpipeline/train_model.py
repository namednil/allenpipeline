import logging
import os
from typing import List, Optional

from allennlp.common import Registrable, Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.data import DataLoader, DatasetReader, Vocabulary
from allennlp.models import Model
from allennlp.training import Trainer

import allennlp.common.util as common_util
import allennlp.training.util as training_util

from allenpipeline import DatasetWriter, BatchDecoder, BaseEvaluationCommand, Annotator, PipelineTrainer
from allenpipeline.callback import Callbacks

logger = logging.getLogger(__name__)

class TrainPipelineModel(Registrable):
    """
    This class exists so that we can easily read a configuration file with the `allennlp train`
    command.  The basic logic is that we call `train_loop =
    TrainModel.from_params(params_from_config_file)`, then `train_loop.run()`.  This class performs
    very little logic, pushing most of it to the `Trainer` that has a `train()` method.  The
    point here is to construct all of the dependencies for the `Trainer` in a way that we can do
    it using `from_params()`, while having all of those dependencies transparently documented and
    not hidden in calls to `params.pop()`.  If you are writing your own training loop, you almost
    certainly should not use this class, but you might look at the code for this class to see what
    we do, to make writing your training loop easier.
    In particular, if you are tempted to call the `__init__` method of this class, you are probably
    doing something unnecessary.  Literally all we do after `__init__` is call `trainer.train()`.  You
    can do that yourself, if you've constructed a `Trainer` already.  What this class gives you is a
    way to construct the `Trainer` by means of a config file.  The actual constructor that we use
    with `from_params` in this class is `from_partial_objects`.  See that method for a description
    of all of the allowed top-level keys in a configuration file used with `allennlp train`.
    """

    default_implementation = "default"
    """
    The default implementation is registered as 'default'.
    """

    def __init__(
            self,
            serialization_dir: str,
            model: Model,
            trainer: Trainer,
            evaluation_data_loader: DataLoader = None,
            evaluate_on_test: bool = False,
            batch_weight_key: str = "",
    ) -> None:
        self.serialization_dir = serialization_dir
        self.model = model
        self.trainer = trainer
        self.evaluation_data_loader = evaluation_data_loader
        self.evaluate_on_test = evaluate_on_test
        self.batch_weight_key = batch_weight_key

    @classmethod
    def from_partial_objects(
            cls,
            serialization_dir: str,
            local_rank: int,
            dataset_reader: DatasetReader,
            train_data_path: str,
            model: Lazy[Model],
            data_loader: Lazy[DataLoader],
            trainer: Lazy[Trainer],
            vocabulary: Lazy[Vocabulary] = None,
            datasets_for_vocab_creation: List[str] = None,
            validation_dataset_reader: DatasetReader = None,
            validation_data_path: str = None,
            validation_data_loader: Lazy[DataLoader] = None,
            test_data_path: str = None,
            evaluate_on_test: bool = False,
            batch_weight_key: str = "",

            dataset_writer : Optional[DatasetWriter] = None,
            decoder : Optional[BatchDecoder] = None,
            validation_command : Optional[BaseEvaluationCommand] = None,
            test_command : Optional[BaseEvaluationCommand] = None,
            annotator : Optional[Annotator] = None,
            ) -> "TrainModel":
        """
        This method is intended for use with our `FromParams` logic, to construct a `TrainModel`
        object from a config file passed to the `allennlp train` command.  The arguments to this
        method are the allowed top-level keys in a configuration file (except for the first three,
        which are obtained separately).
        You *could* use this outside of our `FromParams` logic if you really want to, but there
        might be easier ways to accomplish your goal than instantiating `Lazy` objects.  If you are
        writing your own training loop, we recommend that you look at the implementation of this
        method for inspiration and possibly some utility functions you can call, but you very likely
        should not use this method directly.
        The `Lazy` type annotations here are a mechanism for building dependencies to an object
        sequentially - the `TrainModel` object needs data, a model, and a trainer, but the model
        needs to see the data before it's constructed (to create a vocabulary) and the trainer needs
        the data and the model before it's constructed.  Objects that have sequential dependencies
        like this are labeled as `Lazy` in their type annotations, and we pass the missing
        dependencies when we call their `construct()` method, which you can see in the code below.
        # Parameters
        serialization_dir: `str`
            The directory where logs and model archives will be saved.
            In a typical AllenNLP configuration file, this parameter does not get an entry as a
            top-level key, it gets passed in separately.
        local_rank: `int`
            The process index that is initialized using the GPU device id.
            In a typical AllenNLP configuration file, this parameter does not get an entry as a
            top-level key, it gets passed in separately.
        dataset_reader: `DatasetReader`
            The `DatasetReader` that will be used for training and (by default) for validation.
        train_data_path: `str`
            The file (or directory) that will be passed to `dataset_reader.read()` to construct the
            training data.
        model: `Lazy[Model]`
            The model that we will train.  This is lazy because it depends on the `Vocabulary`;
            after constructing the vocabulary we call `model.construct(vocab=vocabulary)`.
        data_loader: `Lazy[DataLoader]`
            The data_loader we use to batch instances from the dataset reader at training and (by
            default) validation time. This is lazy because it takes a dataset in it's constructor.
        trainer: `Lazy[Trainer]`
            The `Trainer` that actually implements the training loop.  This is a lazy object because
            it depends on the model that's going to be trained.
        vocabulary: `Lazy[Vocabulary]`, optional (default=`None`)
            The `Vocabulary` that we will use to convert strings in the data to integer ids (and
            possibly set sizes of embedding matrices in the `Model`).  By default we construct the
            vocabulary from the instances that we read.
        datasets_for_vocab_creation: `List[str]`, optional (default=`None`)
            If you pass in more than one dataset but don't want to use all of them to construct a
            vocabulary, you can pass in this key to limit it.  Valid entries in the list are
            "train", "validation" and "test".
        validation_dataset_reader: `DatasetReader`, optional (default=`None`)
            If given, we will use this dataset reader for the validation data instead of
            `dataset_reader`.
        validation_data_path: `str`, optional (default=`None`)
            If given, we will use this data for computing validation metrics and early stopping.
        validation_data_loader: `Lazy[DataLoader]`, optional (default=`None`)
            If given, the data_loader we use to batch instances from the dataset reader at
            validation and test time. This is lazy because it takes a dataset in it's constructor.
        test_data_path: `str`, optional (default=`None`)
            If given, we will use this as test data.  This makes it available for vocab creation by
            default, but nothing else.
        evaluate_on_test: `bool`, optional (default=`False`)
            If given, we will evaluate the final model on this data at the end of training.  Note
            that we do not recommend using this for actual test data in every-day experimentation;
            you should only very rarely evaluate your model on actual test data.
        batch_weight_key: `str`, optional (default=`""`)
            The name of metric used to weight the loss on a per-batch basis.  This is only used
            during evaluation on final test data, if you've specified `evaluate_on_test=True`.
        """

        datasets = training_util.read_all_datasets(
            train_data_path=train_data_path,
            dataset_reader=dataset_reader,
            validation_dataset_reader=validation_dataset_reader,
            validation_data_path=validation_data_path,
            test_data_path=test_data_path,
        )

        if datasets_for_vocab_creation:
            for key in datasets_for_vocab_creation:
                if key not in datasets:
                    raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {key}")

            logger.info(
                "From dataset instances, %s will be considered for vocabulary creation.",
                ", ".join(datasets_for_vocab_creation),
            )

        instance_generator = (
            instance
            for key, dataset in datasets.items()
            if datasets_for_vocab_creation is None or key in datasets_for_vocab_creation
            for instance in dataset
        )

        vocabulary_ = vocabulary.construct(instances=instance_generator)
        if not vocabulary_:
            vocabulary_ = Vocabulary.from_instances(instance_generator)
        model_ = model.construct(vocab=vocabulary_)

        # Initializing the model can have side effect of expanding the vocabulary.
        # Save the vocab only in the master. In the degenerate non-distributed
        # case, we're trivially the master. In the distributed case this is safe
        # to do without worrying about race conditions since saving and loading
        # the vocab involves acquiring a file lock.
        if common_util.is_master():
            vocabulary_path = os.path.join(serialization_dir, "vocabulary")
            vocabulary_.save_to_files(vocabulary_path)

        for dataset in datasets.values():
            dataset.index_with(model_.vocab)

        data_loader_ = data_loader.construct(dataset=datasets["train"])
        validation_data = datasets.get("validation")
        if validation_data is not None:
            # Because of the way Lazy[T] works, we can't check it's existence
            # _before_ we've tried to construct it. It returns None if it is not
            # present, so we try to construct it first, and then afterward back off
            # to the data_loader configuration used for training if it returns None.
            validation_data_loader_ = validation_data_loader.construct(dataset=validation_data)
            if validation_data_loader_ is None:
                validation_data_loader_ = data_loader.construct(dataset=validation_data)
        else:
            validation_data_loader_ = None

        test_data = datasets.get("test")
        if test_data is not None:
            test_data_loader = validation_data_loader.construct(dataset=test_data)
            if test_data_loader is None:
                test_data_loader = data_loader.construct(dataset=test_data)
        else:
            test_data_loader = None

        # We don't need to pass serialization_dir and local_rank here, because they will have been
        # passed through the trainer by from_params already, because they were keyword arguments to
        # construct this class in the first place.
        trainer_ = trainer.construct(
            model=model_, data_loader=data_loader_, validation_data_loader=validation_data_loader_,
            validation_command=validation_command, test_command=test_command,
            dataset_reader=dataset_reader, dataset_writer=dataset_writer,
            decoder=decoder, annotator=annotator
        )

        return cls(
            serialization_dir=serialization_dir,
            model=model_,
            trainer=trainer_,
            evaluation_data_loader=test_data_loader,
            evaluate_on_test=evaluate_on_test,
            batch_weight_key=batch_weight_key,
        )


TrainPipelineModel.register("default", constructor="from_partial_objects")(TrainPipelineModel)