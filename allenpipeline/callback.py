from enum import Enum
from typing import Optional, Dict

from allennlp.common import Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.models import Model
from comet_ml import Experiment

from allenpipeline import Annotator


class Callback(Registrable):
    """
    Called after running on the devset etc.
    """

    def call(self, annotator : Annotator, model : Model, trainer : Optional["PipelineTrainer"] = None, experiment : Optional[Experiment] = None):
        """
        Here you can do what you want.
        :param trainer:
        :param annotator:
        :param model:
        :return:
        """
        raise NotImplementedError()


class CallbackName(Enum):
    AFTER_VALIDATION = "after_validation"
    BEFORE_VALIDATION = "before_validation"
    AFTER_TRAINING = "after_training"

class Callbacks(Registrable):
    POSSIBLE_CALLBACKS_NAMES = [name.value for name in CallbackName]
    POSSIBLE_CALLBACKS = [c for c in CallbackName]

    def __init__(self, callbacks : Dict[str, Callback]):
        if not set(callbacks.keys()).issubset(Callbacks.POSSIBLE_CALLBACKS_NAMES):
            raise ConfigurationError("Unknown callbacks in this list: "+str(callbacks.keys()) + " - I know the following callbacks: "+str(Callbacks.POSSIBLE_CALLBACKS_NAMES))
        self.callbacks = callbacks

    def call_if_registered(self, name : CallbackName, annotator : Annotator, model : Model, trainer : Optional["PipelineTrainer"] = None, experiment : Optional[Experiment] = None):
        if name.value in self.callbacks:
            self.callbacks[name.value].call(annotator, model, trainer, experiment)

        if name not in Callbacks.POSSIBLE_CALLBACKS:
            raise ValueError("Problem in allenpipeline: trainer tried to call the callback "+name.value+" but this is not in the list of possible callbacks")