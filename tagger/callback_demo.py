from typing import Optional

from allennlp.models import Model
from comet_ml import Experiment

from allenpipeline import Annotator, PipelineTrainer
from allenpipeline.callback import Callback


@Callback.register("demo-callback")
class SomeCallback(Callback):

    def call(self, annotator: Annotator, model: Model, trainer: Optional[PipelineTrainer] = None,
             experiment: Optional[Experiment] = None):
        print("I am an additional callback. In particular, I might be useful when training has finished, since AllenNLP has limited support.")