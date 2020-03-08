from dataclasses import dataclass

from allennlp.common import Params

from allenpipeline.annotate import Annotator
from allenpipeline.DatasetWriter import DatasetWriter
from allenpipeline.Decoder import BatchDecoder
from allenpipeline.evaluation_commands import BaseEvaluationCommand


@dataclass
class PipelineTrainerPieces:
    dataset_writer : DatasetWriter
    decoder : BatchDecoder
    validation_command : BaseEvaluationCommand
    test_command : BaseEvaluationCommand
    annotator : Annotator

    @staticmethod
    def from_params(params : Params):
        dataset_writer = None
        if "dataset_writer" in  params:
            dataset_writer = DatasetWriter.from_params(params.pop("dataset_writer"))

        decoder = None
        if "decoder" in params:
            decoder = BatchDecoder.from_params(params.pop("decoder"))

        validation_command = None
        if "validation_command" in params:
            validation_command = BaseEvaluationCommand.from_params(params.pop("validation_command"))

        test_command = None
        if "validation_command" in params:
            test_command = BaseEvaluationCommand.from_params(params.pop("test_command"))

        annotator = None
        if "annotator" in params:
            annotator = Annotator.from_params(params.pop("annotator"))

        return PipelineTrainerPieces(dataset_writer, decoder, validation_command, test_command, annotator)
