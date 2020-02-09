from dataclasses import dataclass

from allennlp.common import Params

from pipeline.DatasetWriter import DatasetWriter
from pipeline.Decoder import BatchDecoder
from pipeline.evaluation_commands import BaseEvaluationCommand


@dataclass
class PipelineTrainerPieces:
    dataset_writer : DatasetWriter
    decoder : BatchDecoder
    validation_command : BaseEvaluationCommand

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

        return PipelineTrainerPieces(dataset_writer, decoder, validation_command)
