from typing import List, Dict, Any, Optional

import torch
from allennlp.common import Registrable
from allennlp.data import DataIterator, Instance, DatasetReader
from allennlp.models import Model
import allennlp.nn.util as util

from allenpipeline.DatasetWriter import DatasetWriter
from allenpipeline.Decoder import BatchDecoder, split_up
from allenpipeline.OrderedDatasetReader import OrderedDatasetReader


class Annotator(Registrable):
    """
    Annotate instances with a model, restores original order.
    """

    def __init__(self, data_iterator : DataIterator, dataset_reader : DatasetReader, dataset_writer : DatasetWriter, decoder : Optional[BatchDecoder] = None):

        self.dataset_writer = dataset_writer
        self.data_iterator = data_iterator
        self.decoder = decoder
        self.dataset_reader = dataset_reader

    def annotate_file(self, model: Model, input_file : str, output_file : str) -> None:
        test_instances = self.dataset_reader.read(input_file)
        annotated = self.annotate(model, list(test_instances))

        with open(output_file,"w") as f:
            self.dataset_writer.write_to_file(model.vocab, annotated, f)

    def annotate(self, model : Model, instances : List[Instance]) -> List[Dict[str, Any]]:
        with torch.no_grad():
            self.data_iterator.index_with(model.vocab)
            cuda_device = model._get_prediction_device()
            preds = []
            for dataset in self.data_iterator._create_batches(instances,shuffle=False):
                dataset.index_instances(model.vocab)
                model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
                output_dict = model.decode(model(**model_input))

                output_dict = split_up(output_dict, model_input["order_metadata"])
                preds.extend(output_dict)

            if self.decoder:
                preds = self.decoder.decode_batch(model.vocab, preds)

            return OrderedDatasetReader.restore_order(preds)
