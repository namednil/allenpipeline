from typing import List, Dict, Any, Optional, Callable

import torch
from allennlp.common import Registrable, Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.data import Instance, DatasetReader, DataLoader, AllennlpDataset, BatchSampler
from allennlp.models import Model
import allennlp.nn.util as util

from allenpipeline.DatasetWriter import DatasetWriter
from allenpipeline.Decoder import BatchDecoder, split_up
from allenpipeline.OrderedDatasetReader import OrderedDatasetReader


class Annotator(Registrable):
    """
    Annotate instances with a model, restores original order.
    """

    def __init__(self,  data_loader: Lazy[DataLoader],
                 dataset_reader: DatasetReader,
                 dataset_writer: DatasetWriter,
                 decoder: Optional[BatchDecoder] = None):

        self.dataset_writer = dataset_writer
        self.data_loader = data_loader
        self.decoder = decoder
        self.dataset_reader = dataset_reader

    def annotate(self, model : Model, instances: List[Instance],
                 batch_size: Optional[int] = None,
                 annotation_function: Optional[Callable[[Model, Dict[str, torch.Tensor]], torch.Tensor]] = None) -> List[Dict[str, Any]]:
        dataset = AllennlpDataset(instances, model.vocab)
        with torch.no_grad():
            cuda_device = model._get_prediction_device()

            data_loader = self.data_loader.construct(dataset=dataset)
            if batch_size is not None:
                if data_loader.batch_sampler is not None:
                    data_loader.batch_sampler.batch_size = batch_size #overwrite batch size
                    data_loader = DataLoader(dataset, batch_sampler=data_loader.batch_sampler)
                else:
                    data_loader = DataLoader(dataset, batch_size=batch_size)

            preds = []
            for batch in data_loader:
                model_input = util.move_to_device(batch, cuda_device)
                if annotation_function is not None:
                    outputs = annotation_function(model, model_input)
                else:
                    outputs = model.make_output_human_readable(model(**model_input))

                output_dict = split_up(outputs, model_input["order_metadata"])
                preds.extend(output_dict)

            if self.decoder is not None:
                preds = self.decoder.decode_batch(model.vocab, preds)

            return OrderedDatasetReader.restore_order(preds)

    def annotate_file(self, model : Model, input_file: str, output_file: str, batch_size: Optional[int] = None,
                      annotation_function: Optional[Callable[[Model, Dict[str, torch.Tensor]], torch.Tensor]] = None) -> None:
        test_instances = self.dataset_reader.read(input_file)
        annotated = self.annotate(model, list(test_instances), batch_size=batch_size, annotation_function=annotation_function)

        with open(output_file, "w") as f:
            self.dataset_writer.write_to_file(model.vocab, annotated, f)



Annotator.register("default")(Annotator)