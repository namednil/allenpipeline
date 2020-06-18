# from abc import ABC
# from typing import Optional, Dict, List, Any
#
# import torch
# from allennlp.common import Lazy
# from allennlp.data import Vocabulary, BatchSampler, DatasetReader, Instance, AllennlpDataset, DataLoader
# from allennlp.models import Model
# from allennlp.nn import RegularizerApplicator
#
# from allenpipeline import DatasetWriter, BatchDecoder, OrderedDatasetReader
# from allenpipeline.Decoder import split_up
# import allennlp.nn.util as util
#
#
# class PipelineModel(Model, ABC):
#
#     def __index__(self,
#                   vocab: Vocabulary,
#
#                   regularizer: RegularizerApplicator = None
#                   ):
#         super().__init__(vocab, regularizer)
#
#
#     def annotate(self, instances: List[Instance]) -> List[Dict[str, Any]]:
#         dataset = AllennlpDataset(instances, self.vocab)
#         with torch.no_grad():
#             cuda_device = self._get_prediction_device()
#
#             data_loader = self.data_loader.construct(dataset=dataset)
#             preds = []
#             for batch in data_loader:
#                 model_input = util.move_to_device(batch.as_tensor_dict(), cuda_device)
#                 outputs = self.make_output_human_readable(self(**model_input))
#                 output_dict = split_up(outputs, model_input["order_metadata"])
#                 preds.extend(output_dict)
#
#             if self.decoder is not None:
#                 preds = self.decoder.decode_batch(self.vocab, preds)
#
#             return OrderedDatasetReader.restore_order(preds)
#
#     def annotate_file(self, input_file: str, output_file: str) -> None:
#         test_instances = self.dataset_reader.read(input_file)
#         annotated = self.annotate(list(test_instances))
#
#         with open(output_file, "w") as f:
#             self.dataset_writer.write_to_file(self.vocab, annotated, f)
