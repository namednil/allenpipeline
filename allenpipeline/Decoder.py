import logging
from typing import Dict, Any, List

import torch
from allennlp.common import Registrable
from allennlp.data import Vocabulary

logger = logging.getLogger(__name__)

def split_up(output_dict : Dict[str, torch.Tensor], order_metadata : List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes output from model.decode and the order metadata and creates a list with one element per instance.
    :param output_dict:
    :param order_metadata:
    :return:
    """
    warned_keys = set()
    N = len(order_metadata)
    instance_separated_output = [ {"order_metadata" : d} for d in order_metadata]

    def maybe_warn(name):
        if name not in warned_keys:
            logger.warning(f"Encountered the {name} key in the model's return dictionary which "
                       "couldn't be split by the batch size. Key will be ignored.")
            warned_keys.add(name)

    for name, output in list(output_dict.items()):
        if name == "loss":
            continue
        if isinstance(output, torch.Tensor):
            # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
            # This occurs with batch size 1, because we still want to include the loss in that case.
            if output.dim() == 0:
                output = output.unsqueeze(0)

            if output.size(0) != N:
                maybe_warn(name)
                continue
            output = output.detach().cpu().numpy()
        elif len(output) != N:
            maybe_warn(name)
            continue
        for instance_output, batch_element in zip(instance_separated_output, output):
            instance_output[name] = batch_element

    return instance_separated_output

class BatchDecoder(Registrable):

    def decode_batch(self, vocab: Vocabulary, instances : List[Dict[str,Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError()

