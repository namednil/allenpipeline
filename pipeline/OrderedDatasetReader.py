from typing import Iterable, Dict, Any, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField


class OrderedDatasetReader(DatasetReader):
    """
    DatasetReader that tracks for each instance
    - if it is annotated completely, that is, if it can serve as (supervised) training data
    - where it came from in the corpus
    """

    def __init__(self, lazy:bool = False) -> None:
        super().__init__(lazy=lazy)

    def _read(self, file_path: str) -> Iterable[Instance]:
        for i,instance in enumerate(self.read_file(file_path)):
            instance.add_field("order_metadata",MetadataField({"position_in_corpus" : i, "is_annotated" : self.is_annotated(instance)}))
            yield instance

    def is_annotated(self, instance : Instance) -> bool:
        raise NotImplementedError("is_annotated(instance : Instance) -> bool of your OrderedDatasetReader must be overriden.")

    @staticmethod
    def restore_order(instances : Iterable[Dict[str, Any]]) -> List[Dict[str,Any]]:
        """
        Tries to restore the order of the instances that got mixed up during batching.
        :param instances:
        :return:
        """
        return sorted(instances, key=lambda instance: instance["order_metadata"]["position_in_corpus"])

    def read_file(self, file_path) -> Iterable[Instance]:
        raise NotImplementedError()





