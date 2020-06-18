from typing import Iterable, Dict, Any, List, Optional

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField


class OrderedDatasetReader(DatasetReader):
    """
    DatasetReader that tracks for each instance where it came from in the corpus.
    """

    def __init__(self,  lazy: bool = False,
                 cache_directory: Optional[str] = None,
                 max_instances: Optional[int] = None,
                 manual_distributed_sharding: bool = False,
                 manual_multi_process_sharding: bool = False,) -> None:
        super().__init__(lazy, cache_directory, max_instances, manual_distributed_sharding, manual_multi_process_sharding)

    def _read(self, file_path: str) -> Iterable[Instance]:
        for i,instance in enumerate(self.read_file(file_path)):
            instance.add_field("order_metadata",MetadataField({"position_in_corpus": i}))
            yield instance

    @staticmethod
    def restore_order(instances : Iterable[Dict[str, Any]]) -> List[Dict[str,Any]]:
        """
        Tries to restore the order of the instances that got mixed up during batching.
        :param instances:
        :return:
        """
        return sorted(instances, key=lambda instance: instance["order_metadata"]["position_in_corpus"])

    def read_file(self, file_path) -> Iterable[Instance]:
        """
        Performs the task of _read in the usual DatasetReader.
        Part of the OrderedDatasetReader, needed to add meta-data easily.
        :param file_path:
        :return:
        """
        raise NotImplementedError()





