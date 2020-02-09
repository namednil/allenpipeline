from typing import Dict, Any, Iterable, TextIO

from allennlp.common import Registrable
from allennlp.data import Vocabulary


class DatasetWriter(Registrable):

    def instance_to_str(self, vocab : Vocabulary, instance: Dict[str, Any]) -> str:
        raise NotImplementedError()

    def write_to_file(self, vocab : Vocabulary, instances: Iterable[Dict[str, Any]], file : TextIO) -> None:
        raise NotImplementedError()

