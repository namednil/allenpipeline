from typing import Dict, Any, Iterable, TextIO

from allennlp.common import Registrable
from allennlp.data import Vocabulary


class DatasetWriter(Registrable):


    def write_to_file(self, vocab : Vocabulary, instances: Iterable[Dict[str, Any]], file : TextIO) -> None:
        raise NotImplementedError()

