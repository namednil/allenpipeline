from typing import Dict, Any, Iterable, TextIO

from allennlp.common import Registrable
from allennlp.data import Vocabulary


class DatasetWriter(Registrable):
    """
    Write the result of your neural computation to an output file, for instance in the same format as your training data.
    """

    def write_to_file(self, vocab : Vocabulary, instances: Iterable[Dict[str, Any]], file : TextIO) -> None:
        """
        Write instances coming either from model.decode() or from your own BatchDecoder to a file.
        DON'T close the file in this method.
        :param vocab:
        :param instances:
        :param file:
        :return:
        """
        raise NotImplementedError()

