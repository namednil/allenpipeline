from typing import Dict, Any, Iterable, TextIO

from allennlp.data import Vocabulary

from pipeline.DatasetWriter import DatasetWriter


@DatasetWriter.register("tagging_writer")
class TaggerWriter(DatasetWriter):

    def instance_to_str(self, vocab : Vocabulary, instance: Dict[str, Any]) -> str:
        raise NotImplementedError()

    def write_to_file(self, vocab : Vocabulary, instances: Iterable[Dict[str, Any]], file : TextIO) -> None:
        for inst in instances:
            for w,t in zip(inst["words"],inst["tags"]):
                file.write(w)
                file.write("\t")
                file.write(t)
                file.write("\n")
            file.write("\n")
