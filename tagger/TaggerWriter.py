from typing import Dict, Any, Iterable, TextIO

from allennlp.data import Vocabulary

from allenpipeline.DatasetWriter import DatasetWriter


@DatasetWriter.register("tagging_writer")
class TaggerWriter(DatasetWriter):

    def write_to_file(self, vocab : Vocabulary, instances: Iterable[Dict[str, Any]], file : TextIO) -> None:
        for inst in instances:
            for w,t in zip(inst["words"],inst["tags"]):
                file.write(w)
                file.write("\t")
                file.write(t)
                file.write("\n")
            file.write("\n")
