from typing import Dict, Iterable, List

from allennlp.data import TokenIndexer, Instance, Token, Field, DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer

from allenpipeline.OrderedDatasetReader import OrderedDatasetReader


@DatasetReader.register("tagging_reader")
class TaggerReader(OrderedDatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def read_file(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            sent : List[Token] = []
            sent_tags : List[str] = []
            sent_number = 0

            for i,line in enumerate(f):
                line = line.rstrip("\n")
                if line == "":
                    if len(sent) > 0 and len(sent_tags) > 0 and len(sent) != len(sent_tags):
                        raise ValueError(f"Some words seem not to have tags: {[w.text for w in sent]} has tags {str(sent_tags)}")
                    else:
                        yield self.tokens_to_instance(sent, sent_tags)
                    sent = []
                    sent_tags = []
                    sent_number += 1
                else:
                    info = line.split("\t")
                    if len(info) == 1:
                        sent.append(Token(info[0]))
                    elif len(info) == 2:
                        word, tag = info
                        sent.append((Token(word)))
                        sent_tags.append(tag)
                    else:
                        raise ValueError(f"Illegal number of columns in line {i}: {info}")

            if len(sent) > 0:
                if len(sent) > 0 and len(sent_tags) > 0 and len(sent) != len(sent_tags):
                    raise ValueError(f"Issue with tags: {[w.text for w in sent]} has tags {str(sent_tags)}")
                else:
                    yield self.tokens_to_instance(sent, sent_tags)

    def tokens_to_instance(self,  # type: ignore
                           full_tokens: List[Token],
                           tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here.
        """
        sequence = TextField(full_tokens, self._token_indexers)

        instance_fields: Dict[str, Field] = {'tokens': sequence }

        instance_fields["metadata"] = MetadataField({"words" : [token.text for token in full_tokens]})

        # Add tags to instance
        if tags is not None and len(tags) > 0:
            instance_fields['tags'] = SequenceLabelField(tags, sequence, "labels")

        return Instance(instance_fields)

    def is_annotated(self, instance : Instance) -> bool:
        return "tags" in instance