import csv
import logging
from typing import List, Dict

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers import DatasetReader, Seq2SeqDatasetReader, CopyNetDatasetReader
from allennlp.data.tokenizers import CharacterTokenizer, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)


@DatasetReader.register("sigmorph_seq2seq_datasetreader")
class SigMorphonSeq2SeqDatasetReader(Seq2SeqDatasetReader):
    def __init__(self):
        super().__init__(
            source_tokenizer=CharacterTokenizer(), target_tokenizer=CharacterTokenizer()
        )

    @overrides
    def _read(self, file_path):
        """
        example line:

        korrodieren	korrodierest	V;SBJV;PRS;2;SG
        """
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(csv.reader(data_file, delimiter="\t")):
                if len(row) == 3:
                    source_sequence, target_sequence, feature_sequence = row
                elif len(row) == 2:
                    source_sequence, feature_sequence = row
                    target_sequence = ""
                else:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (row, line_num + 1)
                    )
                # add tab between source and features
                yield self.text_to_instance(
                    source_sequence + "\t" + feature_sequence, target_sequence
                )
