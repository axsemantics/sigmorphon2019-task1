from typing import Dict, Iterable, Set, Tuple, List, Union, Any

import torch
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.models.encoder_decoders.copynet_seq2seq import CopyNetSeq2Seq
from allennlp.models.model import Model
from allennlp.modules import Attention, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.training.metrics import CategoricalAccuracy, MeanAbsoluteError
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric, BLEU
from allennlp.nn.beam_search import BeamSearch

from .metrics import AverageEditDistance, SequenceAccuracy


@Model.register("ax_seq2seq")
class AX_Seq2Seq(SimpleSeq2Seq):
    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        max_decoding_steps: int,
        attention: Attention = None,
        attention_function: SimilarityFunction = None,
        beam_size: int = None,
        target_namespace: str = "tokens",
        target_embedding_dim: int = None,
        scheduled_sampling_ratio: float = 0.0,
        use_bleu: bool = True,
    ) -> None:
        super().__init__(
            vocab,
            source_embedder,
            encoder,
            max_decoding_steps,
            attention,
            attention_function,
            beam_size,
            target_namespace,
            target_embedding_dim,
            scheduled_sampling_ratio,
            use_bleu,
        )
        pad_index = self.vocab.get_token_index(
            self.vocab._padding_token, self._target_namespace
        )
        self._accuracy = SequenceAccuracy(
            exclude_indices={pad_index, self._end_index, self._start_index}
        )
        self._distance = AverageEditDistance(
            exclude_indices={pad_index, self._end_index, self._start_index}
        )

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: Dict[str, torch.LongTensor],
        target_tokens: Dict[str, torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        state = self._encode(source_tokens)

        if target_tokens:
            state = self._init_decoder_state(state)
            output_dict = self._forward_loop(state, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                if self._bleu:
                    self._bleu(best_predictions, target_tokens["tokens"])
                if self._accuracy:
                    self._accuracy(best_predictions, target_tokens["tokens"])
                if self._distance:
                    self._distance(best_predictions, target_tokens["tokens"])

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            if self._accuracy:
                all_metrics.update(self._accuracy.get_metric(reset=reset))
            if self._distance:
                all_metrics.update(self._distance.get_metric(reset=reset))
            if self._bleu:
                all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics
