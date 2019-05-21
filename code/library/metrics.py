import math
from collections import Counter
from typing import Dict, Iterable, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.training.metrics.metric import Metric


class SequenceAccuracy(Metric):
    def __init__(self, *, exclude_indices: Set[int] = None) -> None:
        self._exclude_indices = exclude_indices or set()
        self.reset()

    @overrides
    def reset(self) -> None:
        self.correct_count = 0.0
        self.total_count = 0.0

    @overrides
    def __call__(
        self, predictions: torch.LongTensor, gold_targets: torch.LongTensor
    ) -> None:

        assert len(predictions) == len(gold_targets)

        for src, trg in zip(predictions, gold_targets):
            correct = 0
            count = 0

            # remove all excluded_indices
            mask = get_mask(self._exclude_indices, trg)
            _trg = trg.masked_select(mask)

            mask = get_mask(self._exclude_indices, src)
            _src = src.masked_select(mask)

            for src_item, trg_item in zip(_src, _trg):
                if src_item == trg_item:
                    correct += 1
                count += 1

            self.total_count += count
            self.correct_count += correct

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count) * 100
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return {"acc": accuracy}


class AverageEditDistance(Metric):
    def __init__(self, *, exclude_indices: Set[int] = None) -> None:
        self._exclude_indices = exclude_indices or set()
        self.reset()

    @overrides
    def reset(self) -> None:
        self.distance = 0.0
        self.count = 0.0

    @overrides
    def __call__(
        self, predictions: torch.LongTensor, gold_targets: torch.LongTensor
    ) -> None:

        assert len(predictions) == len(gold_targets)

        for src, trg in zip(predictions, gold_targets):
            # remove all excluded_indices
            mask = get_mask(self._exclude_indices, trg)
            _trg = trg.masked_select(mask)

            mask = get_mask(self._exclude_indices, src)
            _src = src.masked_select(mask)

            self.distance += edit_distance(_src, _trg)
            self.count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self.count > 1e-12:
            distance = float(self.distance) / float(self.count)
        else:
            distancte = 0.0
        if reset:
            self.reset()
        return {"dist": distance}


def get_mask(exclude_indices: Set[int], tensor: torch.LongTensor) -> torch.ByteTensor:
    mask = torch.ones(tensor.size(), dtype=torch.uint8, device=tensor.device)
    for idx in exclude_indices:
        mask = mask & (tensor != idx)
    return mask


# https://github.com/sigmorphon/crosslingual-inflection-baseline/blob/master/src/util.py
def edit_distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(
                table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + dg
            )
    return int(table[len(str2)][len(str1)])
