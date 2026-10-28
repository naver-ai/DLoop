# coding=utf-8
# DLoop
# Copyright (c) 2026-present NAVER Corp.

"""On-the-fly training dataset (``--on-the-fly-loader``).

Raw conversation rows are tokenized in the DataLoader workers when a sample is
read, instead of in an upfront ``build_eagle3_dataset`` pass over the whole
corpus. Both paths share ``build_example_preprocessor``, so a row yields the
same ``input_ids`` / ``loss_mask`` either way.
"""

import random
from typing import Dict, Optional

import torch
from transformers import PreTrainedTokenizer

from datasets import Dataset as HFDataset

from .preprocessing import build_example_preprocessor


class OnTheFlyDataset(torch.utils.data.Dataset):
    """Tokenize raw conversation rows only when they are read.

    An ineligible row (empty, or below ``min_loss_tokens`` trainable tokens)
    is replaced by another row drawn from an index-seeded RNG, so the dataset
    keeps the raw length and every epoch reads the same samples.
    """

    _MAX_ATTEMPTS = 10

    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: PreTrainedTokenizer,
        chat_template: Optional[str],
        max_length: int,
        *,
        is_preformatted: bool = False,
        train_only_last_turn: bool = False,
        min_loss_tokens: int = 0,
    ):
        self.dataset = dataset
        self.min_loss_tokens = min_loss_tokens
        self._preprocess = build_example_preprocessor(
            tokenizer,
            chat_template,
            max_length,
            is_preformatted=is_preformatted,
            train_only_last_turn=train_only_last_turn,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        replacements = random.Random(index)
        row = index
        for _ in range(self._MAX_ATTEMPTS):
            sample = self._sample(row)
            if sample is not None:
                return sample
            row = replacements.randrange(len(self))
        raise ValueError(
            f"no eligible sample for row {index} after {self._MAX_ATTEMPTS} "
            f"attempts; rows need >= {self.min_loss_tokens} trainable tokens"
        )

    def _sample(self, row: int) -> Optional[Dict[str, torch.Tensor]]:
        record = self.dataset[row]
        processed = self._preprocess({key: [value] for key, value in record.items()})
        if not processed["input_ids"]:
            return None
        if processed["loss_mask"][0].sum() < self.min_loss_tokens:
            return None
        return {key: values[0] for key, values in processed.items()}


__all__ = ["OnTheFlyDataset"]
