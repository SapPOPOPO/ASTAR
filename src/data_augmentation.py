"""
data_augmentation.py — CoSeRec augmentation primitives.

Augmentors operate on Python lists of item ids (non-padded, no zeros).
All augmentors handle edge cases: sequences of length <= 1 are returned unchanged.
"""

import random
from typing import List, Optional


class Crop:
    """Crop a contiguous subsequence of length floor(tao * len)."""

    def __init__(self, tao: float = 0.2) -> None:
        self.tao = tao

    def __call__(self, seq: List[int]) -> List[int]:
        n = len(seq)
        if n <= 1:
            return seq[:]
        crop_len = max(1, int(n * self.tao))
        start = random.randint(0, n - crop_len)
        return seq[start : start + crop_len]


class Mask:
    """Randomly replace items with mask_id at rate (1 - gamma)."""

    def __init__(self, gamma: float = 0.7, mask_id: int = 0) -> None:
        self.gamma = gamma
        self.mask_id = mask_id

    def __call__(self, seq: List[int]) -> List[int]:
        n = len(seq)
        if n <= 1:
            return seq[:]
        result = seq[:]
        for i in range(n):
            if random.random() > self.gamma:
                result[i] = self.mask_id
        return result


class Reorder:
    """Shuffle a contiguous sub-window of length floor(beta * len)."""

    def __init__(self, beta: float = 0.2) -> None:
        self.beta = beta

    def __call__(self, seq: List[int]) -> List[int]:
        n = len(seq)
        if n <= 1:
            return seq[:]
        reorder_len = max(2, int(n * self.beta))
        start = random.randint(0, n - reorder_len)
        result = seq[:]
        sub = result[start : start + reorder_len]
        random.shuffle(sub)
        result[start : start + reorder_len] = sub
        return result


class Substitute:
    """Substitute items using an item similarity model.

    If item_similarity_model is None, returns the sequence unchanged (no-op).
    """

    def __init__(
        self,
        item_similarity_model=None,
        substitute_rate: float = 0.1,
    ) -> None:
        self.item_similarity_model = item_similarity_model
        self.substitute_rate = substitute_rate

    def __call__(self, seq: List[int]) -> List[int]:
        if self.item_similarity_model is None:
            return seq[:]
        n = len(seq)
        if n <= 1:
            return seq[:]
        result = seq[:]
        for i in range(n):
            if random.random() < self.substitute_rate:
                similar = self.item_similarity_model.most_similar(result[i])
                if similar:
                    result[i] = random.choice(similar)
        return result


class Insert:
    """Insert similar items at random positions.

    If item_similarity_model is None, returns the sequence unchanged (no-op).
    """

    def __init__(
        self,
        item_similarity_model=None,
        insert_rate: float = 0.1,
    ) -> None:
        self.item_similarity_model = item_similarity_model
        self.insert_rate = insert_rate

    def __call__(self, seq: List[int]) -> List[int]:
        if self.item_similarity_model is None:
            return seq[:]
        n = len(seq)
        if n <= 1:
            return seq[:]
        result = []
        for item in seq:
            result.append(item)
            if random.random() < self.insert_rate:
                similar = self.item_similarity_model.most_similar(item)
                if similar:
                    result.append(random.choice(similar))
        return result


class Random:
    """Randomly pick one augmentation from the available set each call.

    If no similarity model is provided, only uses Crop, Mask, and Reorder.
    """

    def __init__(
        self,
        tao: float = 0.2,
        gamma: float = 0.7,
        beta: float = 0.2,
        item_similarity_model=None,
        substitute_rate: float = 0.1,
        insert_rate: float = 0.1,
    ) -> None:
        self.augmentors = [
            Crop(tao=tao),
            Mask(gamma=gamma),
            Reorder(beta=beta),
        ]
        if item_similarity_model is not None:
            self.augmentors.append(Substitute(item_similarity_model, substitute_rate))
            self.augmentors.append(Insert(item_similarity_model, insert_rate))

    def __call__(self, seq: List[int]) -> List[int]:
        aug = random.choice(self.augmentors)
        return aug(seq)
