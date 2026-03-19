"""
datasets.py — Data loading for sequential recommendation.

Protocol: leave-one-out
    - Last item for test
    - Second-to-last for validation
    - Rest for training

Input format: inter_*.txt with space-separated user-item interactions,
              sorted by time (each row: user item [timestamp]).
"""

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_interactions(data_path: str) -> Tuple[Dict[int, List[int]], int, int]:
    """Read interaction file and return per-user item lists.

    Supports two formats:
        ``user item``  (two columns)
        ``user item timestamp``  (three columns – sorted by timestamp assumed)

    Returns:
        user_seq:    dict mapping user_id → list[item_id] (chronological)
        num_users:   total number of users (max user id)
        num_items:   total number of items (max item id, 1-indexed)
    """
    user_seq: Dict[int, List[int]] = defaultdict(list)
    max_item = 0
    max_user = 0

    with open(data_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, i = int(parts[0]), int(parts[1])
            user_seq[u].append(i)
            max_item = max(max_item, i)
            max_user = max(max_user, u)

    return dict(user_seq), max_user, max_item


# ─────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────────────────────


class RecDataset(Dataset):
    """Base sequential recommendation dataset.

    Applies max-length truncation and left-zero padding so all sequences
    have exactly ``max_seq_len`` items.
    """

    def __init__(
        self,
        user_seq: Dict[int, List[int]],
        max_seq_len: int,
        split: str = "train",
        num_items: int = 0,
    ) -> None:
        """
        Args:
            user_seq:    per-user item sequences (full chronological list)
            max_seq_len: maximum sequence length after truncation
            split:       one of "train" | "valid" | "test"
            num_items:   vocabulary size (used for negative sampling check)
        """
        assert split in ("train", "valid", "test")
        self.max_seq_len = max_seq_len
        self.split = split
        self.num_items = num_items

        self.users: List[int] = []
        self.input_ids: List[List[int]] = []
        self.target_pos: List[int] = []

        for user, seq in user_seq.items():
            if len(seq) < 3:
                continue  # need at least train + valid + test items

            if split == "train":
                input_seq = seq[:-2]
                target = seq[-2]  # second-to-last predicts last-valid
                # For training, use all available prefixes (data augmentation)
                for end in range(1, len(input_seq) + 1):
                    prefix = input_seq[:end]
                    self.users.append(user)
                    self.input_ids.append(prefix)
                    self.target_pos.append(seq[end] if end < len(input_seq) else target)

            elif split == "valid":
                input_seq = seq[:-2]
                target = seq[-2]
                self.users.append(user)
                self.input_ids.append(input_seq)
                self.target_pos.append(target)

            else:  # test
                input_seq = seq[:-1]
                target = seq[-1]
                self.users.append(user)
                self.input_ids.append(input_seq)
                self.target_pos.append(target)

    def _pad(self, seq: List[int]) -> List[int]:
        if len(seq) >= self.max_seq_len:
            return seq[-self.max_seq_len :]
        return [0] * (self.max_seq_len - len(seq)) + seq

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(self._pad(self.input_ids[idx]), dtype=torch.long)
        target_pos = torch.tensor(self.target_pos[idx], dtype=torch.long)
        return {
            "user": torch.tensor(self.users[idx], dtype=torch.long),
            "input_ids": input_ids,
            "target_pos": target_pos,
        }


class TrainDataset(Dataset):
    """Training dataset that also provides negative samples.

    For each training sample we additionally sample one negative item
    (not in the user's history) for BPR / CE loss.
    """

    def __init__(
        self,
        user_seq: Dict[int, List[int]],
        max_seq_len: int,
        num_items: int,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.num_items = num_items

        # Build user history set for fast negative sampling
        self.user_history: Dict[int, set] = {
            u: set(s) for u, s in user_seq.items()
        }

        self.users: List[int] = []
        self.input_ids: List[List[int]] = []
        self.target_pos: List[int] = []

        for user, seq in user_seq.items():
            if len(seq) < 3:
                continue
            # Leave out last two items (val, test)
            train_seq = seq[:-2]
            for end in range(1, len(train_seq) + 1):
                self.users.append(user)
                self.input_ids.append(train_seq[:end])
                self.target_pos.append(
                    train_seq[end] if end < len(train_seq) else seq[-2]
                )

    def _pad(self, seq: List[int]) -> List[int]:
        if len(seq) >= self.max_seq_len:
            return seq[-self.max_seq_len :]
        return [0] * (self.max_seq_len - len(seq)) + seq

    def _sample_negative(self, user: int, pos: int) -> int:
        history = self.user_history.get(user, set())
        while True:
            neg = random.randint(1, self.num_items)
            if neg != pos and neg not in history:
                return neg

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user = self.users[idx]
        pos = self.target_pos[idx]
        neg = self._sample_negative(user, pos)
        input_ids = torch.tensor(self._pad(self.input_ids[idx]), dtype=torch.long)
        return {
            "user": torch.tensor(user, dtype=torch.long),
            "input_ids": input_ids,
            "target_pos": torch.tensor(pos, dtype=torch.long),
            "target_neg": torch.tensor(neg, dtype=torch.long),
        }


class EvalDataset(Dataset):
    """Evaluation dataset (valid / test).

    Full-ranking: the positive item is ranked against ALL other items
    (excluding items seen during training).
    """

    def __init__(
        self,
        user_seq: Dict[int, List[int]],
        max_seq_len: int,
        num_items: int,
        split: str = "valid",
    ) -> None:
        assert split in ("valid", "test")
        self.max_seq_len = max_seq_len
        self.num_items = num_items

        self.users: List[int] = []
        self.input_ids: List[List[int]] = []
        self.target_pos: List[int] = []
        self.train_history: List[List[int]] = []

        for user, seq in user_seq.items():
            if len(seq) < 3:
                continue
            if split == "valid":
                input_seq = seq[:-2]
                target = seq[-2]
                train_items = seq[:-2]
            else:
                input_seq = seq[:-1]
                target = seq[-1]
                train_items = seq[:-1]

            self.users.append(user)
            self.input_ids.append(input_seq)
            self.target_pos.append(target)
            self.train_history.append(train_items)

    def _pad(self, seq: List[int]) -> List[int]:
        if len(seq) >= self.max_seq_len:
            return seq[-self.max_seq_len :]
        return [0] * (self.max_seq_len - len(seq)) + seq

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(self._pad(self.input_ids[idx]), dtype=torch.long)
        return {
            "user": torch.tensor(self.users[idx], dtype=torch.long),
            "input_ids": input_ids,
            "target_pos": torch.tensor(self.target_pos[idx], dtype=torch.long),
            "train_history": torch.tensor(self.train_history[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def build_dataloaders(
    data_path: str,
    max_seq_len: int = 50,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Build train / valid / test DataLoaders from a raw interaction file.

    Returns:
        train_loader, valid_loader, test_loader, num_users, num_items
    """
    random.seed(seed)
    np.random.seed(seed)

    user_seq, num_users, num_items = load_interactions(data_path)

    train_ds = TrainDataset(user_seq, max_seq_len, num_items)
    valid_ds = EvalDataset(user_seq, max_seq_len, num_items, split="valid")
    test_ds = EvalDataset(user_seq, max_seq_len, num_items, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # needed for inter-sequence sampling
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader, num_users, num_items
