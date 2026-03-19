"""
utils.py — Evaluation metrics and miscellaneous helpers.

Metrics:
    HR@K, NDCG@K, MRR computed under full ranking
    (all items not in training history are candidates).
"""

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Rank-based metrics (full ranking, no negative sampling)
# ─────────────────────────────────────────────────────────────────────────────


def _get_rank(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return 1-based rank of target item within scores (higher = better).

    Args:
        scores: [B, num_items]  raw logit / dot-product scores
        target: [B]             target item ids (1-indexed)
    Returns:
        ranks: [B]  1-based rank for each example
    """
    # Score of the positive item
    pos_score = scores.gather(1, target.unsqueeze(1))  # [B, 1]
    # Count items with STRICTLY higher score than positive
    rank = (scores > pos_score).sum(dim=1) + 1  # [B]
    return rank.float()


def compute_metrics(
    scores: torch.Tensor,
    target: torch.Tensor,
    ks: Tuple[int, ...] = (5, 10, 20),
    train_history: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute HR@K and NDCG@K for a batch.

    Args:
        scores:        [B, num_items+1]  (item ids are 1-indexed; col 0 unused)
        target:        [B]               target item ids
        ks:            cut-off values
        train_history: [B, max_hist]     items to mask out (set score = -inf)
    Returns:
        dict of metric_name → mean over batch
    """
    scores = scores.clone()

    # Mask padding token (col 0)
    scores[:, 0] = float("-inf")

    # Mask training items to avoid trivial retrieval
    if train_history is not None:
        # train_history may contain zeros (padding) – skip those
        mask = train_history > 0  # [B, H]
        for b in range(scores.size(0)):
            history_b = train_history[b][mask[b]]
            scores[b, history_b] = float("-inf")

    ranks = _get_rank(scores, target)  # [B]

    results: Dict[str, float] = {}
    for k in ks:
        hit = (ranks <= k).float().mean().item()
        dcg = (1.0 / torch.log2(ranks + 1)).where(ranks <= k, torch.zeros_like(ranks))
        ndcg = dcg.mean().item()
        results[f"HR@{k}"] = hit
        results[f"NDCG@{k}"] = ndcg
    return results


def aggregate_metrics(batch_results: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-batch metric dicts over all batches."""
    agg: Dict[str, List[float]] = {}
    for d in batch_results:
        for k, v in d.items():
            agg.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in agg.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────


class EarlyStopping:
    """Stop training if the monitored metric does not improve for *patience*
    consecutive epochs."""

    def __init__(
        self,
        patience: int = 20,
        delta: float = 1e-4,
        mode: str = "max",
        save_path: Optional[str] = None,
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.save_path = save_path

        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        improved = (
            self.best_score is None
            or (self.mode == "max" and score > self.best_score + self.delta)
            or (self.mode == "min" and score < self.best_score - self.delta)
        )
        if improved:
            self.best_score = score
            self.counter = 0
            if self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# Performance trend (for adaptive λ ceiling)
# ─────────────────────────────────────────────────────────────────────────────


class PerformanceTracker:
    """Track NDCG over a sliding window to compute a trend signal in [-1, 1].

    The trend is used to adapt the λ ceiling:
        trend > 0  →  recommender improving  →  raise λ ceiling (harder aug)
        trend < 0  →  degrading / plateauing →  lower λ ceiling (easier aug)
    """

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self.history: List[float] = []

    def update(self, ndcg: float) -> None:
        self.history.append(ndcg)
        if len(self.history) > self.window * 2:
            self.history = self.history[-(self.window * 2):]

    @property
    def trend(self) -> float:
        # Sensitivity scale: maps NDCG delta of ~0.01 to a trend signal of ~0.97
        # (tanh(0.01 * 100) ≈ 0.76).  Tune this if NDCG values differ in magnitude.
        _TREND_SCALE_FACTOR = 100
        if len(self.history) < 2:
            return 0.0
        recent = self.history[-self.window :]
        older = self.history[: -self.window] if len(self.history) > self.window else self.history[:1]
        delta = np.mean(recent) - np.mean(older)
        return float(math.tanh(delta * _TREND_SCALE_FACTOR))

    @property
    def lambda_ceiling(self) -> float:
        """λ ceiling ∈ [0.2, 0.8] based on trend."""
        return float(np.clip(0.5 + 0.3 * self.trend, 0.2, 0.8))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing helpers
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoint(
    path: str,
    epoch: int,
    recommender: torch.nn.Module,
    augmenter: torch.nn.Module,
    rec_optimizer: torch.optim.Optimizer,
    aug_optimizer: torch.optim.Optimizer,
    metrics: Dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "recommender": recommender.state_dict(),
            "augmenter": augmenter.state_dict(),
            "rec_optimizer": rec_optimizer.state_dict(),
            "aug_optimizer": aug_optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str,
    recommender: torch.nn.Module,
    augmenter: Optional[torch.nn.Module] = None,
    rec_optimizer: Optional[torch.optim.Optimizer] = None,
    aug_optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    ckpt = torch.load(path, map_location=device)
    recommender.load_state_dict(ckpt["recommender"])
    if augmenter is not None and "augmenter" in ckpt:
        augmenter.load_state_dict(ckpt["augmenter"])
    if rec_optimizer is not None and "rec_optimizer" in ckpt:
        rec_optimizer.load_state_dict(ckpt["rec_optimizer"])
    if aug_optimizer is not None and "aug_optimizer" in ckpt:
        aug_optimizer.load_state_dict(ckpt["aug_optimizer"])
    return ckpt
