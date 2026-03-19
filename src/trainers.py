"""
trainers.py — Two-phase adversarial training for ASTAR.

Training protocol per batch:
    Phase 1: Update Recommender (augmenter frozen)
        - Generate aug from augmenter (no_grad)
        - Compute L_B = L_rec_orig + γ·L_rec_aug + λ_cl·L_contrast
        - Update recommender parameters

    Phase 2: Update Augmenter (recommender frozen)
        - Re-generate aug (with gradients)
        - Compute L_A = β·L_rec_aug − α·L_contrast
        - Update augmenter parameters

Warmup strategy:
    Epochs 0..warmup_epochs: λ_ceiling=0.4 (conservative)
    After warmup: full adversarial game with adaptive λ_ceiling
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augmenter import ASTARAugmenter
from modules import NCELoss
from recommender import SASRec
from utils import PerformanceTracker, aggregate_metrics, compute_metrics


class AdvAugmentTrainer:
    """Adversarial Augmentation Trainer.

    Args:
        recommender:    SASRec model
        augmenter:      ASTARAugmenter model
        rec_optimizer:  optimizer for recommender
        aug_optimizer:  optimizer for augmenter
        device:         torch device
        gamma:          weight of L_rec_aug in recommender loss
        lambda_cl:      weight of contrastive loss in recommender loss
        alpha:          weight of -L_contrast in augmenter loss
        beta:           weight of L_rec_aug in augmenter loss
        cl_temperature: InfoNCE temperature
        warmup_epochs:  epochs with conservative λ ceiling
        perf_window:    window for performance trend computation
    """

    def __init__(
        self,
        recommender: SASRec,
        augmenter: ASTARAugmenter,
        rec_optimizer: torch.optim.Optimizer,
        aug_optimizer: torch.optim.Optimizer,
        device: torch.device,
        gamma: float = 0.5,
        lambda_cl: float = 0.1,
        alpha: float = 1.0,
        beta: float = 0.5,
        cl_temperature: float = 0.07,
        warmup_epochs: int = 20,
        perf_window: int = 5,
    ) -> None:
        self.recommender = recommender
        self.augmenter = augmenter
        self.rec_optimizer = rec_optimizer
        self.aug_optimizer = aug_optimizer
        self.device = device

        self.gamma = gamma
        self.lambda_cl = lambda_cl
        self.alpha = alpha
        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.grad_clip_norm: float = 5.0  # gradient clipping threshold

        self.nce_loss = NCELoss(temperature=cl_temperature)
        self.perf_tracker = PerformanceTracker(window=perf_window)

        self._current_epoch = 0

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def is_warmup(self) -> bool:
        return self._current_epoch < self.warmup_epochs

    @property
    def lambda_ceiling(self) -> float:
        if self.is_warmup:
            return 0.4
        return self.perf_tracker.lambda_ceiling

    # ── phase helpers ──────────────────────────────────────────────────────

    def _generate_aug(self, input_ids: torch.Tensor, grad: bool = False):
        """Generate augmented embedding.

        Args:
            input_ids: [B, L]
            grad:      whether to enable gradients for augmenter
        Returns:
            aug: [B, L, D]   augmented embeddings
            T:   [B, L, P]   transformation matrix
            lam: [B, 1]      blend weights
        """
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            aug, T, lam = self.augmenter(input_ids, self.lambda_ceiling)
        return aug, T, lam

    def _contrast_loss(
        self,
        input_ids: torch.Tensor,
        aug: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss between orig and aug views."""
        repr_orig = self.recommender.get_representation(input_ids=input_ids)
        repr_aug = self.recommender.get_representation(inputs_embeds=aug)
        return self.nce_loss(repr_orig, repr_aug)

    # ── Phase 1: recommender update ───────────────────────────────────────

    def _phase1_rec(
        self,
        input_ids: torch.Tensor,
        target_pos: torch.Tensor,
        target_neg: torch.Tensor,
    ) -> Dict[str, float]:
        """Update recommender parameters."""
        self.recommender.train()
        self.augmenter.eval()
        self.rec_optimizer.zero_grad()

        # Generate aug without gradients (augmenter fixed)
        aug, _, _ = self._generate_aug(input_ids, grad=False)
        aug = aug.detach()

        # L_rec_orig: recommendation loss on original sequence
        repr_orig, L_rec_orig = self.recommender(
            input_ids=input_ids,
            target_pos=target_pos,
            target_neg=target_neg,
        )

        # L_rec_aug: recommendation consistency on augmented view
        repr_aug, L_rec_aug = self.recommender(
            inputs_embeds=aug,
            target_pos=target_pos,
            target_neg=target_neg,
        )

        # L_contrast: pull together orig and aug representations
        L_contrast = self.nce_loss(repr_orig, repr_aug)

        L_B = L_rec_orig + self.gamma * L_rec_aug + self.lambda_cl * L_contrast
        L_B.backward()
        torch.nn.utils.clip_grad_norm_(self.recommender.parameters(), self.grad_clip_norm)
        self.rec_optimizer.step()

        return {
            "L_rec_orig": L_rec_orig.item(),
            "L_rec_aug": L_rec_aug.item(),
            "L_contrast": L_contrast.item(),
            "L_B": L_B.item(),
        }

    # ── Phase 2: augmenter update ─────────────────────────────────────────

    def _phase2_aug(
        self,
        input_ids: torch.Tensor,
        target_pos: torch.Tensor,
        target_neg: torch.Tensor,
    ) -> Dict[str, float]:
        """Update augmenter parameters."""
        self.recommender.eval()
        self.augmenter.train()
        self.aug_optimizer.zero_grad()

        # Re-generate aug WITH gradients for augmenter
        aug, _, lam = self._generate_aug(input_ids, grad=True)

        # L_rec_aug through frozen recommender
        with torch.no_grad():
            repr_orig = self.recommender.get_representation(input_ids=input_ids)
        repr_aug = self.recommender.get_representation(inputs_embeds=aug)
        L_rec_aug = self.recommender.rec_loss(repr_aug, target_pos, target_neg)

        # L_contrast: maximise distance (adversarial)
        L_contrast = self.nce_loss(repr_orig.detach(), repr_aug)

        # Augmenter wants: preserve task info (beta * rec_loss) AND fool contrast (-alpha)
        L_A = self.beta * L_rec_aug - self.alpha * L_contrast
        L_A.backward()
        torch.nn.utils.clip_grad_norm_(self.augmenter.parameters(), self.grad_clip_norm)
        self.aug_optimizer.step()

        return {
            "L_rec_aug_A": L_rec_aug.item(),
            "L_contrast_A": L_contrast.item(),
            "L_A": L_A.item(),
            "lambda_mean": lam.mean().item(),
        }

    # ── epoch-level train ─────────────────────────────────────────────────

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """Run one training epoch (both phases).

        Returns:
            mean losses over batches
        """
        epoch_stats: List[Dict[str, float]] = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(self.device)
            target_pos = batch["target_pos"].to(self.device)
            target_neg = batch["target_neg"].to(self.device)

            # Phase 1 – update recommender
            stats1 = self._phase1_rec(input_ids, target_pos, target_neg)

            # Phase 2 – update augmenter
            stats2 = self._phase2_aug(input_ids, target_pos, target_neg)

            combined = {**stats1, **stats2}
            epoch_stats.append(combined)

        return aggregate_metrics(epoch_stats)

    # ── evaluation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        ks: Tuple[int, ...] = (5, 10, 20),
    ) -> Dict[str, float]:
        """Full-ranking evaluation.

        Returns:
            dict of HR@K, NDCG@K metrics
        """
        self.recommender.eval()
        batch_results = []

        for batch in eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            target_pos = batch["target_pos"].to(self.device)
            train_history = batch.get("train_history", None)
            if train_history is not None:
                train_history = train_history.to(self.device)

            repr_ = self.recommender.get_representation(input_ids=input_ids)
            scores = self.recommender.score_all_items(repr_)

            metrics = compute_metrics(scores, target_pos, ks=ks, train_history=train_history)
            batch_results.append(metrics)

        return aggregate_metrics(batch_results)

    # ── full training loop ─────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 200,
        patience: int = 20,
        save_dir: str = "checkpoints",
        verbose: bool = True,
        log_interval: int = 1,
    ) -> Dict[str, float]:
        """Train ASTAR end-to-end.

        Returns:
            best metrics on test set (evaluated at best validation epoch)
        """
        os.makedirs(save_dir, exist_ok=True)
        best_ndcg = -1.0
        best_epoch = 0
        patience_counter = 0
        best_ckpt = os.path.join(save_dir, "best_model.pt")

        for epoch in range(1, num_epochs + 1):
            self._current_epoch = epoch

            t0 = time.time()
            train_stats = self.train_epoch(train_loader)
            elapsed = time.time() - t0

            # Step temperature (only after warmup)
            self.augmenter.step_tau(warmup_active=self.is_warmup)

            if epoch % log_interval == 0:
                val_metrics = self.evaluate(valid_loader)
                ndcg10 = val_metrics.get("NDCG@10", 0.0)

                # Update performance tracker
                self.perf_tracker.update(ndcg10)

                if verbose:
                    print(
                        f"Epoch {epoch:4d} | "
                        f"L_B={train_stats.get('L_B', 0):.4f} "
                        f"L_A={train_stats.get('L_A', 0):.4f} "
                        f"λ_mean={train_stats.get('lambda_mean', 0):.3f} "
                        f"λ_ceil={self.lambda_ceiling:.2f} "
                        f"tau={self.augmenter.tau:.2f} | "
                        f"Val NDCG@10={ndcg10:.4f} | "
                        f"t={elapsed:.1f}s"
                    )

                # Early stopping
                if ndcg10 > best_ndcg:
                    best_ndcg = ndcg10
                    best_epoch = epoch
                    patience_counter = 0
                    # Save both models
                    torch.save(
                        {
                            "epoch": epoch,
                            "recommender": self.recommender.state_dict(),
                            "augmenter": self.augmenter.state_dict(),
                            "val_metrics": val_metrics,
                        },
                        best_ckpt,
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch} (best={best_epoch})")
                        break

        # Load best model and evaluate on test set
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=self.device)
            self.recommender.load_state_dict(ckpt["recommender"])

        test_metrics = self.evaluate(test_loader)
        if verbose:
            print(f"\nTest metrics (epoch {best_epoch}):")
            for k, v in sorted(test_metrics.items()):
                print(f"  {k}: {v:.4f}")

        return test_metrics
