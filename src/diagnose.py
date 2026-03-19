"""
diagnose.py — GAN training diagnostics for ASTAR.

Detects:
    - Temporal collapse:  augmenter produces nearly-identical outputs
                          regardless of input (mode collapse)
    - Player dominance:   one player (rec or aug) overwhelms the other
    - Gradient health:    vanishing / exploding gradients
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GANDiagnostics:
    """Collects and reports GAN training health metrics.

    Usage::

        diag = GANDiagnostics(window=50)
        # Inside training loop:
        diag.update(L_B=stats["L_B"], L_A=stats["L_A"],
                    aug=aug_batch, lam=lam_batch)
        if diag.should_report(epoch):
            report = diag.report()
            print(report)
    """

    def __init__(self, window: int = 50, report_every: int = 10) -> None:
        self.window = window
        self.report_every = report_every

        self._L_B: List[float] = []
        self._L_A: List[float] = []
        self._aug_cosines: List[float] = []  # inter-sample cosine similarity
        self._lam_values: List[float] = []
        self._rec_grad_norms: List[float] = []
        self._aug_grad_norms: List[float] = []

    # ── update ────────────────────────────────────────────────────────────

    def update(
        self,
        L_B: float,
        L_A: float,
        aug: Optional[torch.Tensor] = None,
        lam: Optional[torch.Tensor] = None,
    ) -> None:
        """Record one batch of diagnostics.

        Args:
            L_B:   recommender loss
            L_A:   augmenter loss
            aug:   [B, L, D] augmented embeddings (optional)
            lam:   [B, 1] lambda values (optional)
        """
        self._L_B.append(L_B)
        self._L_A.append(L_A)

        if aug is not None and aug.size(0) > 1:
            # Temporal collapse: if all aug vectors very similar → collapse
            with torch.no_grad():
                # Flatten to [B, L*D] and compute mean cosine of adjacent pairs
                flat = aug.view(aug.size(0), -1).float()
                flat_norm = F.normalize(flat, dim=-1)
                cosine = (flat_norm[:-1] * flat_norm[1:]).sum(-1).mean().item()
                self._aug_cosines.append(cosine)

        if lam is not None:
            self._lam_values.extend(lam.detach().cpu().view(-1).tolist())

        # Trim to window
        for lst in (self._L_B, self._L_A, self._aug_cosines, self._lam_values,
                    self._rec_grad_norms, self._aug_grad_norms):
            if len(lst) > self.window:
                del lst[: len(lst) - self.window]

    def record_grad_norms(
        self,
        recommender: nn.Module,
        augmenter: nn.Module,
    ) -> None:
        """Compute and store current gradient norms for both models."""
        rec_norm = self._grad_norm(recommender)
        aug_norm = self._grad_norm(augmenter)
        self._rec_grad_norms.append(rec_norm)
        self._aug_grad_norms.append(aug_norm)

    @staticmethod
    def _grad_norm(model: nn.Module) -> float:
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    # ── reporting ─────────────────────────────────────────────────────────

    def should_report(self, epoch: int) -> bool:
        return epoch % self.report_every == 0

    def report(self) -> Dict[str, float]:
        """Compute and return diagnostic summary."""
        result: Dict[str, float] = {}

        if self._L_B:
            result["mean_L_B"] = float(np.mean(self._L_B))
        if self._L_A:
            result["mean_L_A"] = float(np.mean(self._L_A))

        # Temporal collapse detection
        if self._aug_cosines:
            mean_cos = float(np.mean(self._aug_cosines))
            result["mean_aug_cosine_similarity"] = mean_cos
            result["temporal_collapse_risk"] = float(mean_cos > 0.95)

        # Player dominance: compare absolute magnitudes of L_B and L_A
        if self._L_B and self._L_A:
            abs_B = abs(float(np.mean(self._L_B)))
            abs_A = abs(float(np.mean(self._L_A)))
            ratio = abs_B / (abs_A + 1e-8)
            result["loss_magnitude_ratio_B_over_A"] = ratio
            # Dominance: if one player is >7x larger in magnitude
            result["player_dominance_risk"] = float(abs(np.log(ratio + 1e-8)) > 2.0)

        # λ statistics
        if self._lam_values:
            result["lambda_mean"] = float(np.mean(self._lam_values))
            result["lambda_std"] = float(np.std(self._lam_values))
            result["lambda_min"] = float(np.min(self._lam_values))
            result["lambda_max"] = float(np.max(self._lam_values))

        # Gradient health
        if self._rec_grad_norms:
            result["rec_grad_norm"] = float(np.mean(self._rec_grad_norms))
        if self._aug_grad_norms:
            result["aug_grad_norm"] = float(np.mean(self._aug_grad_norms))
            vanishing = float(np.mean(self._aug_grad_norms)) < 1e-4
            exploding = float(np.mean(self._aug_grad_norms)) > 100.0
            result["aug_grad_vanishing"] = float(vanishing)
            result["aug_grad_exploding"] = float(exploding)

        return result

    def print_report(self, epoch: int) -> None:
        """Pretty-print diagnostic report for an epoch."""
        r = self.report()
        lines = [f"=== GAN Diagnostics (epoch {epoch}) ==="]
        for k, v in sorted(r.items()):
            flag = " ⚠️" if v > 0.5 and "risk" in k else ""
            flag = flag or (" ⚠️" if v > 0.5 and "vanishing" in k else "")
            flag = flag or (" ⚠️" if v > 0.5 and "exploding" in k else "")
            lines.append(f"  {k:40s}: {v:.4f}{flag}")
        print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Standalone diagnostic probe
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def probe_continuous_input(
    recommender: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    L: int = 50,
) -> Dict[str, float]:
    """Critical diagnostic: does the recommender accept continuous augmented
    embeddings comparably to discrete ids?

    Run BEFORE full training to validate the mixed-representation design
    assumption. Uses the new get_mixed_representation() API: constructs a
    simple T (identity + shift blend) applied to the recommender's own
    item embeddings, then calls get_mixed_representation.

    Args:
        recommender: SASRec model
        input_ids:   [B, L] batch of real item ids
        device:      torch device

    Returns:
        dict with 'orig_loss', 'aug_loss', 'loss_ratio'
        (loss_ratio close to 1 → design assumption holds)
    """
    recommender.eval()
    B = input_ids.size(0)

    # Target: next item (shift by 1)
    target_pos = input_ids[:, -1].clone()
    target_neg = torch.randint(1, recommender.num_items + 1, (B,), device=device)

    # Original forward pass
    repr_orig, orig_loss = recommender(
        input_ids=input_ids,
        target_pos=target_pos,
        target_neg=target_neg,
    )

    # Construct aug_emb via a simple T: identity [B, L, L] as [B, P, L] (P=L,
    # using own sequence only as pool) blended with a shift.
    # T_new[p, j]: weight from pool position p to output position j.
    # Identity T: T_new = eye → aug_emb = org_emb.
    # Shift T: T_new[p, j] = 1 if p == j+1 (output j draws from pool p=j+1).
    S = recommender.item_embeddings(input_ids)  # [B, L, D]

    eye = torch.eye(L, device=device).unsqueeze(0).expand(B, -1, -1)  # [B, L, L]
    # Shift: output position j draws from pool position j+1
    shift = torch.zeros(L, L, device=device)
    if L > 1:
        shift[torch.arange(1, L), torch.arange(L - 1)] = 1.0  # pool[j+1] → output[j]
        shift[0, 0] = 1.0  # first output position stays
    shift = shift.unsqueeze(0).expand(B, -1, -1)

    # T_simple [B, P, L] = [B, L, L]
    T_simple = eye * 0.7 + shift * 0.3
    aug_emb = torch.einsum("bpj,bpd->bjd", T_simple, S)  # [B, L, D]

    lam = torch.full((B, 1), 0.5, device=device)
    repr_aug = recommender.get_mixed_representation(input_ids, aug_emb, lam)
    aug_loss = recommender.rec_loss(repr_aug, target_pos, target_neg)

    orig_val = orig_loss.item()
    aug_val = aug_loss.item()
    ratio = aug_val / (orig_val + 1e-8)

    # Threshold for determining if continuous inputs are acceptable.
    # A ratio > PROBE_LOSS_RATIO_THRESHOLD means augmented embeddings cause
    # significantly higher loss than original ids, suggesting the mixed
    # representation design assumption may not hold.
    _PROBE_LOSS_RATIO_THRESHOLD = 3.0

    print(f"[Probe] orig_loss={orig_val:.4f}  aug_loss={aug_val:.4f}  ratio={ratio:.3f}")
    if ratio < _PROBE_LOSS_RATIO_THRESHOLD:
        print("[Probe] ✓ Recommender accepts mixed inputs. Proceed with ASTAR.")
    else:
        print("[Probe] ✗ WARNING: Large loss increase with mixed inputs.")
        print("         Consider pre-training augmenter longer or revising design.")

    return {
        "orig_loss": orig_val,
        "aug_loss": aug_val,
        "loss_ratio": ratio,
    }
