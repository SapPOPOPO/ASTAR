"""
augmenter.py — ASTAR adversarial augmenter.

Implements:
    - Unified intra + inter sequence transformation matrix T
    - Adaptive blend weight λ
    - Masked softmax over real (non-padding) pool positions
    - Own item embeddings (independent from recommender)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, LayerNorm


class ASTARAugmenter(nn.Module):
    """Adversarial Sequence Transformation Augmenter (ASTAR).

    Notation (from paper):
        B  – batch size
        L  – sequence length
        D  – hidden dimension
        K  – number of inter-sequence samples per example
    """

    def __init__(
        self,
        num_items: int,
        hidden_size: int = 256,
        max_seq_len: int = 50,
        num_heads: int = 2,
        num_layers: int = 2,
        inner_size: int = 512,
        dropout: float = 0.1,
        K: int = 4,
        tau_init: float = 10.0,
        tau_floor: float = 1.0,
        tau_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.K = K
        self.tau = tau_init
        self.tau_floor = tau_floor
        self.tau_decay = tau_decay

        # ── Own item embeddings (NOT shared with recommender) ───────────────
        # Initialised from recommender embeddings before training starts.
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = LayerNorm(hidden_size)

        # ── Sequence encoder ─────────────────────────────────────────────────
        self.encoder = Encoder(hidden_size, num_heads, num_layers, inner_size, dropout)
        self.enc_norm = LayerNorm(hidden_size)

        # ── T head: [B, L, D] → [B, L, (1+K)*L] ────────────────────────────
        pool_size = (1 + K) * max_seq_len
        self.T_head = nn.Linear(hidden_size, pool_size, bias=True)

        # ── λ head: [B, D] → [B, 1] ─────────────────────────────────────────
        self.lambda_head = nn.Linear(hidden_size, 1, bias=True)

        # ── λ formula constants (from paper design) ──────────────────────────
        # λ = MIN_LAMBDA + LAMBDA_RANGE * sigmoid(head) * length_scale
        # Range gives final λ ∈ [MIN_LAMBDA, MIN_LAMBDA + LAMBDA_RANGE] = [0.2, 0.5]
        # after clamp the ceiling extends up to 0.8 adaptively.
        self.MIN_LAMBDA: float = 0.2
        self.LAMBDA_RANGE: float = 0.3
        self.LAMBDA_NOISE_SCALE: float = 0.1

        self._init_weights()

    # ── initialisation ───────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def copy_embeddings_from(self, recommender: nn.Module) -> None:
        """Initialise own embeddings from recommender's item embeddings."""
        with torch.no_grad():
            self.item_embeddings.weight.copy_(
                recommender.item_embeddings.weight
            )

    # ── temperature schedule ─────────────────────────────────────────────────

    def step_tau(self, warmup_active: bool = False) -> None:
        """Decay temperature by tau_decay (called once per epoch after warmup)."""
        if not warmup_active:
            self.tau = max(self.tau_floor, self.tau * self.tau_decay)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _encode(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed + encode a sequence.

        Returns:
            hidden:   [B, L, D]  contextual representations
            pad_mask: [B, L]     1 for real tokens, 0 for padding
        """
        B, L = input_ids.shape
        emb = self.item_embeddings(input_ids)  # [B, L, D]
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]
        emb = emb + self.position_embeddings(pos_ids)
        emb = self.emb_norm(self.emb_dropout(emb))

        pad_mask = (input_ids > 0).long()  # [B, L]
        hidden = self.encoder(emb, pad_mask)
        hidden = self.enc_norm(hidden)
        return hidden, pad_mask

    def _sample_inter_sequences(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample K other sequences from the batch for inter-sequence mixing.

        Returns:
            sampled_ids:  [B, K*L]  item ids of sampled sequences
            sampled_emb:  [B, K*L, D]  their item embeddings
        """
        B, L = input_ids.shape
        K = self.K

        # Random permutation-based sampling (no replacement within each example)
        sampled_ids_list = []
        for b in range(B):
            # Candidate indices: all other rows
            candidates = [i for i in range(B) if i != b]
            # Sample K with replacement if batch too small
            if len(candidates) >= K:
                chosen = torch.tensor(
                    candidates[:K], device=input_ids.device
                )  # deterministic for K≤B-1
                # Randomly shuffle to avoid always picking the same K
                perm = torch.randperm(len(candidates), device=input_ids.device)[:K]
                chosen = torch.tensor(candidates, device=input_ids.device)[perm]
            else:
                # Batch is smaller than K – repeat with replacement
                idx = torch.randint(len(candidates), (K,), device=input_ids.device)
                chosen = torch.tensor(candidates, device=input_ids.device)[idx]

            # Gather K sequences each of length L → [K*L]
            chosen_seqs = input_ids[chosen]  # [K, L]
            sampled_ids_list.append(chosen_seqs.view(-1))  # [K*L]

        sampled_ids = torch.stack(sampled_ids_list, dim=0)  # [B, K*L]
        sampled_emb = self.item_embeddings(sampled_ids)  # [B, K*L, D]
        return sampled_ids, sampled_emb

    def _masked_softmax(
        self,
        T_logits: torch.Tensor,
        pool_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mask then softmax along last dimension.

        Args:
            T_logits:  [B, L, pool_size]
            pool_mask: [B, pool_size]   1 for real tokens, 0 for padding
        Returns:
            T: [B, L, pool_size]
        """
        # Expand mask for broadcasting over L positions
        mask = pool_mask.unsqueeze(1).expand_as(T_logits)  # [B, L, pool_size]
        T_logits = T_logits.masked_fill(~mask.bool(), float("-inf"))
        T = F.softmax(T_logits / self.tau, dim=-1)
        # Replace any NaN rows (all-masked) with uniform over pool
        nan_rows = T.isnan().any(dim=-1, keepdim=True)
        uniform = torch.ones_like(T) / T.size(-1)
        T = torch.where(nan_rows, uniform, T)
        return T

    # ── main augmentation forward ─────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        lambda_ceiling: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate adversarially augmented view.

        Args:
            input_ids:       [B, L]  item ids (0 = padding)
            lambda_ceiling:  upper bound for λ (adapted per epoch)

        Returns:
            aug:     [B, L, D]   augmented embedding sequence
            T:       [B, L, pool_size]  transformation matrix (for viz)
            lam:     [B, 1]      blend weights
        """
        B, L = input_ids.shape
        K = self.K

        # ── 1. Encode own sequence ───────────────────────────────────────────
        h, own_mask = self._encode(input_ids)  # [B,L,D], [B,L]

        # ── 2. Build pool: own sequence + K sampled sequences ────────────────
        S_intra = self.item_embeddings(input_ids)  # [B, L, D]
        sampled_ids, S_others = self._sample_inter_sequences(input_ids)
        # [B, (1+K)*L, D]
        S_pool = torch.cat([S_intra, S_others], dim=1)

        others_mask = (sampled_ids > 0).long()  # [B, K*L]
        pool_mask = torch.cat([own_mask, others_mask], dim=1)  # [B, (1+K)*L]

        # ── 3. Compute T logits ──────────────────────────────────────────────
        T_logits = self.T_head(h)  # [B, L, (1+K)*L]
        T = self._masked_softmax(T_logits, pool_mask)  # [B, L, (1+K)*L]

        # ── 4. Apply T to pool ───────────────────────────────────────────────
        T_S = torch.bmm(T, S_pool)  # [B, L, D]

        # ── 5. Compute λ ─────────────────────────────────────────────────────
        seq_lengths = own_mask.sum(dim=1).float()  # [B]
        length_scale = (seq_lengths / self.max_seq_len).clamp(min=0.1)  # [B]

        h_pool = self._mean_pool(h, own_mask)  # [B, D]
        lam_base = torch.sigmoid(self.lambda_head(h_pool))  # [B, 1]
        lam = self.MIN_LAMBDA + self.LAMBDA_RANGE * lam_base * length_scale.unsqueeze(1)  # [B, 1]

        # Add Gaussian noise to prevent temporal collapse
        noise = torch.randn_like(lam) * self.LAMBDA_NOISE_SCALE
        lam = (lam + noise).clamp(self.MIN_LAMBDA, lambda_ceiling)  # [B, 1]

        # ── 6. Blend ──────────────────────────────────────────────────────────
        lam3 = lam.unsqueeze(2)  # [B, 1, 1] → broadcast over L, D
        aug = lam3 * T_S + (1.0 - lam3) * S_intra  # [B, L, D]

        return aug, T, lam

    @staticmethod
    def _mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pool over non-padding positions.

        Args:
            h:    [B, L, D]
            mask: [B, L]   1 for real
        Returns:
            [B, D]
        """
        mask_f = mask.float().unsqueeze(-1)  # [B, L, 1]
        lengths = mask_f.sum(1).clamp(min=1)  # [B, 1]
        return (h * mask_f).sum(1) / lengths  # [B, D]
