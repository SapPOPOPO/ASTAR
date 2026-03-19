"""
recommender.py — SASRec recommender with inputs_embeds support.

The recommender accepts EITHER:
    input_ids      (discrete token ids)  →  standard usage
    inputs_embeds  (continuous embeddings)  →  used by the augmenter

This dual interface is the key modification enabling ASTAR's pre-encoder
continuous augmentation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, LayerNorm, NCELoss

# Threshold below which an embedding's L1 norm is treated as a padding row
# (used when inferring the padding mask from inputs_embeds).
_EMBEDDING_PADDING_THRESHOLD: float = 1e-6


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model.

    Modifications vs. vanilla SASRec:
        - transformer_encoder() accepts both input_ids and inputs_embeds
        - forward() returns logits over full item catalogue
        - get_representation() returns CLS-like last-non-pad embedding
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
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # +1 for padding token (id=0)
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = LayerNorm(hidden_size)

        self.encoder = Encoder(hidden_size, num_heads, num_layers, inner_size, dropout)
        self.out_norm = LayerNorm(hidden_size)
        self.output_bias = nn.Parameter(torch.zeros(num_items + 1))

        self._init_weights()

    # ── initialisation ──────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    # ── position ids ────────────────────────────────────────────────────────

    def _position_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    # ── core encoder (dual interface) ────────────────────────────────────────

    def transformer_encoder(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the transformer stack.

        Exactly one of input_ids / inputs_embeds must be provided.

        Args:
            input_ids:     [B, L]     long tensor of item ids
            inputs_embeds: [B, L, D]  float tensor of continuous embeddings
        Returns:
            [B, L, D]  contextual representations
        """
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Provide either input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.item_embeddings(input_ids)  # [B, L, D]

        B, L, D = inputs_embeds.shape
        pos_ids = self._position_ids(L, inputs_embeds.device)  # [1, L]
        pos_emb = self.position_embeddings(pos_ids)  # [1, L, D]

        hidden = self.emb_norm(self.emb_dropout(inputs_embeds + pos_emb))

        # Build padding mask: real tokens = 1, padding = 0
        if input_ids is not None:
            pad_mask = (input_ids > 0).long()  # [B, L]
        else:
            # Infer mask from embedding norms (all-zero rows → padding)
            pad_mask = (inputs_embeds.abs().sum(-1) > _EMBEDDING_PADDING_THRESHOLD).long()  # [B, L]

        hidden = self.encoder(hidden, pad_mask)
        hidden = self.out_norm(hidden)
        return hidden

    # ── sequence-level representation ───────────────────────────────────────

    def get_representation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the representation at the last non-padding position.

        Used as the sequence embedding for contrastive learning.

        Returns:
            [B, D]
        """
        hidden = self.transformer_encoder(input_ids=input_ids, inputs_embeds=inputs_embeds)

        if input_ids is not None:
            seq_lengths = (input_ids > 0).sum(dim=1) - 1  # 0-indexed last real pos
        else:
            seq_lengths = (inputs_embeds.abs().sum(-1) > _EMBEDDING_PADDING_THRESHOLD).sum(dim=1) - 1

        seq_lengths = seq_lengths.clamp(min=0)
        # Gather last non-pad position
        idx = seq_lengths.unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden.size(-1))
        repr_ = hidden.gather(1, idx).squeeze(1)  # [B, D]
        return repr_

    # ── scoring ─────────────────────────────────────────────────────────────

    def score_all_items(self, repr_: torch.Tensor) -> torch.Tensor:
        """Inner product against all item embeddings.

        Args:
            repr_: [B, D]
        Returns:
            [B, num_items+1]  logits for item 0 … num_items
        """
        w = self.item_embeddings.weight  # [V, D]
        logits = torch.matmul(repr_, w.T) + self.output_bias  # [B, V]
        return logits

    # ── augmented view ───────────────────────────────────────────────────────

    def get_mixed_representation(
        self,
        input_ids: torch.Tensor,   # [B, L]
        T: torch.Tensor,           # [B, P, L]  (hard or soft)
        pool_ids: torch.Tensor,    # [B, P]
        lam: torch.Tensor,         # [B, 1]
    ) -> torch.Tensor:
        """Produce the λ-blended augmented sequence representation.

        Looks up embeddings from the recommender's own table, applies T to
        select/blend pool embeddings, then λ-blends with the original sequence
        before passing through the transformer encoder.

        Returns:
            [B, D]  sequence representation of the blended view
        """
        org_emb  = self.item_embeddings(input_ids)                        # [B, L, D]
        pool_emb = self.item_embeddings(pool_ids)                         # [B, P, D]
        aug_emb  = torch.einsum('bpl,bpd->bld', T, pool_emb)             # [B, L, D]
        mixed    = lam.unsqueeze(-1) * aug_emb + (1 - lam.unsqueeze(-1)) * org_emb
        return self.get_representation(inputs_embeds=mixed)

    # ── losses ───────────────────────────────────────────────────────────────

    def rec_loss(
        self,
        repr_: torch.Tensor,
        target_pos: torch.Tensor,
        target_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Binary cross-entropy loss on positive / negative items.

        If target_neg is None, uses full softmax cross-entropy over
        all items (slower but exact).
        """
        if target_neg is None:
            logits = self.score_all_items(repr_)  # [B, V]
            return F.cross_entropy(logits, target_pos)

        pos_emb = self.item_embeddings(target_pos)  # [B, D]
        neg_emb = self.item_embeddings(target_neg)  # [B, D]

        pos_score = (repr_ * pos_emb).sum(-1)  # [B]
        neg_score = (repr_ * neg_emb).sum(-1)  # [B]

        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return loss

    # ── full forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        target_pos: Optional[torch.Tensor] = None,
        target_neg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            repr_: [B, D]          sequence representation
            loss:  scalar or None  recommendation loss (if targets provided)
        """
        repr_ = self.get_representation(input_ids=input_ids, inputs_embeds=inputs_embeds)
        loss = None
        if target_pos is not None:
            loss = self.rec_loss(repr_, target_pos, target_neg)
        return repr_, loss
