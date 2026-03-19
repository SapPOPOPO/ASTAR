"""
modules.py — Shared building blocks for ASTAR.

Contains: LayerNorm, SelfAttentionBlock, FeedForwardBlock,
          Encoder (SASRec-style transformer), NCELoss (InfoNCE).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalisation with learnable gain and bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention block (SASRec style, causal)."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(hidden_size)
        self.out_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, L, d]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, d = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, L, H * d)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden
        hidden = self.layer_norm(hidden)

        Q = self._split_heads(self.query(hidden))
        K = self._split_heads(self.key(hidden))
        V = self._split_heads(self.value(hidden))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = scores + attention_mask
        weights = self.attn_dropout(F.softmax(scores, dim=-1))

        context = self._merge_heads(torch.matmul(weights, V))
        out = self.out_dropout(self.out_proj(context))
        return residual + out


class FeedForwardBlock(nn.Module):
    """Position-wise feed-forward block with residual + layer norm."""

    def __init__(self, hidden_size: int, inner_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, inner_size)
        self.fc2 = nn.Linear(inner_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(self.act(self.fc1(hidden)))
        hidden = self.dropout(self.fc2(hidden))
        return residual + hidden


class TransformerBlock(nn.Module):
    """One SASRec transformer block (self-attn + FFN)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        inner_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = SelfAttentionBlock(hidden_size, num_heads, dropout)
        self.ffn = FeedForwardBlock(hidden_size, inner_size, dropout)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.attention(hidden, attention_mask)
        hidden = self.ffn(hidden)
        return hidden


class Encoder(nn.Module):
    """Stack of TransformerBlocks with causal masking support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        inner_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, inner_size, dropout)
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def _build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper triangular mask (additive, -inf above diagonal)."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

    def forward(
        self, hidden: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden:   [B, L, D] input embeddings
            pad_mask: [B, L]    1 for real tokens, 0 for padding
        Returns:
            [B, L, D] contextual representations
        """
        B, L, D = hidden.shape
        causal = self._build_causal_mask(L, hidden.device)
        # Convert padding mask to additive bias: [B, 1, 1, L]
        pad_bias = (1.0 - pad_mask.float()).unsqueeze(1).unsqueeze(2) * float("-inf")
        attn_mask = causal + pad_bias  # broadcast to [B, 1, L, L]
        # Replace nan (0 * -inf) with 0 to avoid gradient issues
        attn_mask = torch.nan_to_num(attn_mask, nan=0.0, posinf=0.0, neginf=float("-inf"))

        for layer in self.layers:
            hidden = layer(hidden, attn_mask)
        return hidden


class NCELoss(nn.Module):
    """InfoNCE contrastive loss (normalised temperature cross-entropy).

    Pulls together positive pairs (z1[i], z2[i]) while pushing apart
    negatives within the batch.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: [B, D] view 1 representations (L2-normalised)
            z2: [B, D] view 2 representations (L2-normalised)
        Returns:
            scalar loss
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        B = z1.size(0)
        # Similarity matrix [B, B]
        sim = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(B, device=z1.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss
