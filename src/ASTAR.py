"""
ASTAR.py — ASTAR adversarial transport augmenters.

ASTARAugmenter:
    - Unified intra + inter sequence transformation matrix T
    - Adaptive blend weight λ
    - Masked softmax over real (non-padding) pool positions

ASTARv2Augmenter (subclass):
    - Dual T_heads producing two independent augmented views
    - View-to-view contrastive learning compatible interface
    - Entropy + usage-balance regularization to prevent transport collapse
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, LayerNorm


class _EncArgs:
    """Minimal args namespace for constructing modules.Encoder."""

    def __init__(self, hidden_size: int, num_heads: int, num_layers: int, dropout: float) -> None:
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_hidden_layers = num_layers
        self.hidden_act = "gelu"
        self.attention_probs_dropout_prob = dropout
        self.hidden_dropout_prob = dropout
        self.initializer_range = 0.02


class ASTARAugmenter(nn.Module):
    """Adversarial Sequence Transformation Augmenter (ASTAR).

    Notation (from paper):
        B  : batch size
        L  : sequence length
        D  : hidden dimension
        K  : number of inter-sequence samples per example
        P  : pool size = (1+K)*L

    forward() returns T [B, P, L], pool_ids [B, P], lam [B, 1].
    Embedding lookup and λ-blending are delegated to the recommender.
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

        # ── Own item embeddings (for context encoding only, not for output) ──
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = LayerNorm(hidden_size)

        # ── Sequence encoder ─────────────────────────────────────────────────
        # Build an args-compatible namespace so we can reuse modules.Encoder
        _enc_args = _EncArgs(hidden_size, num_heads, num_layers, dropout)
        self.encoder = Encoder(_enc_args)
        self.enc_norm = LayerNorm(hidden_size)

        # ── T head: h [B, L, D] → logits [B, L, P], transposed to [B, P, L] ─
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

        # Stores T_logits [B, P, L] from the most recent forward pass so the
        # trainer can apply hard Gumbel ST without re-running the augmenter.
        self._last_T_logits: Optional[torch.Tensor] = None
        # Stores T_logits after masked_fill (padding positions set to -inf) so
        # the hard Gumbel ST in Phase 1 cannot sample from padding pool positions.
        self._last_T_logits_masked: Optional[torch.Tensor] = None

        self._init_weights()

    # ── initialisation ───────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

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

        # Build attention mask in the format expected by modules.Encoder:
        # shape [B, 1, 1, L], with -10000 for padding positions.
        att_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        att_mask = att_mask.to(dtype=emb.dtype)
        att_mask = (1.0 - att_mask) * -10000.0

        all_layers = self.encoder(emb, att_mask, output_all_encoded_layers=False)
        hidden = all_layers[-1]  # [B, L, D]
        hidden = self.enc_norm(hidden)
        return hidden, pad_mask

    def _sample_inter_sequences(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Sample K other sequences from the batch for inter-sequence mixing.

        Returns:
            sampled_ids: [B, K*L]  item ids of sampled sequences
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
        return sampled_ids

    def _masked_softmax(
        self,
        T_logits: torch.Tensor,
        pool_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mask then softmax over pool dimension (dim=1).

        Args:
            T_logits:  [B, P, L]  — pool positions × output positions
            pool_mask: [B, P]     1 for real pool tokens, 0 for padding
        Returns:
            T: [B, P, L]  (each column sums to 1 over pool dim=1)
        """
        # Expand mask for broadcasting over L output positions
        mask = pool_mask.unsqueeze(-1).expand_as(T_logits)  # [B, P, L]
        T_logits = T_logits.masked_fill(~mask.bool(), float("-inf"))
        self._last_T_logits_masked = T_logits  # stored for hard Gumbel ST (padding already masked)
        T = F.softmax(T_logits / self.tau, dim=1)
        # Replace any NaN columns (all-masked pool for a given output position) with
        # uniform over non-masked pool positions (zero weight on padding positions)
        nan_cols = T.isnan().any(dim=1, keepdim=True)
        mask_f = pool_mask.float().unsqueeze(-1).expand_as(T)   # [B, P, L]
        counts = pool_mask.float().sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # [B, 1, 1]
        uniform = mask_f / counts
        T = torch.where(nan_cols, uniform, T)
        return T

    # ── main augmentation forward ─────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        lambda_ceiling: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute transformation matrix T and blend weight λ.

        Args:
            input_ids:       [B, L]  item ids (0 = padding)
            lambda_ceiling:  upper bound for λ (adapted per epoch)

        Returns:
            T:        [B, P, L]   transformation matrix (softmax over pool dim=1)
            pool_ids: [B, P]      integer item ids of the full pool
            lam:      [B, 1]      blend weights
        """
        B, L = input_ids.shape
        K = self.K

        # ── 1. Encode own sequence (using augmenter's own embeddings) ────────
        h, own_mask = self._encode(input_ids)  # [B, L, D], [B, L]

        # ── 2. Build pool_ids: own sequence + K sampled sequences ────────────
        sampled_ids = self._sample_inter_sequences(input_ids)  # [B, K*L]
        pool_ids = torch.cat([input_ids, sampled_ids], dim=1)  # [B, (1+K)*L]

        others_mask = (sampled_ids > 0).long()  # [B, K*L]
        pool_mask = torch.cat([own_mask, others_mask], dim=1)  # [B, (1+K)*L]

        # ── 3. Compute T logits and store for hard Gumbel ST in trainer ──────
        # T_head maps [B, L, D] → [B, L, P], then transpose to [B, P, L]
        T_logits = self.T_head(h).transpose(1, 2)  # [B, P, L]
        self._last_T_logits = T_logits  # stored for Phase 1 hard Gumbel ST

        # ── 4. Masked softmax over pool dim=1 ────────────────────────────────
        T = self._masked_softmax(T_logits, pool_mask)  # [B, P, L]

        # ── 5. Compute λ ─────────────────────────────────────────────────────
        seq_lengths = own_mask.sum(dim=1).float()  # [B]
        length_scale = (seq_lengths / self.max_seq_len).clamp(min=0.1)  # [B]

        h_pool = self._mean_pool(h, own_mask)  # [B, D]
        lam_base = torch.sigmoid(self.lambda_head(h_pool))  # [B, 1]
        lam = self.MIN_LAMBDA + self.LAMBDA_RANGE * lam_base * length_scale.unsqueeze(1)  # [B, 1]

        # Add Gaussian noise to prevent temporal collapse
        noise = torch.randn_like(lam) * self.LAMBDA_NOISE_SCALE
        lam = (lam + noise).clamp(self.MIN_LAMBDA, lambda_ceiling)  # [B, 1]

        return T, pool_ids, lam

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


class ASTARv2Augmenter(ASTARAugmenter):
    """Dual-view transport augmenter for the ASTARv2 ablation.

    Extends ASTARAugmenter with a second transformation head (T_head2) so
    that two independent augmented views can be generated from a single
    encoded representation.  The views are used in a view-to-view
    contrastive learning objective while entropy + usage-balance
    regularisation on the transport matrices prevents collapse.

    Key methods
    -----------
    generate_views(input_ids) → (aug_seq1, aug_seq2, reg_loss)
        Returns two discrete augmented sequences and a scalar regularisation
        term that carries gradient back to T_head / T_head2.
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
        super().__init__(
            num_items, hidden_size, max_seq_len, num_heads, num_layers,
            inner_size, dropout, K, tau_init, tau_floor, tau_decay,
        )
        pool_size = (1 + K) * max_seq_len
        self.T_head2 = nn.Linear(hidden_size, pool_size, bias=True)
        nn.init.normal_(self.T_head2.weight, std=0.02)
        nn.init.zeros_(self.T_head2.bias)

    # ── dual-view interface ───────────────────────────────────────────────────

    def generate_views(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate two augmented views and a transport regularisation loss.

        Args:
            input_ids: [B, L]  item ids (0 = padding)

        Returns:
            aug_seq1: [B, L]  first discrete augmented sequence
            aug_seq2: [B, L]  second discrete augmented sequence
            reg_loss: scalar  transport reg (entropy + usage-balance) with grad
        """
        h, own_mask = self._encode(input_ids)

        # Build pool: own sequence + K sampled sequences
        sampled_ids = self._sample_inter_sequences(input_ids)          # [B, K*L]
        pool_ids = torch.cat([input_ids, sampled_ids], dim=1)          # [B, P]
        others_mask = (sampled_ids > 0).long()
        pool_mask = torch.cat([own_mask, others_mask], dim=1)          # [B, P]

        # View 1: T_head (inherited)
        T1_logits = self.T_head(h).transpose(1, 2)                     # [B, P, L]
        T1 = self._masked_softmax(T1_logits, pool_mask)

        # View 2: T_head2
        T2_logits = self.T_head2(h).transpose(1, 2)                    # [B, P, L]
        T2 = self._masked_softmax(T2_logits, pool_mask)

        # Hard argmax → discrete sequences (no gradient through indices)
        aug_seq1 = self._argmax_select(T1, pool_ids, input_ids)        # [B, L]
        aug_seq2 = self._argmax_select(T2, pool_ids, input_ids)        # [B, L]

        # Regularisation: carries gradient back through T1 and T2
        reg_loss = self._transport_reg(T1, T2, pool_mask)

        return aug_seq1, aug_seq2, reg_loss

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _argmax_select(
        T: torch.Tensor,
        pool_ids: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Select items from pool via argmax over the pool dimension.

        Args:
            T:         [B, P, L]  transport matrix (softmax over dim=1)
            pool_ids:  [B, P]     item ids of the pool
            input_ids: [B, L]     original sequence (for padding mask)

        Returns:
            aug_seq: [B, L]  long tensor of selected item ids (0 at padding)
        """
        pool_idx = T.argmax(dim=1)                                      # [B, L]
        aug_seq = pool_ids.gather(1, pool_idx)                          # [B, L]
        pad_mask = (input_ids > 0)
        return torch.where(pad_mask, aug_seq, torch.zeros_like(aug_seq))

    @staticmethod
    def _transport_reg(
        T1: torch.Tensor,
        T2: torch.Tensor,
        pool_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Entropy + usage-balance regularisation on transport matrices.

        Maximising column-wise entropy prevents the augmenter from always
        selecting the same pool position for each output slot.
        Minimising usage-imbalance encourages all pool positions to be used.

        Both terms have gradient through T1 and T2.

        Args:
            T1, T2:    [B, P, L]  transport matrices (each column sums to 1)
            pool_mask: [B, P]     1 for real pool positions, 0 for padding

        Returns:
            scalar regularisation loss (lower is better)
        """
        eps = 1e-10

        # ── Column-wise entropy: H(T[:, :, j]) = -Σ_i T[i,j]*log(T[i,j]) ──
        ent1 = -(T1 * (T1 + eps).log()).sum(dim=1)  # [B, L]
        ent2 = -(T2 * (T2 + eps).log()).sum(dim=1)  # [B, L]
        # Negate: we want to maximise entropy, so minimise negative entropy
        entropy_loss = -(ent1.mean() + ent2.mean()) / 2.0

        # ── Usage balance: each pool position should be used uniformly ────────
        # Row-wise mean over output positions gives average usage per pool pos
        usage1 = T1.mean(dim=2)                                         # [B, P]
        usage2 = T2.mean(dim=2)                                         # [B, P]
        mask_f = pool_mask.float()                                       # [B, P]
        n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1)          # [B, 1]
        uniform = mask_f / n_valid                                       # [B, P]

        # KL(uniform ‖ usage): penalise under-used pool positions
        kl1 = (uniform * (uniform / (usage1 + eps)).log() * mask_f).sum(dim=1).mean()
        kl2 = (uniform * (uniform / (usage2 + eps)).log() * mask_f).sum(dim=1).mean()
        balance_loss = (kl1 + kl2) / 2.0

        return entropy_loss + balance_loss
