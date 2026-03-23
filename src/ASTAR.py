"""
augmenter.py - ASTAR augmenter

Core Formula:
    mixed = λ * (T @ V_pool) + (1-λ) * S

Where:
    S       [B, L, D]      original sequence embeddings (from recommender table)
    T       [B, P, L]      transformation matrix (softmax over pool dim=1)
    V_pool  [B, P, D]      pool value embeddings
    λ       [B, L, 1]      per-position blend weight
    mixed   [B, L, D]      augmented sequence embeddings → fed to recommender

Pool Structure:
    Position 0:         mask token (learnable parameter)  → masking operation
    Position 1..N:      N random items from vocabulary    → substitution operation
    Position N+1..N+L:  own sequence positions            → shuffling / identity

    Pool size P = 1 + N + L

T Computation (cross-attention):
    Q = Q_proj(h_own)          [B, L, D]    ← what does position j need?
    K = K_proj(V_pool)         [B, P, D]    ← what does pool position p offer?
    T = softmax(Q @ K.T / √D)  [B, L, P]   ← attention weights
    T = T.transpose(1,2)       [B, P, L]   ← rows=pool, cols=output

What T Learns:
    T[0, j] high:      mask position j (attend to mask token)
    T[1..N, j] high:   substitute position j with random item
    T[N+j, j] high:    keep original item at position j (identity)
    T[N+k, j] high:    shuffle: position j takes from own position k

λ Computation:
    h_pool = mean_pool(h_own, own_mask)    [B, D]
    lam    = sigmoid(lam_head(h_own))      [B, L, 1]  fully learned per position
    noise  = randn * 0.05                  training only, prevents temporal collapse

Masking Note:
    T output columns for padding positions are zeroed (own_mask applied)
    Final mixed is zeroed at padding positions
    Recommender attention mask uses original input_ids > 0

Training:
    Augmenter phase: soft T (differentiable, gradients flow to Q_proj, K_proj)
    Recommender phase: hard T via argmax (discrete, detached)
    λ-adjusted loss: loss weighted by mean(λ) — harder augmentation gets more weight

Architecture Components:
    item_embeddings     nn.Embedding(item_size, D)      own embeddings for encoding
    position_embeddings nn.Embedding(max_seq_len, D)    positional encoding
    item_encoder        Encoder(args)                   transformer encoder
    mask_embedding      nn.Parameter(D)                 learnable mask token
    Q_proj              nn.Linear(D, D)                 query projection
    K_proj              nn.Linear(D, D)                 key projection
    lam_head            nn.Linear(D, 1)                 lambda head

Hyperparameters:
    N:          number of random substitution candidates (default 20)
    tau:        softmax temperature (default 2.0, decays to 0.5)
    tau_decay:  temperature decay rate (default 0.99)
    min_tau:    minimum temperature floor (default 0.5)

Forward Returns:
    aug_emb  [B, L, D]   soft augmented embeddings (for augmenter phase)
    hard_ids [B, L]      discrete augmented item IDs (for recommender phase)
    lam      [B, L, 1]   blend weights (for λ-adjusted loss)
    T        [B, P, L]   transformation matrix (for visualization)
    own_mask [B, L]      padding mask

Soft vs Hard:
    Soft (augmenter phase):
        aug_emb = T_soft @ V_pool      ← weighted blend, differentiable
        mixed   = lam * aug_emb + (1-lam) * S

    Hard (recommender phase):
        selected = argmax(T, dim=1)    ← discrete pool index per output position
        hard_emb = V_pool[selected]    ← lookup selected embeddings
        mixed    = lam * hard_emb + (1-lam) * S
        → recommender sees clean discrete-like embeddings
        → lam detached, no gradient to augmenter

Visualization (post training):
    Average T across dataset → [P, L] heatmap
    Row 0:      how often positions get masked
    Rows 1..N:  how often positions get substituted
    Rows N+1..: how often positions get shuffled/kept

    Average λ across dataset → [L] vector
    Shows which positions get transformed most aggressively
    Expected: recent positions (high index) have lower λ
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm


class Augmenter(nn.Module):

    def __init__(
        self,
        args,
        N_rand: int = 20,   # random items from vocabulary
        N_sim:  int = 10,   # similar items (0 = disabled)
        N_hist: int = 10,   # user history items (0 = disabled)
    ):
        super().__init__()
        self.args        = args
        self.N_rand = N_rand
        self.N_sim  = N_sim
        self.N_hist = N_hist
        self.D           = args.hidden_size
        self.L           = args.max_seq_length
        self.num_items   = args.item_size

        # ── Own embeddings (for context encoding only) ───────────────────────
        self.item_embeddings     = nn.Embedding(args.item_size, self.D, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, self.D)
        self.item_encoder        = Encoder(args)
        self.LayerNorm           = LayerNorm(self.D, eps=1e-12)
        self.dropout             = nn.Dropout(args.hidden_dropout_prob)

        # ── Pool components ──────────────────────────────────────────────────
        # Learnable mask token embedding
        self.mask_embedding = nn.Parameter(torch.randn(self.D) * 0.02)

        # ── Cross-attention projections ──────────────────────────────────────
        self.Q_proj = nn.Linear(self.D, self.D)
        self.K_proj = nn.Linear(self.D, self.D)

        # ── Lambda head ──────────────────────────────────────────────────────
        self.lam_head = nn.Linear(self.D, 1)

        # ── Temperature ──────────────────────────────────────────────────────
        self.tau       = getattr(args, 'tau',       2.0)
        self.tau_decay = getattr(args, 'tau_decay', 0.99)
        self.min_tau   = getattr(args, 'min_tau',   0.5)

        self.apply(self.init_weights)

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids [B, L] → h [B, L, D], own_mask [B, L]
        """
        B, L         = input_ids.shape
        own_mask     = (input_ids > 0)                              # [B, L]
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)

        item_emb = self.item_embeddings(input_ids)                  # [B, L, D]
        pos_emb  = self.position_embeddings(position_ids)           # [1, L, D]
        emb      = self.LayerNorm(self.dropout(item_emb + pos_emb)) # [B, L, D]

        # Attention mask for transformer
        ext_mask = own_mask.unsqueeze(1).unsqueeze(2).float()       # [B, 1, 1, L]
        ext_mask = (1.0 - ext_mask) * -10000.0

        layers = self.item_encoder(emb, ext_mask, output_all_encoded_layers=True)
        h      = layers[-1]                                         # [B, L, D]
        return h, own_mask

    # ── Pool building ─────────────────────────────────────────────────────────

    def build_pool(
        self,
        input_ids: torch.Tensor,
        h_own: torch.Tensor,
        own_mask: torch.Tensor,
        item_similarity: Optional[torch.Tensor] = None,
        user_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build pool keys and values.

        Pool structure:
            Position 0:              mask token
            Position 1..N_rand:      random items from vocabulary
            Position N_rand+1..N_sim: similar items (if similarity matrix provided)
            Position N_sim+1..N_hist: user history items (if provided)
            Position N_hist+1..end:  own sequence positions (shuffling/identity)

        Args:
            input_ids:        [B, L]
            h_own:            [B, L, D]
            own_mask:         [B, L]
            item_similarity:  [num_items, top_k] precomputed top-k similar items
                            per item. None = disabled.
            user_history:     [B, H] padded user interaction history (all items
                            the user has interacted with, not just current window)
                            None = disabled.

        Returns:
            K_pool    [B, P, D]
            V_pool    [B, P, D]
            pool_mask [B, P]
        """
        B, L   = input_ids.shape
        D      = self.D
        device = input_ids.device

        K_parts    = []
        V_parts    = []
        mask_parts = []

        # ── 1. Mask token ────────────────────────────────────────────────────────
        mask_emb = self.mask_embedding.view(1, 1, D).expand(B, 1, D)   # [B, 1, D]
        K_parts.append(self.K_proj(mask_emb))
        V_parts.append(mask_emb)
        mask_parts.append(torch.ones(B, 1, device=device, dtype=torch.bool))

        # ── 2. Random items from vocabulary ──────────────────────────────────────
        if self.N_rand > 0:
            rand_ids  = torch.randint(1, self.num_items,
                                    (B, self.N_rand), device=device)  # [B, N_rand]
            E_rand    = self.item_embeddings(rand_ids)                   # [B, N_rand, D]
            K_parts.append(self.K_proj(E_rand))
            V_parts.append(E_rand)
            mask_parts.append(torch.ones(B, self.N_rand,
                                        device=device, dtype=torch.bool))

        # ── 3. Similar items via similarity matrix ────────────────────────────────
        # For each real item in input_ids, look up its top-k similar items
        # item_similarity: [num_items, top_k] precomputed, stored as tensor
        if item_similarity is not None and self.N_sim > 0:
            # Get top-k similar items for each position in input_ids
            # input_ids [B, L] → sim_ids [B, L, top_k]
            sim_ids = item_similarity[input_ids.clamp(min=0)]           # [B, L, top_k]

            # Sample N_sim items per sequence from the full set of similar candidates
            # Flatten L*top_k candidates, sample N_sim
            sim_ids_flat = sim_ids.view(B, -1)                          # [B, L*top_k]

            # Sample N_sim positions (with replacement if needed)
            sample_idx = torch.randint(0, sim_ids_flat.size(1),
                                    (B, self.N_sim), device=device)  # [B, N_sim]
            sampled_sim_ids = sim_ids_flat.gather(1, sample_idx)        # [B, N_sim]

            # Zero out similarities from padding positions
            # (padding items have id=0, their similar items are meaningless)
            has_real = (input_ids > 0).any(dim=1, keepdim=True)        # [B, 1]
            sampled_sim_ids = sampled_sim_ids * has_real.long()

            E_sim = self.item_embeddings(sampled_sim_ids)               # [B, N_sim, D]
            K_parts.append(self.K_proj(E_sim))
            V_parts.append(E_sim)
            # Mask: valid only if the sampled id is non-zero
            mask_parts.append(sampled_sim_ids > 0)                      # [B, N_sim]

        # ── 4. User history items ─────────────────────────────────────────────────
        # user_history: [B, H] padded, contains ALL items user has interacted with
        # This gives T access to the user's broader preference profile
        # beyond just the current window
        if user_history is not None and self.N_hist > 0:
            H = user_history.shape[1]

            if H > self.N_hist:
                # Sample N_hist items from history
                sample_idx  = torch.randint(0, H, (B, self.N_hist), device=device)
                hist_ids    = user_history.gather(1, sample_idx)        # [B, N_hist]
            else:
                hist_ids    = user_history                              # [B, H]

            E_hist = self.item_embeddings(hist_ids)                     # [B, N_hist, D]
            K_parts.append(self.K_proj(E_hist))
            V_parts.append(E_hist)
            mask_parts.append(hist_ids > 0)                             # [B, N_hist]

        # ── 5. Own sequence (shuffling / identity) ────────────────────────────────
        K_parts.append(self.K_proj(h_own))                              # [B, L, D]
        V_parts.append(h_own)                                           # [B, L, D]
        mask_parts.append(own_mask)                                     # [B, L]

        # ── Concatenate all pool parts ────────────────────────────────────────────
        K_pool    = torch.cat(K_parts,    dim=1)                        # [B, P, D]
        V_pool    = torch.cat(V_parts,    dim=1)                        # [B, P, D]
        pool_mask = torch.cat(mask_parts, dim=1)                        # [B, P]

        return K_pool, V_pool, pool_mask

    # ── T matrix (cross-attention) ────────────────────────────────────────────

    def compute_T(
        self,
        h_own: torch.Tensor,
        K_pool: torch.Tensor,
        pool_mask: torch.Tensor,
        own_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attention between own sequence and pool.

        Q = Q_proj(h_own)    [B, L, D]
        K = K_pool           [B, P, D]
        T = softmax(Q@K.T)   [B, L, P] → transpose → [B, P, L]

        Returns T [B, P, L]
        """
        B, L, D = h_own.shape
        P       = K_pool.shape[1]

        # Query from own sequence
        Q = self.Q_proj(h_own)                                      # [B, L, D]

        # Attention scores
        scores = torch.bmm(Q, K_pool.transpose(1, 2)) / (D ** 0.5) # [B, L, P]

        # Mask invalid pool positions
        scores = scores.masked_fill(
            ~pool_mask.unsqueeze(1).expand_as(scores), -1e9
        )

        # Softmax over pool dim (dim=-1 here, will transpose)
        T = F.softmax(scores / self.tau, dim=-1)                    # [B, L, P]
        T = T.transpose(1, 2)                                       # [B, P, L]

        # Zero out padding output positions
        T = T * own_mask.unsqueeze(1).float()                       # [B, P, L]

        return T

    # ── Lambda ────────────────────────────────────────────────────────────────

    def compute_lambda(
        self,
        h_own: torch.Tensor,
        own_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-position blend weight.
        Fully learned, no hand-crafted scale.

        Returns lam [B, L, 1]
        """
        lam = torch.sigmoid(self.lam_head(h_own))                   # [B, L, 1]

        if self.training:
            noise = torch.randn_like(lam) * 0.05
            lam   = (lam + noise).clamp(0.0, 1.0)

        # Zero padding positions
        lam = lam * own_mask.float().unsqueeze(-1)                  # [B, L, 1]
        return lam

    # ── Soft augmentation (augmenter phase) ──────────────────────────────────

    def soft_augment(
        self,
        T: torch.Tensor,
        V_pool: torch.Tensor,
        S: torch.Tensor,
        lam: torch.Tensor,
        own_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable augmentation for augmenter update phase.
        Gradients flow through T → Q_proj, K_proj.

        aug_emb = T.T @ V_pool    (soft weighted blend)
        mixed   = lam * aug_emb + (1-lam) * S

        Returns mixed [B, L, D]
        """
        # T [B, P, L], V_pool [B, P, D]
        aug_emb = torch.einsum('bpl,bpd->bld', T, V_pool)          # [B, L, D]
        mixed   = lam * aug_emb + (1 - lam) * S                    # [B, L, D]
        mixed   = mixed * own_mask.float().unsqueeze(-1)            # [B, L, D]
        return mixed

    # ── Hard augmentation (recommender phase) ─────────────────────────────────

    def hard_augment(
        self,
        T: torch.Tensor,
        V_pool: torch.Tensor,
        S: torch.Tensor,
        lam: torch.Tensor,
        own_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hard discrete-like augmentation for recommender update phase.
        Uses argmax over pool dim → selects single pool item per position.
        Detached from augmenter gradients.

        Returns mixed [B, L, D]
        """
        with torch.no_grad():
            # Argmax over pool dim → [B, L]
            selected = T.argmax(dim=1)                              # [B, L]

            # Gather selected embeddings from V_pool
            # selected [B, L] → expand to [B, L, D]
            idx      = selected.unsqueeze(-1).expand(-1, -1, V_pool.size(-1))
            hard_emb = V_pool.gather(1, idx)                       # [B, L, D]

            lam_det  = lam.detach()
            mixed    = lam_det * hard_emb + (1 - lam_det) * S     # [B, L, D]
            mixed    = mixed * own_mask.float().unsqueeze(-1)      # [B, L, D]

        return mixed

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        recommender_item_embeddings: nn.Embedding,
        item_similarity: Optional[torch.Tensor] = None,   # [num_items, top_k]
        user_history: Optional[torch.Tensor] = None,      # [B, H]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids:                    [B, L]
            recommender_item_embeddings:  recommender's embedding table
                                          used for S (original embeddings)
                                          ensures mixed is in recommender's space

        Returns:
            soft_mixed  [B, L, D]   soft augmentation (augmenter phase)
            hard_mixed  [B, L, D]   hard augmentation (recommender phase)
            lam         [B, L, 1]   blend weights
            T           [B, P, L]   transformation matrix
            own_mask    [B, L]      padding mask
        """
        # ── 1. Encode ────────────────────────────────────────────────────────
        h_own, own_mask = self.encode(input_ids)                    # [B, L, D], [B, L]

        # ── 2. Original embeddings from RECOMMENDER table ────────────────────
        # Critical: S must be in recommender's embedding space
        # so mixed embeddings are compatible with recommender's transformer
        S = recommender_item_embeddings(input_ids)                  # [B, L, D]

        # ── 3. Build pool ────────────────────────────────────────────────────
        K_pool, V_pool, pool_mask = self.build_pool(
            input_ids, h_own, own_mask,
            item_similarity=item_similarity,
            user_history=user_history,
        )

        # ── 4. Compute T ─────────────────────────────────────────────────────
        T = self.compute_T(h_own, K_pool, pool_mask, own_mask)     # [B, P, L]

        # ── 5. Compute λ ─────────────────────────────────────────────────────
        lam = self.compute_lambda(h_own, own_mask)                  # [B, L, 1]

        # ── 6. Soft augmentation (for augmenter update) ──────────────────────
        soft_mixed = self.soft_augment(T, V_pool, S, lam, own_mask)

        # ── 7. Hard augmentation (for recommender update) ────────────────────
        hard_mixed = self.hard_augment(T, V_pool, S, lam, own_mask)

        return soft_mixed, hard_mixed, lam, T, own_mask

    # ── Utilities ─────────────────────────────────────────────────────────────

    def decay_tau(self):
        self.tau = max(self.tau * self.tau_decay, self.min_tau)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
