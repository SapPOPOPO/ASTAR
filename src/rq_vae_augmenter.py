import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm

class Augmenter(nn.Module):
    """
    VAE Augmenter: generates two jointly-masked views from a single latent strategy z.

    Architecture Flow:
        1. Embedding + Position Encoding
        2. Transformer Encoder  →  h [B, L, D]
        3. Pool h to h_pool [B, D]
        4. VAE Encoder: h_pool → (μ, log_σ)  [B, latent_dim]
        5. Reparameterize: z = μ + ε·exp(log_σ)
        6. VAE Decoder: cat(h_pool, z) → [B, 2*L] joint mask logits
           logit1 = logits[:, :L],  logit2 = logits[:, L:]
        7. Gumbel-Softmax Sampling → mask1, mask2  (jointly from one z)
        8. Mask Application

    Key property:
        Both masks come from the SAME z → views are coupled/complementary by design.
        At train time z is stochastic  → different every forward pass, no stagnation.
        At eval  time z = μ            → deterministic.

    Outputs (during training):
        aug_seq1, aug_seq2, probs1, probs2, masks1, masks2, pad_mask
        — identical signature to the previous dual-head Augmenter —

    Hyperparameters:
        - latent_dim:        size of VAE bottleneck   (default: 32)
        - augmentation_type: 'masking' | 'modifying'
    """

    def __init__(self, args, randomness_type='modulation', augmentation_type='modifying'):
        super().__init__()
        self.args = args

        # kept for API compat (trainers.py checks this for the old KL branch)
        self.randomness_type = 'vae'
        self.augmentation_type = augmentation_type

        self.latent_dim = getattr(args, 'latent_dim', 32)
        hidden_size    = args.hidden_size
        seq_len        = args.max_seq_length

        # ── Embedding & Encoder ──────────────────────────────────────────────
        self.item_embeddings    = nn.Embedding(args.item_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(seq_len, hidden_size)
        self.item_encoder       = Encoder(args)
        self.LayerNorm          = LayerNorm(hidden_size, eps=1e-12)
        self.dropout            = nn.Dropout(args.hidden_dropout_prob)

        # ── VAE Encoder heads  (h_pool → μ, log_σ) ──────────────────────────
        self.mu_head     = nn.Linear(hidden_size, self.latent_dim)
        self.logvar_head = nn.Linear(hidden_size, self.latent_dim)

        # ── VAE Decoder  (cat(h_pool, z) → 2*L logits) ──────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + self.latent_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2 * seq_len),   # joint: mask1 || mask2
        )

        # ── Buffers ──────────────────────────────────────────────────────────
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        attn_shape      = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long()
        self.register_buffer('subsequent_mask', subsequent_mask)

        # ── Gumbel temperature ───────────────────────────────────────────────
        self.tau       = getattr(args, 'tau', 10.0)
        self.tau_decay = getattr(args, 'tau_decay', 0.99)
        self.min_tau   = getattr(args, 'min_tau', 1.0)

        self.apply(self.init_weights)

    # ── Embedding helpers ────────────────────────────────────────────────────

    def add_position_embedding(self, sequence):
        seq_length          = sequence.size(1)
        position_ids        = self.position_ids[:, :seq_length]
        item_embeddings     = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb        = item_embeddings + position_embeddings
        sequence_emb        = self.LayerNorm(sequence_emb)
        sequence_emb        = self.dropout(sequence_emb)
        return sequence_emb

    def transformer_encoder(self, input_ids):
        attention_mask          = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        dtype                   = next(self.parameters()).dtype
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        sequence_emb            = self.add_position_embedding(input_ids)
        item_encoded_layers     = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        return item_encoded_layers[-1]   # [B, L, D]

    # ── VAE core ─────────────────────────────────────────────────────────────

    def encode(self, h_pool):
        """h_pool: [B, D]  →  (μ, log_σ) each [B, latent_dim]"""
        return self.mu_head(h_pool), self.logvar_head(h_pool)

    def reparameterize(self, mu, log_sigma):
        """Reparameterization trick; deterministic at eval."""
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(log_sigma)
        return mu

    @staticmethod
    def kl_loss(mu, log_sigma):
        """KL( q(z|x) || N(0,I) ),  mean over batch."""
        return -0.5 * torch.sum(
            1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp(), dim=-1
        ).mean()

    def get_dual_mask_logits(self, input_ids):
        """
        Returns (logit1, logit2, mu, log_sigma) — both logits jointly decoded
        from a single latent z, so the two masks are always coupled.
        """
        seq_out  = self.transformer_encoder(input_ids)   # [B, L, D]
        pad_mask = (input_ids > 0)                       # [B, L]

        # Pool: mean over non-padding positions
        lengths  = pad_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        h_pool   = (seq_out * pad_mask.unsqueeze(-1)).sum(dim=1) / lengths  # [B, D]

        # VAE encode
        mu, log_sigma = self.encode(h_pool)
        z             = self.reparameterize(mu, log_sigma)    # [B, latent_dim]

        # Joint decode → 2*L logits
        joint_logits = self.decoder(torch.cat([h_pool, z], dim=-1))  # [B, 2L]
        L            = input_ids.size(1)
        logit1, logit2 = joint_logits[:, :L], joint_logits[:, L:]    # [B, L] each

        # Zero-out padding
        logit1 = logit1 * pad_mask
        logit2 = logit2 * pad_mask

        return logit1, logit2, mu, log_sigma

    # ── Gumbel-Softmax ───────────────────────────────────────────────────────

    def gumbel_softmax(self, logits, tau=1.0, hard=True):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft  = F.softmax(gumbels, dim=-1)
        if hard:
            index  = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def sample_masks(self, input_ids, tau=None, hard=True, return_probs=True):
        tau      = tau or self.tau
        pad_mask = input_ids > 0
        logit1, logit2, mu, log_sigma = self.get_dual_mask_logits(input_ids)

        full_logits1 = torch.stack([torch.zeros_like(logit1), logit1], dim=-1)
        full_logits2 = torch.stack([torch.zeros_like(logit2), logit2], dim=-1)

        samples1 = self.gumbel_softmax(full_logits1, tau=tau, hard=hard)
        samples2 = self.gumbel_softmax(full_logits2, tau=tau, hard=hard)

        masks1 = samples1[..., 1]
        masks2 = samples2[..., 1]

        if hard:
            masks1 = masks1.long()
            masks2 = masks2.long()

        masks1 = masks1 * pad_mask
        masks2 = masks2 * pad_mask

        if return_probs:
            probs1 = self.gumbel_softmax(full_logits1, tau=tau, hard=False)[..., 1] * pad_mask
            probs2 = self.gumbel_softmax(full_logits2, tau=tau, hard=False)[..., 1] * pad_mask
            return masks1, masks2, probs1, probs2, pad_mask, mu, log_sigma
        else:
            return masks1, masks2, pad_mask

    # ── Sequence modification helpers ────────────────────────────────────────

    def _vectorized_shift(self, input_ids, drop_mask):
        """for 'modifying' augmentation"""
        B, L   = input_ids.shape
        device = input_ids.device
        indices    = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        sort_key   = indices.float() - (drop_mask.float() * 1e9)
        sorted_idx = torch.argsort(sort_key, dim=1)
        shifted    = torch.gather(input_ids, 1, sorted_idx)
        num_dropped = drop_mask.sum(dim=1, keepdim=True)
        range_mat   = torch.arange(L, device=device).unsqueeze(0)
        shifted[range_mat < num_dropped] = 0
        return shifted

    # ── Entropy (unchanged) ──────────────────────────────────────────────────

    def compute_entropy(self, probs1, probs2, pad_mask):
        eps      = 1e-10
        entropy1 = -(probs1 * torch.log(probs1 + eps) +
                     (1 - probs1) * torch.log(1 - probs1 + eps)) * pad_mask
        entropy2 = -(probs2 * torch.log(probs2 + eps) +
                     (1 - probs2) * torch.log(1 - probs2 + eps)) * pad_mask
        valid    = pad_mask.sum(dim=1, keepdim=True).float() + eps
        e1       = (entropy1.sum(dim=1) / valid.squeeze(1)).mean()
        e2       = (entropy2.sum(dim=1) / valid.squeeze(1)).mean()
        return (e1 + e2) / 2

    # ── Forward (same 7-tuple API as before) ─────────────────────────────────

    def forward(self, input_ids, tau=None):
        masks1, masks2, probs1, probs2, pad_mask, mu, log_sigma = self.sample_masks(
            input_ids, tau=tau, hard=True, return_probs=True
        )

        # store for trainer to read
        self.last_mu        = mu
        self.last_log_sigma = log_sigma

        if self.augmentation_type == 'modifying':
            aug_seq1 = self._vectorized_shift(input_ids, masks1)
            aug_seq2 = self._vectorized_shift(input_ids, masks2)
        else:  # 'masking'
            aug_seq1            = input_ids.clone()
            aug_seq2            = input_ids.clone()
            aug_seq1[masks1 == 1] = 0
            aug_seq2[masks2 == 1] = 0

        return aug_seq1, aug_seq2, probs1, probs2, masks1, masks2, pad_mask

    # ── Utilities ────────────────────────────────────────────────────────────

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def decay_tau(self):
        self.tau = max(self.tau * self.tau_decay, self.min_tau)