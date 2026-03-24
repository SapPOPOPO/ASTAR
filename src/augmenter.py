import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm

class Augmenter(nn.Module):
    """
    Augmenter model generates two masked versions of input sequences

    Design Philosophy:
    1. Learn to generate meaningful augmentations that preserve original information
    2. Prevent mode collapse through structured randomness injection
    3. Enable controllable diversity via modulation

    Inputs:
        input_ids: [batch_size, seq_length] - input sequences of item IDs

    Outputs (during training):
        aug_seq1: [batch_size, seq_length] - first augmented sequence
        aug_seq2: [batch_size, seq_length] - second augmented sequence
        mask_probs1: [batch_size, seq_length] - mask probabilities for first head
        mask_probs2: [batch_size, seq_length] - mask probabilities for second head
        masks1: [batch_size, seq_length] - binary masks for first augmentation
        masks2: [batch_size, seq_length] - binary masks for second augmentation

    Outputs (during inference):
        aug_seq1, aug_seq2: deterministic augmentations (no randomness)

    Architecture Flow:
        1. Embedding + Position Encoding
        2. Transformer Encoder
        3. Randomness Injection (Modulation/Addition/Concatenation)
        4. Dual MLP Heads for Mask Logits
        5. Gumbel-Softmax Sampling
        6. Mask Application
    
    Versions:
        Randomness:
            - 1. Addition: seq_rep' = seq_rep + noise
            - 2. Modulaiton: seq_rep' = seq_rep * (1+f(noise))
            - 3. Concatenation: seq_rep' = concat(seq_rep, noise)
        Augmentation:
            - 1. Masking: mask items based on learned probabilities
            - 2. Modifying: mask and shift items based on learned probabilities

    Hyperparameters:
        - randomness_type: 'modulation' | 'addition' | 'concatenation'
        - augmentation_type: 'masking' | 'modifying'
    """

    def __init__(self, args, shared_item_embeddings=None, randomness_type='modulation', augmentation_type='modifying'):
        super().__init__()
        self.args = args

        self.randomness_type = randomness_type
        self.augmentation_type = augmentation_type
        self.noise_dim = getattr(args, 'noise_dim', 8)
        self.modulation_strength = getattr(args, 'modul_strengh', 0.01)

        # FIX 1: Share embeddings if provided, else create own
        if shared_item_embeddings is not None:
            self.item_embeddings = shared_item_embeddings
        else:
            self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        if self.randomness_type == 'modulation':
            self.noise_projection = nn.Sequential(
                nn.Linear(self.noise_dim, args.hidden_size),
                nn.Tanh()
            )
            mlp_input_dim = args.hidden_size
        elif self.randomness_type == 'concatenation':
            mlp_input_dim = args.hidden_size + self.noise_dim
        else:
            mlp_input_dim = args.hidden_size

        self.mlp_head1 = nn.Sequential(
            nn.Linear(mlp_input_dim, args.hidden_size),
            nn.GELU(),
            nn.Linear(args.hidden_size, 1)
        )
        self.mlp_head2 = nn.Sequential(
            nn.Linear(mlp_input_dim, args.hidden_size),
            nn.GELU(),
            nn.Linear(args.hidden_size, 1)
        )

        position_ids = torch.arange(args.max_seq_length, dtype=torch.long).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)
        
        attn_shape = (1, args.max_seq_length, args.max_seq_length)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long()
        self.register_buffer('subsequent_mask', subsequent_mask)

        self.tau = getattr(args, 'tau', 10.0)
        self.tau_decay = getattr(args, 'tau_decay', 0.99)
        self.min_tau = getattr(args, 'min_tau', 1.0)

        self.apply(self.init_weights)

        # FIX 1: Re-assign shared embeddings AFTER init_weights
        # because init_weights would have overwritten shared weights
        if shared_item_embeddings is not None:
            self.item_embeddings = shared_item_embeddings

    def add_position_embedding(self, sequence):
        '''
        Embed Sequence
        sequence: [batch_size, seq_length] -> sequence_emb: [batch_size, seq_length, hidden_size]            
        '''
        seq_length = sequence.size(1)
        
        # Use cached position_ids, sliced to current length
        position_ids = self.position_ids[:, :seq_length]
        
        item_embeddings = self.item_embeddings(sequence).detach()
        # item_embeddings = self.item_embeddings(sequence).detach()
        position_embeddings = self.position_embeddings(position_ids)
        
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def transformer_encoder(self, input_ids):
        '''
        input_ids: [batch_size, seq_length] -> sequence_output: [batch_size, seq_length, hidden_size]    
        '''
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, L]

        dtype = next(self.parameters()).dtype
        extended_attention_mask = extended_attention_mask.to(dtype=dtype) 
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True
        )

        sequence_output = item_encoded_layers[-1]
        return sequence_output
    
    def apply_randomness(self, seq_rep, noise_scale=None):
        if not self.training or self.randomness_type == 'none':
            return seq_rep
        
        B, L, D = seq_rep.shape
        noise_scale = noise_scale or self.modulation_strength

        # FIX 2: Adaptive scaling — noise scales with representation magnitude
        rep_norm = seq_rep.norm(dim=-1, keepdim=True).detach()   # [B, L, 1]
        mean_norm = rep_norm.mean().clamp(min=1e-8)              # scalar
        adaptive_scale = rep_norm / mean_norm                    # [B, L, 1]

        if self.randomness_type == 'addition':
            noise = torch.randn_like(seq_rep)
            return seq_rep + noise_scale * adaptive_scale * noise
            
        elif self.randomness_type == 'modulation':
            noise = torch.randn(B, L, self.noise_dim, device=seq_rep.device)
            modulation = self.noise_projection(noise)            # [B, L, D]
            return seq_rep * (1.0 + noise_scale * 1 * modulation)
            
        elif self.randomness_type == 'concatenation':
            noise = torch.randn(B, L, self.noise_dim, device=seq_rep.device)
            self.noise = noise
            return torch.cat([seq_rep, noise], dim=-1)

        return seq_rep
    
    def get_dual_mask_logits(self, input_ids, noise_scale=None):
        seq_out = self.transformer_encoder(input_ids)
        seq_out = self.apply_randomness(seq_out)

        h = F.gelu(seq_out)
        logit1 = self.mlp_head1(h).squeeze(-1)
        logit2 = self.mlp_head2(h).squeeze(-1)
        
        pad_mask = (input_ids > 0)
        logit1 = logit1 * pad_mask
        logit2 = logit2 * pad_mask
        
        return logit1, logit2

    def gumbel_softmax(self, logits, tau=1.0, hard=True):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = F.softmax(gumbels, dim=-1)

        if hard:
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft

    def compute_contrastive_regularization(self, probs1, probs2, pad_mask):
        """
        MCLRec's gamma*R term.
        Enforces a margin between positive and negative pair similarities.
        Penalizes:
            - positive pairs (same sequence) with too-low similarity
            - negative pairs (different sequences) with too-high similarity
        
        probs1, probs2: [B, L] mask probability outputs
        pad_mask:       [B, L] valid position mask
        """
        # Pool per-sequence mask vector (mean over valid positions)
        lengths = pad_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        z1 = (probs1 * pad_mask).sum(dim=1) / lengths.squeeze(1)  # [B]
        z2 = (probs2 * pad_mask).sum(dim=1) / lengths.squeeze(1)  # [B]

        # Similarity matrix: inner product between all pairs in batch
        # sigma_plus[i]  = sim(z1[i], z2[i])  — same sequence, positive pair
        # sigma_minus[i] = sim(z1[i], z2[j])  — different sequence, negative pair
        sim_matrix = torch.mm(z1.unsqueeze(1), z2.unsqueeze(0))   # [B, B]

        # Positive scores: diagonal (same sequence)
        sigma_plus = sim_matrix.diag()                             # [B]

        # Negative scores: off-diagonal (different sequences)
        mask_off_diag = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sigma_minus = sim_matrix[mask_off_diag]                    # [B*(B-1)]

        # Boundary values
        o_min = torch.min(torch.stack([sigma_plus.min(), sigma_minus.max()]))
        o_max = torch.max(torch.stack([sigma_plus.min(), sigma_minus.max()]))

        # Penalize positives that dip below o_min
        pos_penalty = torch.clamp(sigma_plus - o_min, min=0).mean()

        # Penalize negatives that rise above o_max
        neg_penalty = torch.clamp(o_max - sigma_minus, min=0).mean()

        return pos_penalty + neg_penalty
        
    # def sample_masks(self, input_ids, tau=None, hard=True, return_probs=True):
    #     pad_mask = input_ids > 0
    #     logit1, logit2 = self.get_dual_mask_logits(input_ids)
        
    #     full_logits1 = torch.stack([torch.zeros_like(logit1), logit1], dim=-1)
    #     samples1 = self.gumbel_softmax(full_logits1, tau=tau, hard=hard)
    #     masks1 = samples1[..., 1]
        
    #     full_logits2 = torch.stack([torch.zeros_like(logit2), logit2], dim=-1)
    #     samples2 = self.gumbel_softmax(full_logits2, tau=tau, hard=hard)
    #     masks2 = samples2[..., 1]

    #     if hard:
    #         masks1 = masks1.long()
    #         masks2 = masks2.long()
        
    #     masks1 = masks1 * pad_mask
    #     masks2 = masks2 * pad_mask

    #     if return_probs:
    #         probs1 = self.gumbel_softmax(full_logits1, tau=tau, hard=False)[..., 1] * pad_mask
    #         probs2 = self.gumbel_softmax(full_logits2, tau=tau, hard=False)[..., 1] * pad_mask
    #         return masks1, masks2, probs1, probs2, pad_mask
    #     else:
    #         return masks1, masks2, pad_mask

    def sample_masks(self, input_ids, tau=None, hard=True, return_probs=True, deterministic=False):
        # Fall back to the module's current tau if none is supplied
        tau = tau if tau is not None else self.tau
        pad_mask = input_ids > 0
        logit1, logit2 = self.get_dual_mask_logits(input_ids)
        
        full_logits1 = torch.stack([torch.zeros_like(logit1), logit1], dim=-1)
        full_logits2 = torch.stack([torch.zeros_like(logit2), logit2], dim=-1)

        if deterministic:
            # Deterministic soft masking: simple Softmax, no Gumbel noise
            # We treat the second dimension (index 1) as the "drop" probability
            probs1 = F.softmax(full_logits1 / (tau or 1.0), dim=-1)[..., 1] * pad_mask
            probs2 = F.softmax(full_logits2 / (tau or 1.0), dim=-1)[..., 1] * pad_mask
            masks1 = probs1
            masks2 = probs2
        else:
            # Original stochastic behavior (Bernoulli/Gumbel)
            samples1 = self.gumbel_softmax(full_logits1, tau=tau, hard=hard)
            masks1 = samples1[..., 1]
            
            samples2 = self.gumbel_softmax(full_logits2, tau=tau, hard=hard)
            masks2 = samples2[..., 1]

            if hard:
                masks1 = masks1.long()
                masks2 = masks2.long()
            
            masks1 = masks1 * pad_mask
            masks2 = masks2 * pad_mask

            if return_probs:
                probs1 = self.gumbel_softmax(full_logits1, tau=tau, hard=False)[..., 1] * pad_mask
                probs2 = self.gumbel_softmax(full_logits2, tau=tau, hard=False)[..., 1] * pad_mask

        if return_probs or deterministic:
            # For deterministic mode, masks are the probs
            p1 = masks1 if deterministic else probs1
            p2 = masks2 if deterministic else probs2
            return masks1, masks2, p1, p2, pad_mask
        else:
            return masks1, masks2, pad_mask

    def _vectorized_shift(self, input_ids, drop_mask):
        """for 'modifying' augmentaion"""
        B, L = input_ids.shape
        device = input_ids.device
        
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        sort_key = indices.float() - (drop_mask.float() * 1e9)
        
        sorted_indices = torch.argsort(sort_key, dim=1)
        shifted_ids = torch.gather(input_ids, 1, sorted_indices)
        
        num_dropped = drop_mask.sum(dim=1, keepdim=True)
        range_mat = torch.arange(L, device=device).unsqueeze(0)
        clean_mask = range_mat < num_dropped
        shifted_ids[clean_mask] = 0
        
        return shifted_ids

    def compute_entropy(self, probs1, probs2, pad_mask):
        eps = 1e-10
        entropy1 = - (probs1 * torch.log(probs1 + eps) + 
                     (1 - probs1) * torch.log(1 - probs1 + eps))
        entropy2 = - (probs2 * torch.log(probs2 + eps) + 
                     (1 - probs2) * torch.log(1 - probs2 + eps))
        
        entropy1 = entropy1 * pad_mask
        entropy2 = entropy2 * pad_mask
        
        valid_positions = pad_mask.sum(dim=1, keepdim=True).float() + eps
        mean_entropy1 = (entropy1.sum(dim=1) / valid_positions.squeeze(1)).mean()
        mean_entropy2 = (entropy2.sum(dim=1) / valid_positions.squeeze(1)).mean()
        
        return (mean_entropy1 + mean_entropy2) / 2
    
    def forward(self, input_ids, tau=None, soft_masking=False):
        # If soft_masking is True, we enforce deterministic behavior
        masks1, masks2, probs1, probs2, pad_mask = self.sample_masks(
            input_ids, tau=tau, hard=(not soft_masking), return_probs=True, deterministic=soft_masking
        )
        
        if soft_masking:
            # "Soft masking": directly multiply onto the original sequence
            # We assume mask contains "drop" probabilities, so we keep (1 - mask)
            # Result will be FloatTensor
            aug_seq1 = input_ids * (1.0 - masks1)
            aug_seq2 = input_ids * (1.0 - masks2)
        elif self.augmentation_type == 'modifying':
            aug_seq1 = self._vectorized_shift(input_ids, masks1)
            aug_seq2 = self._vectorized_shift(input_ids, masks2)
        else:  # 'masking'
            # Apply hard mask (replace dropped items with 0)
            aug_seq1 = input_ids.clone()
            aug_seq2 = input_ids.clone()
            aug_seq1[masks1.bool()] = 0
            aug_seq2[masks2.bool()] = 0

        return aug_seq1, aug_seq2, probs1, probs2, masks1, masks2, pad_mask

    # def _vectorized_shift(self, input_ids, drop_mask):
        # """for 'modifying' augmentaion"""
        # B, L = input_ids.shape
        # device = input_ids.device
        
        # indices = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        # sort_key = indices.float() - (drop_mask.float() * 1e9)
        
        # sorted_indices = torch.argsort(sort_key, dim=1)
        # shifted_ids = torch.gather(input_ids, 1, sorted_indices)
        
        # num_dropped = drop_mask.sum(dim=1, keepdim=True)
        # range_mat = torch.arange(L, device=device).unsqueeze(0)
        # clean_mask = range_mat < num_dropped
        # shifted_ids[clean_mask] = 0
        
        # return shifted_ids

    # def compute_entropy(self, probs1, probs2, pad_mask):
    #     eps = 1e-10
    #     entropy1 = - (probs1 * torch.log(probs1 + eps) + 
    #                  (1 - probs1) * torch.log(1 - probs1 + eps))
    #     entropy2 = - (probs2 * torch.log(probs2 + eps) + 
    #                  (1 - probs2) * torch.log(1 - probs2 + eps))
        
    #     entropy1 = entropy1 * pad_mask
    #     entropy2 = entropy2 * pad_mask
        
    #     valid_positions = pad_mask.sum(dim=1, keepdim=True).float() + eps
    #     mean_entropy1 = (entropy1.sum(dim=1) / valid_positions.squeeze(1)).mean()
    #     mean_entropy2 = (entropy2.sum(dim=1) / valid_positions.squeeze(1)).mean()
        
    #     return (mean_entropy1 + mean_entropy2) / 2
    
    # def forward(self, input_ids, tau=None):
    #     masks1, masks2, probs1, probs2, pad_mask = self.sample_masks(
    #         input_ids, tau=tau, hard=True, return_probs=True
    #     )
        
    #     if self.augmentation_type == 'modifying':
    #         aug_seq1 = self._vectorized_shift(input_ids, masks1)
    #         aug_seq2 = self._vectorized_shift(input_ids, masks2)
    #     else:  # 'masking'
    #         aug_seq1 = input_ids.clone()
    #         aug_seq2 = input_ids.clone()
    #         aug_seq1[masks1 == 1] = 0
    #         aug_seq2[masks2 == 1] = 0
        
    #     return aug_seq1, aug_seq2, probs1, probs2, masks1, masks2, pad_mask

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

    def modify_sequence(self, input_ids, tau=None):
        """Return (aug_seq1, aug_seq2, probs1, probs2) — compatibility alias for forward()."""
        aug_seq1, aug_seq2, probs1, probs2, _masks1, _masks2, _pad_mask = \
            self.forward(input_ids, tau=tau)
        return aug_seq1, aug_seq2, probs1, probs2