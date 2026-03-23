"""
analyze_T.py — Post-training analysis of ASTAR's T matrix.

Graphs:
    1. Operation distribution over training epochs (time graph)
       Shows how masking/substitution/shuffle/identity evolves during training

    2. Per-position operation distribution
       Shows which operations dominate at each sequence position

    3. Average λ per position
       Shows blend intensity across sequence positions

    4. λ and operation distribution vs sequence length
       Shows how augmentation strategy adapts to short vs long sequences

    5. T matrix heatmap
       Average T [P, L] across dataset — visualizes routing patterns

Usage:
    # During training, call analyzer.record(T, lam, own_mask, pool_config, epoch)
    # After training, call analyzer.plot(save_dir)

    analyzer = TMatrixAnalyzer(args, N_rand, N_sim)
    # in trainer, after augmenter forward:
    analyzer.record(T, lam, own_mask, epoch)
    # after training:
    analyzer.plot(save_dir='plots/')
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from collections import defaultdict


class TMatrixAnalyzer:
    """
    Records and analyzes T matrix patterns during/after training.

    Pool structure assumed:
        Position 0:              mask token
        Position 1..N_rand:      random substitution candidates
        Position N_rand+1..N_sim: similar item substitution candidates
        Position N_sim+1..end:   own sequence positions (identity/shuffle)

    Args:
        args:   training args (for max_seq_length)
        N_rand: number of random pool positions
        N_sim:  number of similarity-based pool positions
    """

    def __init__(self, args, N_rand=20, N_sim=0):
        self.L      = args.max_seq_length
        self.N_rand = N_rand
        self.N_sim  = N_sim

        # Pool boundaries
        self.mask_idx   = 0
        self.rand_start = 1
        self.rand_end   = 1 + N_rand
        self.sim_start  = 1 + N_rand
        self.sim_end    = 1 + N_rand + N_sim
        self.own_start  = 1 + N_rand + N_sim   # own sequence starts here

        # Storage: per epoch
        # Each entry: dict of operation → [B, L] tensors accumulated
        self.epoch_records  = defaultdict(lambda: defaultdict(list))
        self.lam_records    = defaultdict(list)   # epoch → list of [B, L] lam tensors
        self.length_records = defaultdict(list)   # epoch → list of seq_lengths [B]
        self.T_records      = defaultdict(list)   # epoch → list of T [B, P, L] tensors
        self.recorded_epochs = []

    # ── Recording ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def record(self, T, lam, own_mask, epoch):
        """
        Record one batch of T matrices.

        Args:
            T:        [B, P, L]  transformation matrix (soft, after softmax)
            lam:      [B, L, 1]  blend weights
            own_mask: [B, L]     padding mask
            epoch:    int        current training epoch
        """
        B, P, L = T.shape
        T        = T.cpu().float()
        lam      = lam.cpu().float().squeeze(-1)   # [B, L]
        own_mask = own_mask.cpu().float()           # [B, L]

        # ── Classify operations per output position ───────────────────────────
        # For each output position j, T[:, p, j] = weight on pool position p
        # Classify by which pool region has the highest total weight

        # Mask weight: T[:, 0, :]
        w_mask = T[:, self.mask_idx, :]                              # [B, L]

        # Random substitution weight: sum over random pool positions
        w_rsub = T[:, self.rand_start:self.rand_end, :].sum(dim=1)  # [B, L]

        # Semantic substitution weight: sum over sim pool positions
        if self.N_sim > 0:
            w_ssub = T[:, self.sim_start:self.sim_end, :].sum(dim=1) # [B, L]
        else:
            w_ssub = torch.zeros_like(w_mask)

        # Own sequence weights: T[:, own_start:, :]
        w_own = T[:, self.own_start:, :]                             # [B, L, L]

        # Identity: weight on own position j (diagonal)
        diag_idx = torch.arange(L)
        w_identity = w_own[:, diag_idx, diag_idx]                   # [B, L]

        # Shuffle: weight on own positions other than j
        w_shuffle = w_own.sum(dim=1) - w_identity                   # [B, L]

        # Store weighted averages (masked to real positions only)
        for name, w in [
            ('mask',     w_mask),
            ('r_sub',    w_rsub),
            ('s_sub',    w_ssub),
            ('identity', w_identity),
            ('shuffle',  w_shuffle),
        ]:
            # Zero out padding positions
            masked = (w * own_mask).sum(0) / own_mask.sum(0).clamp(min=1)  # [L]
            self.epoch_records[epoch][name].append(masked.numpy())

        # Store λ (masked)
        lam_masked = (lam * own_mask).sum(0) / own_mask.sum(0).clamp(min=1)
        self.lam_records[epoch].append(lam_masked.numpy())

        # Store sequence lengths
        seq_lengths = own_mask.sum(dim=1).long()                     # [B]
        self.length_records[epoch].append(seq_lengths.numpy())

        # Store T sample (one per batch, for heatmap — don't store all)
        if len(self.T_records[epoch]) < 50:
            self.T_records[epoch].append(T.mean(0).numpy())          # [P, L]

        if epoch not in self.recorded_epochs:
            self.recorded_epochs.append(epoch)

    def _aggregate_epoch(self, epoch):
        """Aggregate all batches for an epoch into mean per-position values."""
        ops = {}
        for name, batches in self.epoch_records[epoch].items():
            ops[name] = np.stack(batches, axis=0).mean(axis=0)       # [L]
        lam = np.stack(self.lam_records[epoch], axis=0).mean(axis=0) # [L]
        return ops, lam

    def _aggregate_by_length(self, epoch):
        """
        Group sequences by length bucket and compute mean operation weights.
        Returns dict: length_bucket → {op_name: mean_weight}
        """
        # Collect per-sequence operation weights and lengths
        # Re-run over stored data — we stored per-position averages, not per-sequence
        # For length analysis we need per-sequence data
        # Instead: approximate using sequence length distribution
        lengths = np.concatenate(self.length_records[epoch])         # [N]
        buckets = {}
        for name, batches in self.epoch_records[epoch].items():
            stacked = np.stack(batches, axis=0)                      # [n_batches, L]
            buckets[name] = stacked.mean(0)
        return lengths, buckets

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(self, save_dir='plots/', dataset_name='dataset'):
        """
        Generate all analysis plots.

        Args:
            save_dir:     directory to save plots
            dataset_name: name for plot titles and filenames
        """
        os.makedirs(save_dir, exist_ok=True)

        epochs = sorted(self.recorded_epochs)
        if len(epochs) == 0:
            print("[TMatrixAnalyzer] No data recorded.")
            return

        colors = {
            'mask':     '#e74c3c',
            'r_sub':    '#3498db',
            's_sub':    '#2ecc71',
            'identity': '#95a5a6',
            'shuffle':  '#f39c12',
        }
        labels = {
            'mask':     'Mask',
            'r_sub':    'Random Sub',
            's_sub':    'Semantic Sub',
            'identity': 'Identity',
            'shuffle':  'Shuffle',
        }

        # ── Plot 1: Operation distribution over epochs (time graph) ───────────
        self._plot_time_graph(epochs, colors, labels, save_dir, dataset_name)

        # ── Plot 2: Per-position operation distribution ───────────────────────
        self._plot_position_ops(epochs, colors, labels, save_dir, dataset_name)

        # ── Plot 3: Average λ per position ────────────────────────────────────
        self._plot_lambda_position(epochs, save_dir, dataset_name)

        # ── Plot 4: Operations vs sequence length ─────────────────────────────
        self._plot_ops_vs_length(epochs, colors, labels, save_dir, dataset_name)

        # ── Plot 5: T matrix heatmap ──────────────────────────────────────────
        self._plot_T_heatmap(epochs, save_dir, dataset_name)

        print(f"[TMatrixAnalyzer] Plots saved to {save_dir}")

    def _plot_time_graph(self, epochs, colors, labels, save_dir, dataset_name):
        """Operation distribution over training epochs."""
        fig, ax = plt.subplots(figsize=(12, 5))

        op_names = ['mask', 'r_sub', 's_sub', 'identity', 'shuffle']
        op_means = {name: [] for name in op_names}

        for epoch in epochs:
            ops, _ = self._aggregate_epoch(epoch)
            for name in op_names:
                # Mean across all real positions
                op_means[name].append(ops[name].mean())

        # Normalize to sum to 1 per epoch
        totals = np.array([
            sum(op_means[name][i] for name in op_names)
            for i in range(len(epochs))
        ])

        bottom = np.zeros(len(epochs))
        for name in op_names:
            vals = np.array(op_means[name]) / totals.clip(min=1e-8)
            ax.fill_between(epochs, bottom, bottom + vals,
                            alpha=0.8, color=colors[name], label=labels[name])
            bottom += vals

        ax.set_xlabel('Epoch',    fontsize=12)
        ax.set_ylabel('Operation Proportion', fontsize=12)
        ax.set_title(f'{dataset_name} — Operation Distribution Over Training', fontsize=13)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(epochs[0], epochs[-1])
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_ops_time.png'), dpi=150)
        plt.close()

    def _plot_position_ops(self, epochs, colors, labels, save_dir, dataset_name):
        """Per-position operation distribution (last epoch)."""
        fig, ax = plt.subplots(figsize=(12, 5))

        # Use last epoch
        last_epoch = epochs[-1]
        ops, _     = self._aggregate_epoch(last_epoch)
        op_names   = ['mask', 'r_sub', 's_sub', 'identity', 'shuffle']

        positions = np.arange(self.L)

        # Normalize per position
        total = sum(ops[name] for name in op_names)
        total = np.clip(total, a_min=1e-8, a_max=None)

        bottom = np.zeros(self.L)
        for name in op_names:
            vals = ops[name] / total
            ax.bar(positions, vals, bottom=bottom,
                   color=colors[name], label=labels[name],
                   width=1.0, alpha=0.85)
            bottom += vals

        ax.set_xlabel('Sequence Position (0=oldest, L-1=newest)', fontsize=12)
        ax.set_ylabel('Operation Proportion', fontsize=12)
        ax.set_title(f'{dataset_name} — Per-Position Operation Distribution (Epoch {last_epoch})',
                     fontsize=13)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(-0.5, self.L - 0.5)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_ops_position.png'), dpi=150)
        plt.close()

    def _plot_lambda_position(self, epochs, save_dir, dataset_name):
        """Average λ per position across training epochs."""
        fig, ax = plt.subplots(figsize=(12, 5))

        positions  = np.arange(self.L)
        n_epochs   = min(5, len(epochs))
        # Sample epochs evenly
        sampled    = [epochs[int(i * (len(epochs)-1) / max(n_epochs-1, 1))]
                      for i in range(n_epochs)]
        cmap       = plt.cm.viridis

        for idx, epoch in enumerate(sampled):
            _, lam = self._aggregate_epoch(epoch)
            color  = cmap(idx / max(n_epochs - 1, 1))
            ax.plot(positions, lam, color=color, linewidth=2,
                    label=f'Epoch {epoch}', alpha=0.85)

        ax.set_xlabel('Sequence Position (0=oldest, L-1=newest)', fontsize=12)
        ax.set_ylabel('Mean λ', fontsize=12)
        ax.set_title(f'{dataset_name} — Average λ Per Position', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim(0, self.L - 1)
        ax.set_ylim(0, 1)
        ax.axvline(x=self.L * 0.7, color='red', linestyle='--',
                   alpha=0.4, label='Recent region')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_lambda_position.png'), dpi=150)
        plt.close()

    def _plot_ops_vs_length(self, epochs, colors, labels, save_dir, dataset_name):
        """
        Operation distribution and mean λ vs sequence length.
        Buckets sequences by length and shows how strategy changes.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        last_epoch = epochs[-1]
        lengths    = np.concatenate(self.length_records[last_epoch])  # [N]

        # Define length buckets
        buckets    = [(1, 5), (6, 10), (11, 20), (21, 35), (36, 50)]
        bucket_labels = ['1-5', '6-10', '11-20', '21-35', '36-50']
        op_names   = ['mask', 'r_sub', 's_sub', 'identity', 'shuffle']

        # For each bucket, compute mean operation weights
        # Use per-position averages weighted by sequence length
        # Approximate: shorter sequences use earlier positions less
        bucket_ops = {name: [] for name in op_names}
        bucket_lam = []
        bucket_counts = []

        for lo, hi in buckets:
            mask = (lengths >= lo) & (lengths <= hi)
            count = mask.sum()
            bucket_counts.append(count)

            if count == 0:
                for name in op_names:
                    bucket_ops[name].append(0.0)
                bucket_lam.append(0.0)
                continue

            # Use only positions 0..mean_length for this bucket
            mean_len = int(lengths[mask].mean())
            start_pos = self.L - mean_len  # left-padded sequences

            ops, lam = self._aggregate_epoch(last_epoch)
            for name in op_names:
                bucket_ops[name].append(ops[name][start_pos:].mean())
            bucket_lam.append(lam[start_pos:].mean())

        # ── Subplot 1: Stacked bar of operations per bucket ───────────────────
        ax   = axes[0]
        x    = np.arange(len(buckets))
        totals = np.array([
            sum(bucket_ops[name][i] for name in op_names)
            for i in range(len(buckets))
        ]).clip(min=1e-8)

        bottom = np.zeros(len(buckets))
        for name in op_names:
            vals = np.array(bucket_ops[name]) / totals
            ax.bar(x, vals, bottom=bottom,
                   color=colors[name], label=labels[name], alpha=0.85)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([f'{l}\n(n={c})' for l, c in
                            zip(bucket_labels, bucket_counts)], fontsize=9)
        ax.set_ylabel('Operation Proportion', fontsize=12)
        ax.set_title(f'{dataset_name} — Operations vs Sequence Length', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        # ── Subplot 2: Mean λ per bucket ──────────────────────────────────────
        ax2 = axes[1]
        bars = ax2.bar(x, bucket_lam, color='#8e44ad', alpha=0.8, width=0.6)

        # Annotate bars
        for bar, val in zip(bars, bucket_lam):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax2.set_xticks(x)
        ax2.set_xticklabels(bucket_labels, fontsize=10)
        ax2.set_ylabel('Mean λ', fontsize=12)
        ax2.set_title(f'{dataset_name} — Mean λ vs Sequence Length', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_ops_vs_length.png'), dpi=150)
        plt.close()

    def _plot_T_heatmap(self, epochs, save_dir, dataset_name):
        """Average T matrix heatmap [P, L]."""
        last_epoch = epochs[-1]

        if not self.T_records[last_epoch]:
            return

        T_avg = np.stack(self.T_records[last_epoch], axis=0).mean(0)  # [P, L]
        P, L  = T_avg.shape

        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(T_avg, aspect='auto', cmap='YlOrRd',
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Mean Attention Weight')

        # Draw pool region boundaries
        ax.axhline(y=0.5,              color='blue',  linewidth=1.5,
                   linestyle='--', label='Mask token')
        ax.axhline(y=self.rand_end - 0.5, color='green', linewidth=1.5,
                   linestyle='--', label=f'Random pool end (pos {self.rand_end})')
        if self.N_sim > 0:
            ax.axhline(y=self.sim_end - 0.5, color='orange', linewidth=1.5,
                       linestyle='--', label=f'Sim pool end (pos {self.sim_end})')

        # Y-axis labels for pool regions
        ytick_pos = [0,
                     (self.rand_start + self.rand_end) // 2,
                     max((self.sim_start + self.sim_end) // 2, self.rand_end + 1),
                     self.own_start + L // 2]
        ytick_labels = ['Mask', 'Random\nSub', 'Semantic\nSub', 'Own Seq']
        if self.N_sim == 0:
            ytick_pos   = [0, (self.rand_start + self.rand_end) // 2, self.own_start + L // 2]
            ytick_labels = ['Mask', 'Random\nSub', 'Own Seq']

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels, fontsize=10)
        ax.set_xlabel('Output Position (0=oldest, L-1=newest)', fontsize=12)
        ax.set_ylabel('Pool Position (Source)',                  fontsize=12)
        ax.set_title(f'{dataset_name} — Average T Matrix Heatmap (Epoch {last_epoch})',
                     fontsize=13)
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_T_heatmap.png'), dpi=150)
        plt.close()


# ── Summary stats (printed to log) ───────────────────────────────────────────

    def print_summary(self, dataset_name='dataset'):
        """Print a text summary of the final epoch's T matrix patterns."""
        if not self.recorded_epochs:
            return

        last_epoch = sorted(self.recorded_epochs)[-1]
        ops, lam   = self._aggregate_epoch(last_epoch)
        op_names   = ['mask', 'r_sub', 's_sub', 'identity', 'shuffle']

        total = sum(ops[name].mean() for name in op_names)
        total = max(total, 1e-8)

        print(f"\n[TMatrixAnalyzer] {dataset_name} — Epoch {last_epoch} Summary")
        print(f"  {'Operation':<15} {'Mean Weight':>12}  {'Proportion':>12}")
        print(f"  {'-'*42}")
        for name in op_names:
            mean_w = ops[name].mean()
            print(f"  {name:<15} {mean_w:>12.4f}  {mean_w/total:>11.1%}")
        print(f"\n  Mean λ (all positions): {lam.mean():.4f}")
        print(f"  Mean λ (recent half):   {lam[self.L//2:].mean():.4f}")
        print(f"  Mean λ (early half):    {lam[:self.L//2].mean():.4f}")