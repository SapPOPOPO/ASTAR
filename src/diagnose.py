import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class GANDiagnostics:

    def __init__(self, window_size=50):
        self.window_size  = window_size
        self.history      = defaultdict(list)
        self.probe_ids    = None   # fixed sequences for temporal collapse check
        self.probe_masks  = None   # their masks from previous epoch

    # ── Setup ────────────────────────────────────────────────────────────────

    def register_probe(self, input_ids):
        """
        Call ONCE before training with any fixed batch.
        These sequences will be reused every epoch for temporal collapse.
        """
        self.probe_ids   = input_ids.clone().cpu()
        self.probe_masks = None   # populated on first diagnostic call

    # ── Individual checks ────────────────────────────────────────────────────

    def check_temporal_collapse(self, augmenter, tau, threshold=0.95):
        """Same sequences → same masks across epochs?"""
        assert self.probe_ids is not None, "Call register_probe() before training."

        device = next(augmenter.parameters()).device
        ids    = self.probe_ids.to(device)

        with torch.no_grad():
            masks, _, _, _, pad_mask = augmenter.sample_masks(
                ids, tau=tau, hard=True, return_probs=True
            )
        masks = masks.cpu().float()

        if self.probe_masks is None:
            self.probe_masks = masks
            return {'diagnosis': 'Collecting first epoch'}

        # Cosine similarity per sequence between current and previous epoch
        curr = F.normalize(masks * pad_mask.cpu().float(), dim=-1)
        prev = F.normalize(self.probe_masks * pad_mask.cpu().float(), dim=-1)
        sims = (curr * prev).sum(dim=-1)   # [B]

        # Only count sequences that had actual masking in both epochs
        valid      = (masks.sum(dim=-1) > 0) & (self.probe_masks.sum(dim=-1) > 0)
        sims       = sims[valid]

        self.probe_masks = masks

        if len(sims) == 0:
            return {'diagnosis': 'No valid masked sequences to compare'}

        mean_sim      = sims.mean().item()
        pct_collapsed = (sims > threshold).float().mean().item()
        self.history['temporal_sim'].append(mean_sim)

        return {
            'mean_sim':       mean_sim,
            'pct_frozen':     pct_collapsed,
            'n_tracked':      len(sims),
            'diagnosis': (
                f'COLLAPSE: {pct_collapsed:.0%} sequences frozen (mean={mean_sim:.3f})'
                if mean_sim > threshold else
                f'OK (mean={mean_sim:.3f})'
            )
        }

    def check_player_dominance(self, loss_A, loss_B):
        """Is one player winning completely?"""
        self.history['loss_A'].append(loss_A)
        self.history['loss_B'].append(loss_B)

        if len(self.history['loss_A']) < self.window_size:
            return {'diagnosis': f'Collecting ({len(self.history["loss_A"])}/{self.window_size})'}

        A = np.array(self.history['loss_A'][-self.window_size:])
        B = np.array(self.history['loss_B'][-self.window_size:])
        x = np.arange(self.window_size)

        trend_A = np.polyfit(x, A, 1)[0]
        trend_B = np.polyfit(x, B, 1)[0]

        if   trend_A >  0.01 and trend_B < -0.01:
            diagnosis = 'RECOMMENDER DOMINANCE'
        elif trend_B >  0.01 and trend_A < -0.01:
            diagnosis = 'AUGMENTER DOMINANCE'
        elif abs(trend_A) < 0.001 and abs(trend_B) < 0.001:
            diagnosis = 'STAGNATION'
        else:
            diagnosis = 'OK'

        return {
            'trend_A':   trend_A,
            'trend_B':   trend_B,
            'diagnosis': diagnosis
        }

    def check_gradient_health(self, model, name):
        """Vanishing or exploding gradients?"""
        norms     = {n: p.grad.norm().item()
                     for n, p in model.named_parameters()
                     if p.grad is not None}

        if not norms:
            return {'diagnosis': f'{name}: no gradients found'}

        vanishing = [n for n, v in norms.items() if v < 1e-7]
        exploding = [n for n, v in norms.items() if v > 100.0]

        if len(vanishing) > len(norms) * 0.5:
            diagnosis = f'VANISHING: {len(vanishing)}/{len(norms)} layers near zero'
        elif exploding:
            diagnosis = f'EXPLODING: {len(exploding)} layers too large'
        else:
            diagnosis = 'OK'

        return {
            'mean_norm': np.mean(list(norms.values())),
            'max_norm':  max(norms.values()),
            'diagnosis': diagnosis
        }

    def check_oscillation(self, key):
        """Is loss bouncing without converging?"""
        if len(self.history[key]) < self.window_size:
            return {'diagnosis': f'Collecting ({len(self.history[key])}/{self.window_size})'}

        x        = np.array(self.history[key][-self.window_size:])
        autocorr = np.corrcoef(x[:-1], x[1:])[0, 1]
        changes  = np.mean(np.diff(np.sign(np.diff(x))) != 0)

        oscillating = autocorr < -0.3 and changes > 0.4

        return {
            'autocorr':   autocorr,
            'change_rate': changes,
            'diagnosis':  f'OSCILLATION in {key}' if oscillating else 'OK'
        }

    def check_per_sequence_quality(self, masks1, masks2,
                                    pad_mask, seq_lengths,
                                    min_rate=0.05, max_rate=0.95,
                                    max_asymmetry=0.4):
        """Per-sequence degenerate pairs — what batch averages hide."""
        rate1 = (masks1.float() * pad_mask).sum(1) / seq_lengths
        rate2 = (masks2.float() * pad_mask).sum(1) / seq_lengths
        diff  = (rate1 - rate2).abs()

        overlap = ((masks1 == masks2).float() * pad_mask).sum(1) / seq_lengths
        B       = masks1.size(0)

        pct = lambda t: (t.float().sum() / B).item()

        results = {
            'pct_under_masked_1':  pct(rate1 < min_rate),
            'pct_over_masked_1':   pct(rate1 > max_rate),
            'pct_under_masked_2':  pct(rate2 < min_rate),
            'pct_over_masked_2':   pct(rate2 > max_rate),
            'pct_asymmetric':      pct(diff > max_asymmetry),
            'pct_identical_views': pct(overlap > 0.95),
            'mean_rate1':          rate1.mean().item(),
            'mean_rate2':          rate2.mean().item(),
            'mean_asymmetry':      diff.mean().item(),
        }

        issues = [
            msg for cond, msg in [
                (results['pct_under_masked_1']  > 0.1, f"{results['pct_under_masked_1']:.0%} under-masked (v1)"),
                (results['pct_over_masked_1']   > 0.1, f"{results['pct_over_masked_1']:.0%} over-masked (v1)"),
                (results['pct_under_masked_2']  > 0.1, f"{results['pct_under_masked_2']:.0%} under-masked (v2)"),
                (results['pct_over_masked_2']   > 0.1, f"{results['pct_over_masked_2']:.0%} over-masked (v2)"),
                (results['pct_asymmetric']      > 0.1, f"{results['pct_asymmetric']:.0%} asymmetric pairs"),
                (results['pct_identical_views'] > 0.05, f"{results['pct_identical_views']:.0%} identical views"),
            ] if cond
        ]

        results['diagnosis'] = ' | '.join(issues) if issues else 'OK'
        return results

    def check_contrastive_signal(self, recommender, aug_seq1, aug_seq2):
        """Are the contrastive pairs actually informative?"""
        with torch.no_grad():
            z1 = F.normalize(recommender.transformer_encoder(aug_seq1)[:, -1, :], dim=-1)
            z2 = F.normalize(recommender.transformer_encoder(aug_seq2)[:, -1, :], dim=-1)

        sim   = torch.mm(z1, z2.T)
        B     = z1.size(0)
        pos   = sim.diag()
        off   = ~torch.eye(B, dtype=torch.bool, device=sim.device)
        neg   = sim[off].view(B, B - 1)

        mean_pos      = pos.mean().item()
        mean_neg      = neg.mean().item()
        margin        = mean_pos - mean_neg
        hard_positive = (pos < neg.max(1).values).float().mean().item()

        self.history['margin'].append(margin)
        self.history['mean_pos_sim'].append(mean_pos)

        issues = [
            msg for cond, msg in [
                (mean_pos > 0.95,  'TRIVIAL: views too similar'),
                (margin   < 0.05,  'WEAK MARGIN: pos/neg indistinguishable'),
            ] if cond
        ]
        if hard_positive > 0.3:
            issues.append(f'HARD PAIRS: {hard_positive:.0%} genuinely challenging ✓')

        return {
            'mean_pos':      mean_pos,
            'mean_neg':      mean_neg,
            'margin':        margin,
            'hard_positive': hard_positive,
            'diagnosis':     ' | '.join(issues) if issues else 'OK'
        }

    # ── Plot ─────────────────────────────────────────────────────────────────

    def plot(self, save_path=None):
        tracked = {k: v for k, v in self.history.items() if len(v) > 1}
        if not tracked:
            print("Nothing to plot yet.")
            return

        fig, axes = plt.subplots(len(tracked), 1,
                                  figsize=(12, 3 * len(tracked)))
        if len(tracked) == 1:
            axes = [axes]

        for ax, (key, vals) in zip(axes, tracked.items()):
            ax.plot(vals, linewidth=1.5)
            ax.set_title(key)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    # ── Master call ──────────────────────────────────────────────────────────

    def run(self, epoch,
            masks1, masks2, pad_mask, seq_lengths,
            loss_A, loss_B,
            augmenter, recommender,
            aug_seq1, aug_seq2):

        results = {
            'temporal_collapse':  self.check_temporal_collapse(augmenter, augmenter.tau),
            'player_dominance':   self.check_player_dominance(loss_A, loss_B),
            'grad_augmenter':     self.check_gradient_health(augmenter,   'augmenter'),
            'grad_recommender':   self.check_gradient_health(recommender, 'recommender'),
            'oscillation_A':      self.check_oscillation('loss_A'),
            'oscillation_B':      self.check_oscillation('loss_B'),
            'per_sequence':       self.check_per_sequence_quality(
                                      masks1, masks2, pad_mask, seq_lengths),
            'contrastive_signal': self.check_contrastive_signal(
                                      recommender, aug_seq1, aug_seq2),
        }

        print(f"\n{'='*55}")
        print(f"  DIAGNOSTIC — Epoch {epoch}")
        print(f"{'='*55}")
        for name, res in results.items():
            d = res.get('diagnosis', 'N/A')
            icon = '✓' if d in ('OK',) or d.startswith('Collect') else '✗'
            print(f"  {icon}  {name:<25} {d}")
        print(f"{'='*55}\n")

        return results