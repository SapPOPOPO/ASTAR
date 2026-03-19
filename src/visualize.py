"""
visualize.py — Interpretability visualisations for ASTAR.

Includes:
    - T heatmap (intra + inter attention)
    - λ distribution and length correlation
    - Intra vs inter attention per dataset
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _require_matplotlib() -> None:
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualisations. pip install matplotlib")


# ─────────────────────────────────────────────────────────────────────────────
# T matrix visualisation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def visualize_T(
    augmenter,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str = "plots/T_heatmap.png",
    max_batches: int = 50,
    lambda_ceiling: float = 0.8,
) -> None:
    """Average T matrix across the dataset and plot as heatmap.

    Saves:
        {save_path}:   T_intra [L, L] heatmap  (own-sequence attention)
        Alongside:     T_inter [L, K*L] mean per position  (cross-sequence)
    """
    _require_matplotlib()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    augmenter.eval()
    L = augmenter.max_seq_len
    K = augmenter.K
    pool_size = (1 + K) * L

    T_accum = torch.zeros(L, pool_size)
    count = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        _, T, _ = augmenter(input_ids, lambda_ceiling=lambda_ceiling)
        T_accum += T.cpu().mean(0)  # average over batch
        count += 1

    T_avg = T_accum / max(count, 1)  # [L, (1+K)*L]

    T_intra = T_avg[:, :L].numpy()  # [L, L]
    T_inter = T_avg[:, L:].numpy()  # [L, K*L]
    inter_per_pos = T_inter.mean(axis=1)  # [L]  mean weight to inter-sequences
    intra_per_pos = T_intra.sum(axis=1)   # [L]  total weight on own sequence

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[3, 1, 1])

    # ── T_intra heatmap ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(T_intra, aspect="auto", cmap="hot", vmin=0)
    ax0.set_title("T_intra: Own-sequence attention")
    ax0.set_xlabel("Source position j")
    ax0.set_ylabel("Target position i")
    plt.colorbar(im, ax=ax0)

    # ── Intra vs inter per position ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    positions = np.arange(L)
    ax1.barh(positions, intra_per_pos, color="steelblue", label="intra")
    ax1.barh(positions, inter_per_pos, left=intra_per_pos, color="coral", label="inter")
    ax1.set_title("Intra vs Inter\nweight per position")
    ax1.set_xlabel("Attention weight")
    ax1.set_ylabel("Position")
    ax1.legend(fontsize=8)
    ax1.invert_yaxis()

    # ── Summary bar ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    mean_intra = float(intra_per_pos.mean())
    mean_inter = float(inter_per_pos.mean())
    ax2.bar(["Intra", "Inter"], [mean_intra, mean_inter], color=["steelblue", "coral"])
    ax2.set_title("Mean attention\nIntra vs Inter")
    ax2.set_ylabel("Mean weight")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize_T] Saved → {save_path}")
    print(f"  Mean intra weight: {mean_intra:.4f}")
    print(f"  Mean inter weight: {mean_inter:.4f}")
    if mean_inter > mean_intra:
        print("  → Cross-user mixing dominant (expect: sparse dataset like Beauty)")
    else:
        print("  → Intra-sequence ops dominant (expect: dense dataset like Yelp)")


# ─────────────────────────────────────────────────────────────────────────────
# λ distribution visualisation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def visualize_lambda(
    augmenter,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str = "plots/lambda_dist.png",
    max_batches: int = 50,
    lambda_history: Optional[list] = None,
    lambda_ceiling: float = 0.8,
) -> None:
    """Plot λ distribution and its correlation with sequence length.

    Args:
        lambda_history: list of (epoch, mean_lambda) tuples for trajectory plot
    """
    _require_matplotlib()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    augmenter.eval()
    lam_list = []
    len_list = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        seq_lens = (input_ids > 0).sum(dim=1).float().cpu().numpy()

        _, _, lam = augmenter(input_ids, lambda_ceiling=lambda_ceiling)
        lam_list.extend(lam.cpu().view(-1).numpy().tolist())
        len_list.extend(seq_lens.tolist())

    lam_arr = np.array(lam_list)
    len_arr = np.array(len_list)

    ncols = 3 if lambda_history else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    # ── λ distribution ────────────────────────────────────────────────────
    axes[0].hist(lam_arr, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].set_title("λ Distribution")
    axes[0].set_xlabel("λ value")
    axes[0].set_ylabel("Count")
    axes[0].axvline(lam_arr.mean(), color="red", linestyle="--", label=f"mean={lam_arr.mean():.3f}")
    axes[0].legend()

    # ── λ vs sequence length ──────────────────────────────────────────────
    axes[1].scatter(len_arr, lam_arr, alpha=0.3, s=5, color="steelblue")
    # Trend line
    if len(len_arr) > 1:
        z = np.polyfit(len_arr, lam_arr, 1)
        p = np.poly1d(z)
        xs = np.linspace(len_arr.min(), len_arr.max(), 100)
        axes[1].plot(xs, p(xs), color="red", linewidth=2, label=f"slope={z[0]:.4f}")
    axes[1].set_title("λ vs Sequence Length")
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("λ")
    axes[1].legend()

    # ── λ trajectory over training ────────────────────────────────────────
    if lambda_history and ncols == 3:
        epochs, means = zip(*lambda_history)
        axes[2].plot(epochs, means, marker="o", markersize=3, color="steelblue")
        axes[2].set_title("λ Trajectory Over Training")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Mean λ")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize_lambda] Saved → {save_path}")
    print(f"  λ mean={lam_arr.mean():.4f}  std={lam_arr.std():.4f}")
    print(f"  λ min={lam_arr.min():.4f}   max={lam_arr.max():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Intra vs inter analysis across datasets
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def visualize_intra_inter(
    augmenter_results: dict,
    save_path: str = "plots/intra_inter_comparison.png",
) -> None:
    """Compare intra vs inter attention across multiple datasets.

    Args:
        augmenter_results: dict of dataset_name → {"intra": float, "inter": float}
            e.g. {"Beauty": {"intra": 0.4, "inter": 0.6},
                  "Yelp":   {"intra": 0.7, "inter": 0.3}}
    """
    _require_matplotlib()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    datasets = list(augmenter_results.keys())
    intra_vals = [augmenter_results[d]["intra"] for d in datasets]
    inter_vals = [augmenter_results[d]["inter"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, intra_vals, width, label="Intra (own sequence)", color="steelblue")
    ax.bar(x + width / 2, inter_vals, width, label="Inter (cross-user)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Mean attention weight")
    ax.set_title("ASTAR: Intra vs Inter Sequence Attention by Dataset")
    ax.legend()
    ax.set_ylim(0, 1)

    # Annotation
    for xi, (iv, ev) in enumerate(zip(intra_vals, inter_vals)):
        dominant = "intra" if iv > ev else "inter"
        ax.annotate(
            f"{dominant} dominant",
            xy=(xi, max(iv, ev) + 0.02),
            ha="center",
            fontsize=9,
            color="black",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize_intra_inter] Saved → {save_path}")


@torch.no_grad()
def compute_intra_inter_ratio(
    augmenter,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 50,
    lambda_ceiling: float = 0.8,
) -> Tuple[float, float]:
    """Compute mean intra / inter attention weights across the dataset.

    Returns:
        (mean_intra_weight, mean_inter_weight)
    """
    augmenter.eval()
    L = augmenter.max_seq_len
    K = augmenter.K

    intra_total = 0.0
    inter_total = 0.0
    count = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        _, T, _ = augmenter(input_ids, lambda_ceiling=lambda_ceiling)
        # T: [B, L, (1+K)*L]
        intra_weight = T[:, :, :L].sum(dim=-1).mean().item()
        inter_weight = T[:, :, L:].sum(dim=-1).mean().item()
        intra_total += intra_weight
        inter_total += inter_weight
        count += 1

    n = max(count, 1)
    return intra_total / n, inter_total / n
