"""
main.py — Entry point for ASTAR training and evaluation.

Usage:
    python main.py --data_path data/Beauty.txt --dataset Beauty
    python main.py --data_path data/Sports.txt --dataset Sports --ablation intra
    python main.py --data_path data/Yelp.txt   --dataset Yelp   --probe_only
"""

import argparse
import os
import sys
import json
from typing import Optional

import torch

# Add src to path when running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augmenter import ASTARAugmenter
from datasets import build_dataloaders
from diagnose import GANDiagnostics, probe_continuous_input
from recommender import SASRec
from trainers import AdvAugmentTrainer
from utils import set_seed
from visualize import (
    compute_intra_inter_ratio,
    visualize_T,
    visualize_intra_inter,
    visualize_lambda,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ASTAR: Adversarial Sequential Transformation Augmentation for Recommendation"
    )

    # Data
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to interaction file (space-separated user item [ts])")
    p.add_argument("--dataset", type=str, default="dataset",
                   help="Dataset name (used for logging and save paths)")
    p.add_argument("--max_seq_len", type=int, default=50)

    # Model architecture
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--inner_size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # Augmenter
    p.add_argument("--K", type=int, default=4,
                   help="Number of inter-sequence samples per example")
    p.add_argument("--tau_init", type=float, default=10.0)
    p.add_argument("--tau_floor", type=float, default=1.0)
    p.add_argument("--tau_decay", type=float, default=0.99)

    # Training
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--rec_lr", type=float, default=1e-3)
    p.add_argument("--aug_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)

    # Loss weights
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Weight of L_rec_aug in recommender loss")
    p.add_argument("--lambda_cl", type=float, default=0.1,
                   help="Weight of contrastive loss in recommender loss")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Weight of -L_contrast in augmenter loss")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Weight of L_rec_aug in augmenter loss")
    p.add_argument("--cl_temperature", type=float, default=0.07)

    # Evaluation
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20],
                   help="Cut-offs for HR@K and NDCG@K")

    # Ablations
    p.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "intra", "inter", "fixed_lambda", "no_adv"],
        help=(
            "full:         complete ASTAR model\n"
            "intra:        T operates on own sequence only (K=0)\n"
            "inter:        T operates on other sequences only (no own sequence)\n"
            "fixed_lambda: λ fixed at 0.5, not learned\n"
            "no_adv:       no adversarial game, T trained cooperatively"
        ),
    )

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   help="'auto', 'cpu', 'cuda', 'cuda:0', etc.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--plot_dir", type=str, default="plots")
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--probe_only", action="store_true",
                   help="Run continuous-input diagnostic probe and exit")
    p.add_argument("--visualize_only", action="store_true",
                   help="Load checkpoint and run visualisations only")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint to load for evaluation / visualisation")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--no_verbose", dest="verbose", action="store_false")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────


def build_models(args: argparse.Namespace, num_items: int, device: torch.device):
    """Instantiate recommender and augmenter."""
    K = 0 if args.ablation == "intra" else args.K

    recommender = SASRec(
        num_items=num_items,
        hidden_size=args.hidden_size,
        max_seq_len=args.max_seq_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        inner_size=args.inner_size,
        dropout=args.dropout,
    ).to(device)

    augmenter = ASTARAugmenter(
        num_items=num_items,
        hidden_size=args.hidden_size,
        max_seq_len=args.max_seq_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        inner_size=args.inner_size,
        dropout=args.dropout,
        K=K,
        tau_init=args.tau_init,
        tau_floor=args.tau_floor,
        tau_decay=args.tau_decay,
    ).to(device)

    # Initialise augmenter embeddings from recommender
    augmenter.copy_embeddings_from(recommender)

    return recommender, augmenter


# ─────────────────────────────────────────────────────────────────────────────
# Ablation modifications
# ─────────────────────────────────────────────────────────────────────────────


def apply_ablation(args: argparse.Namespace, trainer: AdvAugmentTrainer) -> None:
    """Patch trainer / augmenter for ablation studies."""
    if args.ablation == "fixed_lambda":
        # Monkey-patch the augmenter forward to fix λ at 0.5
        original_forward = trainer.augmenter.forward

        def fixed_lam_forward(input_ids, lambda_ceiling=0.8):
            aug_orig, T, lam = original_forward(input_ids, lambda_ceiling)
            B = input_ids.size(0)
            fixed = torch.full((B, 1), 0.5, device=input_ids.device)
            # Recompute aug with fixed lambda=0.5 using the underlying T_S and S_intra
            # We must rebuild from the T and S_pool since aug_orig used learned lam.
            # Retrieve S_intra from augmenter embeddings directly:
            S_intra = trainer.augmenter.item_embeddings(input_ids)
            # Recover T_S from the original blend equation:
            # aug_orig = lam3 * T_S + (1 - lam3) * S_intra
            # => T_S = (aug_orig - (1 - lam3) * S_intra) / lam3
            lam3 = lam.unsqueeze(2).clamp(min=1e-4)
            T_S = (aug_orig - (1.0 - lam3) * S_intra) / lam3
            # Apply fixed lambda=0.5
            aug = 0.5 * T_S + 0.5 * S_intra
            return aug, T, fixed

        trainer.augmenter.forward = fixed_lam_forward

    elif args.ablation == "no_adv":
        # Cooperative training: augmenter also minimises contrast loss (no negation)
        original_phase2 = trainer._phase2_aug

        def cooperative_phase2(input_ids, target_pos, target_neg):
            trainer.recommender.eval()
            trainer.augmenter.train()
            trainer.aug_optimizer.zero_grad()

            aug, _, lam = trainer._generate_aug(input_ids, grad=True)
            with torch.no_grad():
                repr_orig = trainer.recommender.get_representation(input_ids=input_ids)
            repr_aug = trainer.recommender.get_representation(inputs_embeds=aug)
            L_rec_aug = trainer.recommender.rec_loss(repr_aug, target_pos, target_neg)
            # Cooperative: minimise contrast (no negation)
            L_contrast = trainer.nce_loss(repr_orig.detach(), repr_aug)
            L_A = trainer.beta * L_rec_aug + trainer.alpha * L_contrast
            L_A.backward()
            torch.nn.utils.clip_grad_norm_(trainer.augmenter.parameters(), trainer.grad_clip_norm)
            trainer.aug_optimizer.step()
            return {
                "L_rec_aug_A": L_rec_aug.item(),
                "L_contrast_A": L_contrast.item(),
                "L_A": L_A.item(),
                "lambda_mean": lam.mean().item(),
            }

        trainer._phase2_aug = cooperative_phase2


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    set_seed(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, valid_loader, test_loader, num_users, num_items = build_dataloaders(
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Dataset: {args.dataset}  |  users={num_users}  items={num_items}")
    print(f"  train={len(train_loader.dataset)}  valid={len(valid_loader.dataset)}  test={len(test_loader.dataset)}")

    # ── Models ────────────────────────────────────────────────────────────
    recommender, augmenter = build_models(args, num_items, device)
    print(f"Recommender params: {sum(p.numel() for p in recommender.parameters()):,}")
    print(f"Augmenter params:   {sum(p.numel() for p in augmenter.parameters()):,}")

    # Load checkpoint if provided
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        recommender.load_state_dict(ckpt["recommender"])
        if "augmenter" in ckpt:
            augmenter.load_state_dict(ckpt["augmenter"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Diagnostic probe ──────────────────────────────────────────────────
    if args.probe_only:
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        probe_continuous_input(recommender, input_ids, device, L=args.max_seq_len)
        return

    # ── Visualise only ────────────────────────────────────────────────────
    if args.visualize_only:
        plot_dir = os.path.join(args.plot_dir, args.dataset)
        visualize_T(augmenter, test_loader, device,
                    save_path=os.path.join(plot_dir, "T_heatmap.png"))
        visualize_lambda(augmenter, test_loader, device,
                         save_path=os.path.join(plot_dir, "lambda_dist.png"))
        intra, inter = compute_intra_inter_ratio(augmenter, test_loader, device)
        print(f"Intra: {intra:.4f}  Inter: {inter:.4f}")
        return

    # ── Optimisers ────────────────────────────────────────────────────────
    rec_optimizer = torch.optim.Adam(
        recommender.parameters(),
        lr=args.rec_lr,
        weight_decay=args.weight_decay,
    )
    aug_optimizer = torch.optim.Adam(
        augmenter.parameters(),
        lr=args.aug_lr,
        weight_decay=args.weight_decay,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    save_dir = os.path.join(args.save_dir, args.dataset, args.ablation)
    trainer = AdvAugmentTrainer(
        recommender=recommender,
        augmenter=augmenter,
        rec_optimizer=rec_optimizer,
        aug_optimizer=aug_optimizer,
        device=device,
        gamma=args.gamma,
        lambda_cl=args.lambda_cl,
        alpha=args.alpha,
        beta=args.beta,
        cl_temperature=args.cl_temperature,
        warmup_epochs=args.warmup_epochs,
    )

    # Apply ablation patches
    apply_ablation(args, trainer)

    # ── Run diagnostic probe before training ──────────────────────────────
    sample_batch = next(iter(train_loader))
    sample_ids = sample_batch["input_ids"].to(device)
    probe_result = probe_continuous_input(recommender, sample_ids, device, L=args.max_seq_len)

    # ── Train ─────────────────────────────────────────────────────────────
    test_metrics = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        patience=args.patience,
        save_dir=save_dir,
        verbose=args.verbose,
        log_interval=args.log_interval,
    )

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "dataset": args.dataset,
        "ablation": args.ablation,
        "seed": args.seed,
        "probe": probe_result,
        "test_metrics": test_metrics,
        "args": vars(args),
    }
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {result_path}")

    # ── Visualisations ────────────────────────────────────────────────────
    plot_dir = os.path.join(args.plot_dir, args.dataset, args.ablation)
    try:
        visualize_T(augmenter, test_loader, device,
                    save_path=os.path.join(plot_dir, "T_heatmap.png"))
        visualize_lambda(augmenter, test_loader, device,
                         save_path=os.path.join(plot_dir, "lambda_dist.png"))
    except Exception as exc:
        print(f"[warn] Visualisation failed: {exc}")


if __name__ == "__main__":
    main()
