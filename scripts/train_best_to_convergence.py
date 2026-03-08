"""
Train the best architecture from a SENN run to convergence (cosine LR, full epochs).
Reports test accuracy and saves final metrics for paper-style evaluation.

Usage:
  python scripts/train_best_to_convergence.py --run_dir outputs/run_20260224_141329 [--epochs 200]
  python scripts/train_best_to_convergence.py --run_dir outputs/run_20260224_141329 --epochs 100 --no_cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import Config
from data.CIFAR10 import get_cifar10_loaders
from evolution.dna_builder import build_model_from_dna
from evolution.dna_schema import ArchitectureDNA
from evaluation.train_eval import train_epochs_cosine, evaluate
from evaluation.metrics import count_trainable_params, estimate_flops_conv_linear
from utils.utils import set_seed, get_device, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Train best SENN architecture to convergence")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run dir (e.g. outputs/run_YYYYMMDD_HHMMSS)")
    parser.add_argument("--epochs", type=int, default=200, help="Full training epochs (default 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")
    parser.add_argument("--out_dir", type=str, default=None, help="Write final_metrics.json here (default: run_dir)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")
    best_arch_path = run_dir / "best_architecture.json"
    if not best_arch_path.exists():
        raise SystemExit(f"Best architecture not found: {best_arch_path}")

    cfg = Config()
    set_seed(args.seed)
    device = get_device("cpu" if args.no_cuda else cfg.device)
    ensure_dir("outputs")

    with open(best_arch_path, "r") as f:
        best_dna_dict = json.load(f)
    best_dna = ArchitectureDNA.from_dict(best_dna_dict)
    model = build_model_from_dna(best_dna).to(device)
    # No pruning for final training — full capacity

    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg.batch_size, cfg.num_workers)
    lr = best_dna.lr if best_dna.lr is not None else 1e-3
    wd = getattr(cfg, "weight_decay", 1e-4)

    print(f"Training best arch (arch_id={best_dna.arch_id()}) for {args.epochs} epochs with cosine LR (lr={lr}, wd={wd})")
    best_val_acc = train_epochs_cosine(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs, lr=lr, weight_decay=wd,
        warmup_epochs=5, log_every=20
    )

    # Final test evaluation
    _, test_acc, y_true, y_pred = evaluate(model, test_loader, device, input_size=(1, 3, 32, 32))
    n_params = count_trainable_params(model)
    flops = estimate_flops_conv_linear(model, input_shape=(1, 3, 32, 32))

    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_metrics = {
        "arch_id": best_dna.arch_id(),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_val_accuracy": round(best_val_acc, 6),
        "test_accuracy": round(float(test_acc), 6),
        "param_count": int(n_params),
        "flops": int(flops),
    }
    metrics_path = out_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n=== FINAL TEST ACCURACY === {test_acc:.4f}")
    print(f"Params: {n_params}, FLOPs: {flops}")
    print(f"Saved: {metrics_path}")

    # Save final weights next to run_dir or in run_dir
    weights_path = run_dir / "best_model_trained_to_convergence.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved: {weights_path}")


if __name__ == "__main__":
    main()
