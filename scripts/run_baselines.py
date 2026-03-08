"""
Train baseline architectures (SmallVGG, SmallResNet) with the same data and recipe as SENN.
Outputs test accuracy, params, FLOPs for paper comparison table.

Usage:
  python scripts/run_baselines.py --epochs 200
  python scripts/run_baselines.py --baselines small_vgg,small_resnet --epochs 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from config import Config
from data.CIFAR10 import get_cifar10_loaders
from evaluation.train_eval import train_epochs_cosine, evaluate
from evaluation.metrics import count_trainable_params, estimate_flops_conv_linear
from models.baselines import get_baseline
from utils.utils import set_seed, get_device, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Train baselines for SENN comparison")
    parser.add_argument("--baselines", type=str, default="small_vgg,small_resnet", help="Comma-separated baseline names")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/baseline_results.json")
    args = parser.parse_args()

    cfg = Config()
    set_seed(args.seed)
    device = get_device(cfg.device)
    ensure_dir("outputs")

    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg.batch_size, cfg.num_workers)
    baseline_names = [s.strip() for s in args.baselines.split(",")]
    results = []

    for name in baseline_names:
        print(f"\n{'='*50}\nBaseline: {name}\n{'='*50}")
        model = get_baseline(name, num_classes=cfg.num_classes).to(device)
        best_val = train_epochs_cosine(
            model, train_loader, val_loader, device,
            num_epochs=args.epochs, lr=1e-3, weight_decay=1e-4,
            warmup_epochs=5, log_every=20
        )
        _, test_acc, _, _ = evaluate(model, test_loader, device, input_size=(1, 3, 32, 32))
        n_params = count_trainable_params(model)
        flops = estimate_flops_conv_linear(model, input_shape=(1, 3, 32, 32))
        results.append({
            "name": name,
            "test_accuracy": float(test_acc),
            "best_val_accuracy": float(best_val),
            "param_count": int(n_params),
            "flops": int(flops),
            "epochs": args.epochs,
            "seed": args.seed,
        })
        print(f"Test accuracy: {test_acc:.4f}, Params: {n_params}, FLOPs: {flops}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"baselines": results, "epochs": args.epochs, "seed": args.seed}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
