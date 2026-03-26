import argparse
import json
import csv
from pathlib import Path

import torch
import torch.optim as optim

from config import Config
from data.CIFAR10 import get_cifar10_loaders
from evolution.dna_builder import build_model_from_dna
from evolution.dna_schema import ArchitectureDNA
from evaluation.train_eval import evaluate, train_one
from pruning.prune_model import prune_model
from utils.utils import get_device

def get_latest_run_dir():
    """Dynamically fetch the most recent run directory under outputs/."""
    outputs = Path("outputs")
    if not outputs.exists():
        return None
    runs = sorted([d for d in outputs.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not runs:
        return None
    latest_run = runs[-1]
    print(f"Dynamically loaded latest run: {latest_run}")
    return latest_run

def get_best_arch_from_metrics(run_dir: Path):
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    
    best_acc = -1.0
    best_arch_id = None
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                acc = float(row["val_accuracy"])
                if acc > best_acc:
                    best_acc = acc
                    best_arch_id = row["arch_id"]
            except ValueError:
                continue
    return best_arch_id

def evaluate_best_model(run_dir: str | Path | None = None):
    run_dir = Path(run_dir) if run_dir else get_latest_run_dir()
    if not run_dir or not run_dir.is_dir():
        raise FileNotFoundError("No run directory found.")
    
    best_dna_dict = None
    best_arch_path = run_dir / "best_architecture.json"
    
    if best_arch_path.exists():
        with open(best_arch_path, "r") as f:
            best_dna_dict = json.load(f)
        print(f"Loaded existing best_architecture.json from {best_arch_path}")
    else:
        best_arch_id = get_best_arch_from_metrics(run_dir)
        if not best_arch_id:
            raise FileNotFoundError("Could not find best_architecture.json or valid metrics.csv")
            
        arch_json_path = run_dir / "population" / best_arch_id / "arch.json"
        with open(arch_json_path, "r") as f:
            best_dna_dict = json.load(f)

    # Rebuild the model from the DNA
    best_dna = ArchitectureDNA.from_dict(best_dna_dict)
    print(f"Best Architecture ID: {best_dna.arch_id()}")

    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_dna(best_dna).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SENN Parameter Count (Pre-Pruning): {total_params / 1e6:.4f} M")

    cfg = Config()  
    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg.batch_size, cfg.num_workers)

    opt = optim.AdamW(
        model.parameters(),
        lr=(best_dna.lr if best_dna.lr is not None else 1e-3),
        weight_decay=getattr(cfg, "weight_decay", 1e-4)
    )
    
    # Cosine Annealing for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.final_train_epochs)

    # --- PHASE 1: FULL TRAINING TO CONVERGENCE (Early Stopping) ---
    print(f"\n=== Phase 1: Training to Convergence (Max {cfg.final_train_epochs} epochs) ===")
    best_val_acc = 0.0
    patience = 10 
    epochs_no_improve = 0
    out_weights = run_dir / "best_phase1_model.pth"

    for ep in range(cfg.final_train_epochs):
        train_one(model, train_loader, opt, device)
        _, val_acc, _, _ = evaluate(model, val_loader, device, input_size=(3, 32, 32))
        scheduler.step()
        
        print(f"Epoch {ep+1}/{cfg.final_train_epochs} | val_acc={val_acc:.4f} | LR={opt.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_weights) 
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break 

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(out_weights, map_location=device, weights_only=True))

    # --- PHASE 2: POST-TRAINING PRUNING ---
    print("\n=== Phase 2: Executing Post-Training Pruning (10%) ===")
    model = prune_model(model, pruning_percentage=0.1)
    
    pruned_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SENN Parameter Count (Post-Pruning): {pruned_params / 1e6:.4f} M")

    # --- PHASE 3: FINE-TUNING ---
    print("\n=== Phase 3: Fine-tuning pruned model (5 epochs) ===")
    # Lower learning rate so we heal the pruned connections without destroying learned weights
    opt_finetune = optim.AdamW(model.parameters(), lr=1e-4) 
    
    for ep in range(5):
        train_one(model, train_loader, opt_finetune, device)
        _, val_acc, _, _ = evaluate(model, val_loader, device, input_size=(3, 32, 32))
        print(f"Fine-tune Epoch {ep+1}/5 | val_acc={val_acc:.4f}")

    # Save the final, pruned, and healed weights
    torch.save(model.state_dict(), out_weights)

    # --- FINAL EVALUATION ---
    _, test_acc, _, _ = evaluate(model, test_loader, device, input_size=(3, 32, 32))
    print(f"\n=== FINAL TEST ACCURACY (Pruned & Converged) === {test_acc:.4f}")
    print(f"Saved optimal weights to: {out_weights}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()
    evaluate_best_model(run_dir=args.run_dir)