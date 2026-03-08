import json
import matplotlib.pyplot as plt
import torch
from config import Config
from data.CIFAR10 import get_cifar10_loaders
from evolution.evolve import evolve
from evolution.dna_builder import build_model_from_dna
from evaluation.train_eval import evaluate, train_one
from utils.utils import set_seed, get_device, ensure_dir
import os
#os.makedirs("outputs/pareto_fronts", exist_ok=True)

# Phase 1 reproducibility (recommended)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def plot_history(history, out_path):
    gens = [h["gen"] + 1 for h in history]
    accs = [h["best_acc"] for h in history]
    plt.figure()
    plt.plot(gens, accs)
    plt.xlabel("Generation")
    plt.ylabel("Best Val Accuracy")
    plt.title("SENN: Accuracy vs Generation")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    import os
    cfg = Config()
    seed = int(os.environ.get("SENN_SEED", cfg.seed))
    set_seed(seed)
    cfg.seed = seed
    device = get_device(cfg.device)
    ensure_dir("outputs")

    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg.batch_size, cfg.num_workers)

    # Phase 1: evolve returns best_dna (ArchitectureDNA)
    best_dna, best_fit, (best_acc, best_params), history, run_dir = evolve(cfg, train_loader, val_loader, device)


    print("\n=== BEST ARCHITECTURE DNA ===")
    print(best_dna.to_json())
    print(f"Best arch_id: {best_dna.arch_id()}")
    print(f"Best fitness: {best_fit:.6f} | best val acc: {best_acc:.4f} | params: {best_params}")

    acc_plot_path = run_dir / "acc_vs_gen.png"
    plot_history(history, str(acc_plot_path))
    print(f"Saved: {acc_plot_path}")

    # Save the best model DNA and other info for later evaluation
    best_arch_path = run_dir / "best_architecture.json"
    with open(best_arch_path, "w") as f:
        json.dump(best_dna.to_dict(), f, indent=2)
    print(f"Saved: {best_arch_path}")

    # -------------------------
    # Phase 1: Final retrain best DNA (commented out for now)
    # -------------------------
    # You can execute this in a separate script manually
    # model = build_model_from_dna(best_dna).to(device)
    # opt = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=(best_dna.lr if best_dna.lr is not None else 1e-3),
    #     weight_decay=getattr(cfg, "weight_decay", 1e-4)
    # )

    # print(f"\n=== Final retrain for {cfg.final_train_epochs} epochs ===")
    # for ep in range(cfg.final_train_epochs):
    #     train_one(model, train_loader, opt, device)
    #     _, val_acc, _, _ = evaluate(model, val_loader, device)
    #     print(f"Epoch {ep+1}/{cfg.final_train_epochs} | val_acc={val_acc:.4f}")

    # # final test
    # _, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    # print(f"\n=== FINAL TEST ACCURACY (retrained) === {test_acc:.4f}")

    # # save final weights
    # torch.save(model.state_dict(), "outputs/best_phase1_model.pth")
    # print("Saved: outputs/best_phase1_model.pth")


if __name__ == "__main__":
    main()
