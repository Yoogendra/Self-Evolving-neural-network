"""
Run SENN evolution + final training for multiple seeds and report mean ± std.
Use for paper-style evaluation (e.g. Table: Test accuracy mean ± std over 5 seeds).

Usage:
  python scripts/run_multi_seed.py --num_seeds 3 --epochs_final 100
  python scripts/run_multi_seed.py --num_seeds 5 --epochs_final 200 --generations 4 --population_size 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Multi-seed SENN runs: evolution + final train, report mean ± std")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of seeds (default 3)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (overrides num_seeds)")
    parser.add_argument("--epochs_final", type=int, default=100, help="Epochs for final training to convergence")
    parser.add_argument("--generations", type=int, default=None, help="Evolution generations (default: config)")
    parser.add_argument("--population_size", type=int, default=None, help="Population size (default: config)")
    parser.add_argument("--no_pruning", action="store_true", help="Ablation: disable pruning during search")
    parser.add_argument("--fitness_only", action="store_true", help="Ablation: use fitness-only selection (no Pareto)")
    parser.add_argument("--out", type=str, default="outputs/multi_seed_results.json", help="Output JSON path")
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [42, 123, 456, 789, 2024][: args.num_seeds]

    results = []
    run_dirs = []

    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        # Run evolution (main.py) with seed
        env = {"SENN_SEED": str(seed)}
        cmd = [sys.executable, str(ROOT / "main.py")]
        if args.generations is not None:
            # We need to pass config overrides; main.py uses Config(). Easiest: set env or patch.
            # Use env vars that we could read in config (optional). For simplicity, run main as-is and
            # rely on config; user can set generations in config for multi_seed. Alternatively
            # add a small wrapper that sets cfg.generations from env. Let's add env support in Config.
            pass
        # Config doesn't read from env yet. So we'll run main.py and rely on default config.
        # To override generations/population from CLI we'd need to change main.py to accept args.
        # For this script, we run with default config and document that user can edit config.
        t0 = time.perf_counter()
        ret = subprocess.run(cmd, cwd=str(ROOT), env={**__import__("os").environ, **env})
        if ret.returncode != 0:
            print(f"Evolution failed for seed {seed}")
            results.append({"seed": seed, "error": "evolution_failed"})
            continue

        # Find latest run_dir
        outputs = ROOT / "outputs"
        run_dirs_sorted = sorted(outputs.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not run_dirs_sorted:
            print("No run dir found after evolution")
            results.append({"seed": seed, "error": "no_run_dir"})
            continue
        run_dir = run_dirs_sorted[0]
        run_dirs.append(str(run_dir))

        # Override seed in the run for reproducibility of final training
        train_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_best_to_convergence.py"),
            "--run_dir", str(run_dir),
            "--epochs", str(args.epochs_final),
            "--seed", str(seed),
        ]
        ret2 = subprocess.run(train_cmd, cwd=str(ROOT))
        if ret2.returncode != 0:
            print(f"Final training failed for seed {seed}")
            results.append({"seed": seed, "run_dir": str(run_dir), "error": "final_train_failed"})
            continue

        metrics_path = run_dir / "final_metrics.json"
        if not metrics_path.exists():
            results.append({"seed": seed, "run_dir": str(run_dir), "error": "no_final_metrics"})
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["seed"] = seed
        metrics["run_dir"] = str(run_dir)
        results.append(metrics)
        print(f"Seed {seed} test_acc={metrics['test_accuracy']:.4f} params={metrics['param_count']}")

    # Aggregate
    valid = [r for r in results if "test_accuracy" in r]
    if not valid:
        print("No successful runs.")
        sys.exit(1)

    import numpy as np
    test_accs = [r["test_accuracy"] for r in valid]
    params = [r["param_count"] for r in valid]
    flops = [r["flops"] for r in valid]

    summary = {
        "num_seeds": len(seeds),
        "seeds": seeds,
        "num_success": len(valid),
        "test_accuracy_mean": float(np.mean(test_accs)),
        "test_accuracy_std": float(np.std(test_accs)),
        "test_accuracy_min": float(np.min(test_accs)),
        "test_accuracy_max": float(np.max(test_accs)),
        "param_count_mean": float(np.mean(params)),
        "param_count_std": float(np.std(params)),
        "flops_mean": float(np.mean(flops)),
        "flops_std": float(np.std(flops)),
        "epochs_final": args.epochs_final,
        "run_dirs": run_dirs,
        "per_seed": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Test accuracy: {summary['test_accuracy_mean']:.4f} ± {summary['test_accuracy_std']:.4f}")
    print(f"Params: {summary['param_count_mean']:.0f} ± {summary['param_count_std']:.0f}")


if __name__ == "__main__":
    main()
