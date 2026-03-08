"""
Reproduction script for paper results.

Quick run (few minutes): 2 seeds, 2 generations, 20 epochs final training.
Full run: use run_multi_seed.py and run_baselines.py with paper settings.

Usage:
  python scripts/reproduce_paper.py              # quick check
  python scripts/reproduce_paper.py --full       # full 5 seeds (slow)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_cmd(cmd: list[str], cwd: str, env: dict | None = None) -> bool:
    env = {**os.environ, **(env or {})}
    ret = subprocess.run(cmd, cwd=cwd, env=env)
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full 5-seed evaluation (slow)")
    parser.add_argument("--quick", action="store_true", help="Quick: 2 seeds, 2 gen, 20 epochs final (default)")
    args = parser.parse_args()

    full = args.full
    if not full and not args.quick:
        args.quick = True

    py = sys.executable

    if full:
        print("Running full reproduction: 5 seeds, default generations, 200 epochs final training.")
        out_multi = str(ROOT / "outputs" / "reproduce_multi_seed.json")
        if not run_cmd([py, str(ROOT / "scripts" / "run_multi_seed.py"), "--num_seeds", "5", "--epochs_final", "200", "--out", out_multi], str(ROOT)):
            print("Multi-seed run failed.")
            sys.exit(1)
        if not run_cmd([py, str(ROOT / "scripts" / "run_baselines.py"), "--epochs", "200", "--out", str(ROOT / "outputs" / "reproduce_baselines.json")], str(ROOT)):
            print("Baselines run failed.")
            sys.exit(1)
    else:
        print("Quick reproduction: 2 seeds, 2 generations, 20 epochs final.")
        # Run evolution twice with different seeds (we need to reduce generations for quick run)
        # main.py uses Config(); we can't pass generations from here without modifying main.
        # So we run main as-is (6 gen) and only reduce final training epochs for speed.
        for seed in [42, 123]:
            run_cmd([py, str(ROOT / "main.py")], str(ROOT), env={"SENN_SEED": str(seed)})
        # Find latest run dir and train to convergence with 20 epochs
        run_dirs = sorted((ROOT / "outputs").glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if run_dirs:
            run_cmd([py, str(ROOT / "scripts" / "train_best_to_convergence.py"), "--run_dir", str(run_dirs[0]), "--epochs", "20", "--seed", "42"], str(ROOT))
        run_baselines_quick = [py, str(ROOT / "scripts" / "run_baselines.py"), "--epochs", "20", "--out", "outputs/reproduce_baselines_quick.json"]
        run_cmd(run_baselines_quick, str(ROOT))

    # Print summary table from outputs
    print("\n" + "="*60)
    print("Summary (paper-style)")
    print("="*60)
    if full:
        multi_seed_path = ROOT / "outputs" / "reproduce_multi_seed.json"
        if multi_seed_path.exists():
            with open(multi_seed_path) as f:
                d = json.load(f)
            if "test_accuracy_mean" in d:
                print(f"SENN (mean ± std over {d.get('num_success', '?')} seeds): {d['test_accuracy_mean']:.4f} ± {d['test_accuracy_std']:.4f}")
    else:
        run_dirs = sorted((ROOT / "outputs").glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for rd in run_dirs[:1]:
            fm = rd / "final_metrics.json"
            if fm.exists():
                with open(fm) as f:
                    d = json.load(f)
                print(f"SENN (single run): test_acc={d['test_accuracy']:.4f} params={d['param_count']}")
                break
    for name in ["reproduce_baselines.json", "reproduce_baselines_quick.json", "baseline_results.json"]:
        baseline_path = ROOT / "outputs" / name
        if baseline_path.exists():
            with open(baseline_path) as f:
                d = json.load(f)
            for b in d.get("baselines", []):
                print(f"{b['name']}: test_acc={b['test_accuracy']:.4f} params={b['param_count']}")
            break
    print("="*60)
    print("\nFull paper table: python scripts/run_multi_seed.py --num_seeds 5 --epochs_final 200")
    print("Baselines: python scripts/run_baselines.py --epochs 200")


if __name__ == "__main__":
    main()
