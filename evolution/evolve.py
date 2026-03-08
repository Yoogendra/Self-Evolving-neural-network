import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from models.model_space import random_genome, ACTS, POOLS
from evaluation.train_eval import train_one, evaluate

from evolution.dna_schema import ArchitectureDNA
from evolution.dna_builder import build_model_from_dna
from evolution.dna_validator import validate_dna
from evolution.dna_mutation import mutate_dna
from utils.latency import estimate_latency
from utils.run_registry import ArchMeta, create_run_dir, save_arch_artifacts
from utils.lineage_logger import init_lineage_csv, append_lineage
from utils.mutation_logger import init_mutation_history, append_mutation
from evaluation.metrics import count_trainable_params, estimate_flops_conv_linear
from utils.metrics_logger import init_metrics_csv, append_metrics_row
from evolution.pareto import Candidate, pareto_front, select_survivors_from_front
from evaluation.pareto_plots import plot_generation_fronts, plot_trends
from evolution.pareto import nsga2_select
from pruning.prune_model import prune_model
from evaluation.evaluate_model import evaluate_model

# -------------------------
# Phase 1 Individual
# -------------------------
@dataclass
class Individual:
    dna: ArchitectureDNA
    meta: ArchMeta


# -------------------------
# Fitness (unchanged)
# -------------------------
def fitness(acc: float, n_params: int, lam: float) -> float:
    return acc - lam * n_params


# -------------------------
# Deterministic repair helpers (Phase 1 safe)
# -------------------------
_ALLOWED_FILTERS = [16, 32, 64, 128]

def _snap_to_allowed(x: int, allowed: list[int]) -> int:
    # deterministic: choose nearest; tie -> smaller
    return min(allowed, key=lambda a: (abs(a - x), a))

def _map_activation(act: str) -> str:
    a = str(act).lower()
    if a in ("relu",):
        return "relu"
    # treat anything else as leaky_relu if it resembles it
    return "leaky_relu"

def _map_pool(pool) -> str:
    p = "none" if pool is None else str(pool).lower()
    if p in ("none",):
        return "none"
    if p in ("max", "maxpool", "max_pool"):
        return "max"
    if p in ("avg", "avgpool", "avg_pool"):
        return "avg"
    # fallback (deterministic)
    return "none"


# -------------------------
# Phase 0 genome -> Phase 1 DNA bridge
# -------------------------
def genome_to_dna(genome: dict, num_classes: int) -> ArchitectureDNA:
    blocks = []
    for b in genome["blocks"]:
        out_raw = int(b["out"])
        out_fixed = _snap_to_allowed(out_raw, _ALLOWED_FILTERS)  # deterministic repair

        drop = float(b["drop"])
        blocks.append({
            "out_channels": out_fixed,
            "kernel_size": int(b["k"]),
            "activation": _map_activation(b["act"]),
            "pooling": _map_pool(b["pool"]),
            "batchnorm": bool(b["bn"]),
            "dropout": None if drop <= 0.0 else drop,
        })

    # Keep head consistent + simple in Phase 1 (can be expanded later safely)
    head = {"hidden_dim": None, "dropout": None}

    d = {
        "version": 1,
        "input_shape": [3, 32, 32],
        "num_classes": int(num_classes),
        "conv_blocks": blocks,
        "use_gap": True,                 # stable across variable shapes
        "head": head,
        "optimizer": "adamw",
        "lr": float(genome["opt"]["lr"]),  # hint; optimizer uses this if present
    }
    return ArchitectureDNA.from_dict(d)


# -------------------------
# Population init (DNA Individuals) — pass rng for full determinism
# -------------------------
def init_population(P: int, num_classes: int, rng: random.Random) -> list[Individual]:
    pop: list[Individual] = []
    for _ in range(P):
        g = random_genome(rng=rng)
        dna = genome_to_dna(g, num_classes=num_classes)
        arch_id = dna.arch_id()
        pop.append(Individual(
            dna=dna,
            meta=ArchMeta(arch_id=arch_id, parent_id=None, generation=0)
        ))
    return pop


# -------------------------
# Evolution loop (Phase 1)
# -------------------------
def evolve(cfg, train_loader, val_loader, device):
    # ---- Phase 1: run artifacts
    from evaluation.metrics import estimate_latency
    run_dir = create_run_dir(Path("outputs"))
    lineage_csv = run_dir / "lineage.csv"
    mutation_json = run_dir / "mutation_history.json"
    init_lineage_csv(lineage_csv)
    init_mutation_history(mutation_json)

    # ---- Phase 2: global metrics log (required)
    metrics_csv = run_dir / "metrics.csv"
    init_metrics_csv(str(metrics_csv))

    # ---- Phase 2: pareto plots output (required)
    pareto_dir = run_dir / "pareto_fronts"
    pareto_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic RNG (use cfg.seed if present)
    seed = getattr(cfg, "seed", 42)
    rng = random.Random(seed)

    pop = init_population(cfg.population_size, cfg.num_classes, rng=rng)
    history = []

    best_dna = None
    best_fit = -1e9
    best_metrics = None

    use_pruning = getattr(cfg, "use_pruning", True)
    use_pareto = getattr(cfg, "use_pareto", True)
    pruning_pct = getattr(cfg, "pruning_percentage", 0.1)

    search_start_time = time.perf_counter()
    total_archs_evaluated = 0

    for gen in range(cfg.generations):
        scored = []
        print(f"\n=== Generation {gen+1}/{cfg.generations} ===")

        for i, indiv in enumerate(pop):
            # ---- Phase 1: validate DNA before training (never crash)
            val = validate_dna(indiv.dna)
            if not val.ok:
                save_arch_artifacts(
                    run_dir,
                    indiv.meta,
                    indiv.dna,
                    validation={"ok": False, "errors": val.errors, "warnings": val.warnings},
                    metrics={"skipped": True, "reason": "invalid_dna"}
                )
                print(f"Model {i+1}: SKIPPED (invalid DNA) first_error={val.errors[0] if val.errors else 'unknown'}")
                continue

            # ---- Phase 1: build model ONLY from DNA (deterministic)
            model = build_model_from_dna(indiv.dna).to(device)
            if use_pruning:
                model = prune_model(model, pruning_percentage=pruning_pct)
                val_acc_pre = evaluate_model(model, val_loader, device)
                print(f"Validation accuracy after pruning: {val_acc_pre:.4f}")

            # Optimizer (keeps your behavior)
            lr = indiv.dna.lr if indiv.dna.lr is not None else getattr(cfg, "lr", 1e-3)
            wd = getattr(cfg, "weight_decay", 1e-4)

            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            for _ in range(cfg.train_epochs):
                train_one(model, train_loader, opt, device)

            # Inside the evolve function
            input_size = (3, 32, 32)  # CIFAR-10 input size
            _, val_acc, _, _ = evaluate(model, val_loader, device, input_size=(1, 3, 32, 32))

            # ---- Phase 2 metrics (deterministic)
            n_params = count_trainable_params(model)
            flops = estimate_flops_conv_linear(model, input_shape=(1, 3, 32, 32))
            latency = estimate_latency(model, input_shape=(1, 3, 32, 32))  # Measure latency

            # ---- Phase 1 fitness kept for comparison (do NOT remove yet)
            fit = fitness(val_acc, n_params, cfg.size_penalty_lambda)

            # store full metrics for selection
            scored.append({
                "fitness": float(fit),
                "val_acc": float(val_acc),
                "params": int(n_params),
                "flops": int(flops),
                "latency": float(latency),
                "indiv": indiv,
            })

            # ---- Phase 2 required CSV logging
            append_metrics_row(str(metrics_csv), {
                "arch_id": indiv.meta.arch_id,
                "generation": gen,
                "val_accuracy": float(val_acc),
                "param_count": int(n_params),
                "flops": int(flops),
                "latency": float(latency)if latency is not None else 0.0,  # Include latency in the row
            })


            # ---- Phase 1: save per-arch artifacts
            arch_dir = save_arch_artifacts(
                run_dir,
                indiv.meta,
                indiv.dna,
                validation={"ok": True, "errors": val.errors, "warnings": val.warnings},
                metrics={
                    "val_acc": float(val_acc),
                    "params": int(n_params),
                    "flops": int(flops),
                    "fitness": float(fit)
                }
            )

            # optional: save weights for every evaluated arch
            if getattr(cfg, "save_all_weights", False):
                torch.save(model.state_dict(), arch_dir / "model.pth")

            total_archs_evaluated += 1
            print(f"Model {i+1}: acc={val_acc:.4f} params={n_params} flops={flops} fit={fit:.6f}")

        if not scored:
            raise RuntimeError("All architectures were invalid and skipped. Check validator constraints / mutation space.")

        # ---- Phase 1: keep best-by-fitness tracking (unchanged behavior)
        scored.sort(key=lambda r: r["fitness"], reverse=True)
        top = scored[0]
        top_fit, top_acc, top_params = top["fitness"], top["val_acc"], top["params"]
        top_indiv = top["indiv"]

        history.append({
            "gen": gen,
            "best_acc": top_acc,
            "best_fit": top_fit,
            "best_params": top_params,
            "best_flops": top["flops"],
        })

        if top_fit > best_fit:
            best_fit = top_fit
            best_dna = top_indiv.dna
            best_metrics = (top_acc, top_params)

        # ---- Phase 2A: Survivor selection (Pareto/NSGA-II or fitness-only)
        if use_pareto:
            candidates = [
                Candidate(
                    arch_id=r["indiv"].meta.arch_id,
                    generation=gen,
                    val_accuracy=r["val_acc"],
                    param_count=r["params"],
                    flops=r["flops"],
                    latency=r["latency"],
                )
                for r in scored
            ]
            survivors = nsga2_select(candidates, k=cfg.elite_k)
            survivor_ids = {c.arch_id for c in survivors}
        else:
            # Fitness-only: take top-k by fitness
            scored_by_fit = sorted(scored, key=lambda r: r["fitness"], reverse=True)
            survivor_ids = {scored_by_fit[i]["indiv"].meta.arch_id for i in range(min(cfg.elite_k, len(scored_by_fit)))}
        elites = [r["indiv"] for r in scored if r["indiv"].meta.arch_id in survivor_ids]


        # ---- next population: elites + mutated children (same logic)
        new_pop: list[Individual] = []

        # carry elites forward
        for e in elites:
            new_pop.append(Individual(
                dna=e.dna,
                meta=ArchMeta(
                    arch_id=e.dna.arch_id(),
                    parent_id=e.meta.arch_id,   # elite copied forward → parent is itself
                    generation=gen + 1
                )
            ))

        # fill remaining with mutated children
        while len(new_pop) < cfg.population_size:
            parent = rng.choice(elites)
            child_dna, mut = mutate_dna(parent.dna, rng)
            d = child_dna.to_dict()
            d["use_gap"] = True
            child_dna = ArchitectureDNA.from_dict(d)
            mut["after"] = child_dna.to_dict()
            child_id = child_dna.arch_id()

            child = Individual(
                dna=child_dna,
                meta=ArchMeta(arch_id=child_id, parent_id=parent.meta.arch_id, generation=gen + 1)
            )

            # lineage + mutation logs
            append_lineage(lineage_csv, child_id, parent.meta.arch_id, gen + 1)
            append_mutation(mutation_json, {
                "child_id": child_id,
                "parent_id": parent.meta.arch_id,
                "generation": gen + 1,
                **mut
            })

            new_pop.append(child)

        pop = new_pop

    search_elapsed = time.perf_counter() - search_start_time
    search_cost = {
        "total_architectures_evaluated": total_archs_evaluated,
        "search_wall_time_seconds": round(search_elapsed, 2),
        "generations": cfg.generations,
        "population_size": cfg.population_size,
        "seed": seed,
        "use_pruning": use_pruning,
        "use_pareto": use_pareto,
    }
    (run_dir / "search_cost.json").write_text(json.dumps(search_cost, indent=2), encoding="utf-8")
    print(f"Search cost: {total_archs_evaluated} archs, {search_elapsed:.1f}s wall time")

    # ---- Phase 2 required plots
    plot_generation_fronts(str(metrics_csv), str(pareto_dir))
    plot_trends(str(metrics_csv), str(pareto_dir))

    # Make canonical "gen_k.png" files from acc_vs_params plots
    for g in range(cfg.generations):
        src = pareto_dir / f"gen_{g}_acc_vs_params.png"
        dst = pareto_dir / f"gen_{g}.png"
        if src.exists():
            shutil.copyfile(src, dst)
    best_arch_path = run_dir / "best_architecture.json"
    with open(best_arch_path, "w") as f:
        json.dump(best_dna.to_dict(), f, indent=2)
    print(f"Saved: {best_arch_path}")

    return best_dna, best_fit, best_metrics, history , run_dir