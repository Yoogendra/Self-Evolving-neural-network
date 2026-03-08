# senn/plots_pareto.py
from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_pareto_scatter(df: pd.DataFrame, out_path: str, x: str, y: str, title: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure()
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_generation_fronts(metrics_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(metrics_csv)

    # per-generation plots
    for gen, gdf in df.groupby("generation"):
        plot_pareto_scatter(
            gdf, os.path.join(out_dir, f"gen_{gen}_acc_vs_params.png"),
            x="param_count", y="val_accuracy",
            title=f"Gen {gen}: Accuracy vs Params"
        )
        plot_pareto_scatter(
            gdf, os.path.join(out_dir, f"gen_{gen}_acc_vs_flops.png"),
            x="flops", y="val_accuracy",
            title=f"Gen {gen}: Accuracy vs FLOPs"
        )

def plot_trends(metrics_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(metrics_csv)

    # best accuracy per gen
    best_acc = df.groupby("generation")["val_accuracy"].max().reset_index()
    plt.figure()
    plt.plot(best_acc["generation"], best_acc["val_accuracy"])
    plt.xlabel("generation")
    plt.ylabel("best_val_accuracy")
    plt.title("Best Validation Accuracy Trend")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trend_best_accuracy.png"), dpi=200)
    plt.close()

    # efficiency trend (min params, min flops) per gen
    min_params = df.groupby("generation")["param_count"].min().reset_index()
    plt.figure()
    plt.plot(min_params["generation"], min_params["param_count"])
    plt.xlabel("generation")
    plt.ylabel("min_param_count")
    plt.title("Efficiency Trend: Minimum Params per Generation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trend_min_params.png"), dpi=200)
    plt.close()

    min_flops = df.groupby("generation")["flops"].min().reset_index()
    plt.figure()
    plt.plot(min_flops["generation"], min_flops["flops"])
    plt.xlabel("generation")
    plt.ylabel("min_flops")
    plt.title("Efficiency Trend: Minimum FLOPs per Generation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trend_min_flops.png"), dpi=200)
    plt.close()
