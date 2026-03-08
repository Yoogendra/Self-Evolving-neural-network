# senn/logging_metrics.py
from __future__ import annotations
import csv
import os
from typing import Dict, Any

METRICS_HEADER = ["arch_id", "generation", "val_accuracy", "param_count", "flops", "latency"]

def init_metrics_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(METRICS_HEADER)

def append_metrics_row(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_HEADER)
        w.writerow({
            "arch_id": row["arch_id"],
            "generation": row["generation"],
            "val_accuracy": float(row["val_accuracy"]),
            "param_count": int(row["param_count"]),
            "flops": int(row["flops"]),
            "latency": float(row["latency"]),  # Ensure latency is included
        })


