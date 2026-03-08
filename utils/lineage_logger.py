from __future__ import annotations
from pathlib import Path
import csv
from typing import Optional

def init_lineage_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["child_id", "parent_id", "generation"])

def append_lineage(path: Path, child_id: str, parent_id: Optional[str], generation: int) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([child_id, parent_id or "", generation])
