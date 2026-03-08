from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, List

def init_mutation_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps([], indent=2), encoding="utf-8")

def append_mutation(path: Path, record: Dict[str, Any]) -> None:
    data: List[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    data.append(record)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
