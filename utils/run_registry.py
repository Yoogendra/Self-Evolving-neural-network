from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from evolution.dna_schema import ArchitectureDNA

@dataclass(frozen=True)
class ArchMeta:
    arch_id: str
    parent_id: Optional[str]
    generation: int

def create_run_dir(base: Path = Path("outputs")) -> Path:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base / run_name
    (run_dir / "population").mkdir(parents=True, exist_ok=True)
    return run_dir

def save_arch_artifacts(run_dir: Path, meta: ArchMeta, dna: ArchitectureDNA, validation: Dict[str, Any], metrics: Dict[str, Any]) -> Path:
    arch_dir = run_dir / "population" / meta.arch_id
    arch_dir.mkdir(parents=True, exist_ok=True)

    (arch_dir / "arch.json").write_text(dna.to_json(), encoding="utf-8")
    (arch_dir / "meta.json").write_text(json.dumps({
        "arch_id": meta.arch_id,
        "parent_id": meta.parent_id,
        "generation": meta.generation
    }, indent=2, sort_keys=True), encoding="utf-8")
    (arch_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    (arch_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return arch_dir
