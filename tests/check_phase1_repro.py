from pathlib import Path
from evolution.dna_schema import ArchitectureDNA

runs = sorted(Path("outputs").glob("run_*"))

if not runs:
    raise RuntimeError(
        "No Phase-1 run found in outputs/. "
        "Run `python main.py` first and ensure evolution completes."
    )

latest_run = runs[-1]
population_dir = latest_run / "population"

arch_dirs = [p for p in population_dir.iterdir() if p.is_dir()]
if not arch_dirs:
    raise RuntimeError("No architectures found in population/. Evolution may have failed.")

arch_dir = arch_dirs[0]
dna = ArchitectureDNA.from_json((arch_dir / "arch.json").read_text(encoding="utf-8"))

print("Run directory:", latest_run.name)
print("Folder arch_id:", arch_dir.name)
print("Recomputed arch_id:", dna.arch_id())
print("MATCH:", arch_dir.name == dna.arch_id())
