from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    seed: int = 42
    device = "cuda"  # auto-fallback in utils
    generations: int = 6
    population_size: int = 8
    elite_k: int = 3
    train_epochs: int = 2

    # Fitness penalty (model size)
    size_penalty_lambda: float = 1e-7  # tune later

    num_classes: int = 10
    batch_size: int = 256
    num_workers: int = 2
    # Phase 1: final retrain of best DNA
    final_train_epochs: int = 40
    weight_decay: float = 1e-4

    # ---- Research / ablation flags ----
    use_pruning: bool = True
    pruning_percentage: float = 0.1
    use_pareto: bool = True  # If False, use fitness-only top-k selection