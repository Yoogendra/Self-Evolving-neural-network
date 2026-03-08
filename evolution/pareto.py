from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

@dataclass(frozen=True)
class Candidate:
    arch_id: str
    generation: int
    val_accuracy: float
    param_count: int
    flops: int
    latency: float

def dominates(a: Candidate, b: Candidate) -> bool:
    not_worse = (
        a.val_accuracy >= b.val_accuracy and
        a.param_count <= b.param_count and
        a.flops <= b.flops and
        a.latency <= b.latency  # Compare latency here
    )
    strictly_better = (
        a.val_accuracy > b.val_accuracy or
        a.param_count < b.param_count or
        a.flops < b.flops or
        a.latency < b.latency  # Add latency to strict improvement
    )
    return not_worse and strictly_better

def pareto_front(cands: List[Candidate]) -> List[Candidate]:
    front = []
    for i, a in enumerate(cands):
        dominated = False
        for j, b in enumerate(cands):
            if i != j and dominates(b, a):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front

def select_survivors_from_front(
    front: List[Candidate],
    k: int,
    tie_break: str = "accuracy",
    uniform_sample: bool = False,
    rng=None
) -> List[Candidate]:
    if len(front) <= k:
        return front

    if uniform_sample:
        if rng is None:
            raise ValueError("rng required for uniform sampling to preserve reproducibility")
        return rng.sample(front, k)

    if tie_break == "accuracy":
        return sorted(front, key=lambda c: (-c.val_accuracy, c.arch_id))[:k]

    return sorted(front, key=lambda c: (-c.val_accuracy, c.param_count, c.flops, c.arch_id))[:k]

def non_dominated_sort(cands: List[Candidate]) -> List[List[Candidate]]:
    fronts: List[List[Candidate]] = []
    S = {c: [] for c in cands}
    n = {c: 0 for c in cands}

    for p in cands:
        for q in cands:
            if p is q:
                continue
            if dominates(p, q):
                S[p].append(q)
            elif dominates(q, p):
                n[p] += 1

        if n[p] == 0:
            if not fronts:
                fronts.append([])
            fronts[0].append(p)

    i = 0
    while i < len(fronts):
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts

def crowding_distance(front: List[Candidate]) -> Dict[str, float]:
    distance = {c.arch_id: 0.0 for c in front}
    if len(front) <= 2:
        for c in front:
            distance[c.arch_id] = float("inf")
        return distance

    objectives = [
        ("val_accuracy", True),   # maximize
        ("param_count", False),   # minimize
        ("flops", False),         # minimize
    ]

    for attr, maximize in objectives:
        front_sorted = sorted(front, key=lambda c: getattr(c, attr), reverse=maximize)

        distance[front_sorted[0].arch_id] = float("inf")
        distance[front_sorted[-1].arch_id] = float("inf")

        vals = [getattr(c, attr) for c in front_sorted]
        min_val = min(vals)
        max_val = max(vals)
        if max_val == min_val:
            continue

        for i in range(1, len(front_sorted) - 1):
            prev_val = getattr(front_sorted[i - 1], attr)
            next_val = getattr(front_sorted[i + 1], attr)
            distance[front_sorted[i].arch_id] += (next_val - prev_val) / (max_val - min_val)

    return distance

def nsga2_select(candidates: List[Candidate], k: int) -> List[Candidate]:
    fronts = non_dominated_sort(candidates)
    selected: List[Candidate] = []

    for front in fronts:
        if len(selected) + len(front) <= k:
            selected.extend(front)
        else:
            distances = crowding_distance(front)
            # deterministic tie-breaks after distance
            front_sorted = sorted(
                front,
                key=lambda c: (
                    distances[c.arch_id],
                    c.val_accuracy,
                    -c.param_count,
                    -c.flops,
                    c.arch_id
                ),
                reverse=True
            )
            remaining = k - len(selected)
            selected.extend(front_sorted[:remaining])
            break

    return selected
