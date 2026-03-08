from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
from evolution.dna_schema import ArchitectureDNA

ALLOWED_FILTERS = {16, 32, 64, 128}
ALLOWED_KERNELS = {3, 5}
ALLOWED_ACT = {"relu", "leaky_relu"}
ALLOWED_POOL = {"max", "avg", "none"}

@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]

def validate_dna(dna: ArchitectureDNA, *, min_spatial: int = 1) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    if dna.version != 1:
        errors.append(f"Unsupported DNA version: {dna.version}")

    if not (2 <= len(dna.conv_blocks) <= 6):
        errors.append(f"conv_blocks must be 2–6, got {len(dna.conv_blocks)}")

    for i, b in enumerate(dna.conv_blocks):
        if b.out_channels not in ALLOWED_FILTERS:
            errors.append(f"block[{i}].out_channels invalid: {b.out_channels}")
        if b.kernel_size not in ALLOWED_KERNELS:
            errors.append(f"block[{i}].kernel_size invalid: {b.kernel_size}")
        if b.activation not in ALLOWED_ACT:
            errors.append(f"block[{i}].activation invalid: {b.activation}")
        if b.pooling not in ALLOWED_POOL:
            errors.append(f"block[{i}].pooling invalid: {b.pooling}")
        if b.dropout is not None and not (0.0 <= b.dropout < 1.0):
            errors.append(f"block[{i}].dropout invalid: {b.dropout}")
        if not dna.use_gap:
            errors.append("use_gap must be True (enforced)")


    # shape simulation (deterministic)
    try:
        c, h, w = dna.input_shape
        _ = torch.zeros(1, c, h, w)
        for i, b in enumerate(dna.conv_blocks):
            # conv keeps H,W due to padding=k//2 in builder
            if b.pooling in ("max", "avg"):
                h //= 2
                w //= 2
            if h < min_spatial or w < min_spatial:
                errors.append(f"Spatial too small after block[{i}]: {h}x{w}")
                break

        if not dna.use_gap and (h * w) > (8 * 8):
            warnings.append(f"No GAP and spatial={h}x{w} may create a large classifier input.")
    except Exception as e:
        errors.append(f"Shape simulation failed: {repr(e)}")

    return ValidationResult(ok=(len(errors) == 0), errors=errors, warnings=warnings)
