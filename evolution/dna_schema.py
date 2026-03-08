from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal
import json
import hashlib

Activation = Literal["relu", "leaky_relu"]
Pooling = Literal["max", "avg", "none"]

@dataclass(frozen=True)
class ConvBlockDNA:
    out_channels: int                 # 16/32/64/128
    kernel_size: int                  # 3 or 5
    activation: Activation            # relu/leaky_relu
    pooling: Pooling                  # max/avg/none
    batchnorm: bool                   # True/False
    dropout: Optional[float] = None   # None or 0<drop<1

@dataclass(frozen=True)
class HeadDNA:
    hidden_dim: Optional[int] = None  # None => linear only
    dropout: Optional[float] = None

@dataclass(frozen=True)
class ArchitectureDNA:
    version: int
    input_shape: List[int]            # [3,32,32]
    num_classes: int                  # 10
    conv_blocks: List[ConvBlockDNA]   # 2–6
    use_gap: bool                     # True/False
    head: HeadDNA
    # optional hints (Phase 1 keeps them optional)
    optimizer: Optional[str] = None
    lr: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def arch_id(self) -> str:
        # stable, deterministic identifier
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]

    def to_json(self) -> str:
        return self.canonical_json()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ArchitectureDNA":
        blocks = [ConvBlockDNA(**b) for b in d["conv_blocks"]]
        head = HeadDNA(**d["head"])
        return ArchitectureDNA(
            version=d["version"],
            input_shape=d["input_shape"],
            num_classes=d["num_classes"],
            conv_blocks=blocks,
            use_gap=d["use_gap"],
            head=head,
            optimizer=d.get("optimizer"),
            lr=d.get("lr"),
        )

    @staticmethod
    def from_json(s: str) -> "ArchitectureDNA":
        return ArchitectureDNA.from_dict(json.loads(s))
