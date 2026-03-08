from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any, Tuple
import random
from evolution.dna_schema import ArchitectureDNA, ConvBlockDNA, HeadDNA

FILTERS = [16, 32, 64, 128]
KERNELS = [3, 5]
ACTS = ["relu", "leaky_relu"]
POOLS = ["max", "avg", "none"]
DROPS = [None, 0.1, 0.2, 0.3, 0.4]

def _pick_diff(rng: random.Random, cur, choices):
    x = rng.choice(choices)
    if x == cur and len(choices) > 1:
        while x == cur:
            x = rng.choice(choices)
    return x

def mutate_dna(parent: ArchitectureDNA, rng: random.Random) -> Tuple[ArchitectureDNA, Dict[str, Any]]:
    before = parent.to_dict()
    blocks = list(parent.conv_blocks)

    mutation_type = rng.choice([
        "change_filters", "change_kernel", "change_activation",
        "toggle_batchnorm", "change_pooling", "change_dropout",
        "add_block", "remove_block",
        "change_head_hidden", "change_head_dropout" # "toggle_gap",
    ])

    idx = rng.randrange(len(blocks))

    if mutation_type == "change_filters":
        b = blocks[idx]
        blocks[idx] = ConvBlockDNA(**{**asdict(b), "out_channels": _pick_diff(rng, b.out_channels, FILTERS)})

    elif mutation_type == "change_kernel":
        b = blocks[idx]
        blocks[idx] = ConvBlockDNA(**{**asdict(b), "kernel_size": _pick_diff(rng, b.kernel_size, KERNELS)})

    elif mutation_type == "change_activation":
        b = blocks[idx]
        blocks[idx] = ConvBlockDNA(**{**asdict(b), "activation": _pick_diff(rng, b.activation, ACTS)})

    elif mutation_type == "toggle_batchnorm":
        b = blocks[idx]
        blocks[idx] = ConvBlockDNA(**{**asdict(b), "batchnorm": (not b.batchnorm)})

    elif mutation_type == "change_pooling":
        b = blocks[idx]
        blocks[idx] = ConvBlockDNA(**{**asdict(b), "pooling": _pick_diff(rng, b.pooling, POOLS)})

    elif mutation_type == "change_dropout":
        b = blocks[idx]
        blocks[idx] = ConvBlockDNA(**{**asdict(b), "dropout": _pick_diff(rng, b.dropout, DROPS)})

    elif mutation_type == "add_block":
        if len(blocks) < 6:
            src = blocks[-1]
            blocks.append(ConvBlockDNA(**asdict(src)))

    elif mutation_type == "remove_block":
        if len(blocks) > 2:
            blocks.pop(idx)

    #elif mutation_type == "toggle_gap":
    #   child = ArchitectureDNA(
    #       version=parent.version, input_shape=parent.input_shape, num_classes=parent.num_classes,
    #       conv_blocks=blocks, use_gap=not parent.use_gap, head=parent.head,
    #        optimizer=parent.optimizer, lr=parent.lr
    #    )
    #    return child, {"type": mutation_type, "before": before, "after": child.to_dict()}

    elif mutation_type == "change_head_hidden":
        choices = [None, 128, 256, 512]
        new_hidden = _pick_diff(rng, parent.head.hidden_dim, choices)
        head = HeadDNA(hidden_dim=new_hidden, dropout=parent.head.dropout)
        child = ArchitectureDNA(parent.version, parent.input_shape, parent.num_classes, blocks, parent.use_gap, head, parent.optimizer, parent.lr)
        return child, {"type": mutation_type, "before": before, "after": child.to_dict()}

    elif mutation_type == "change_head_dropout":
        choices = [None, 0.1, 0.2, 0.3, 0.4, 0.5]
        new_drop = _pick_diff(rng, parent.head.dropout, choices)
        head = HeadDNA(hidden_dim=parent.head.hidden_dim, dropout=new_drop)
        child = ArchitectureDNA(parent.version, parent.input_shape, parent.num_classes, blocks, parent.use_gap, head, parent.optimizer, parent.lr)
        return child, {"type": mutation_type, "before": before, "after": child.to_dict()}

    child = ArchitectureDNA(
        version=parent.version,
        input_shape=parent.input_shape,
        num_classes=parent.num_classes,
        conv_blocks=blocks,
        use_gap=parent.use_gap,
        head=parent.head,
        optimizer=parent.optimizer,
        lr=parent.lr
    )
    return child, {"type": mutation_type, "before": before, "after": child.to_dict()}
