# senn/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import time
import torch.nn as nn
from utils.latency import estimate_latency

@dataclass(frozen=True)
class ArchMetrics:
    val_accuracy: float
    param_count: int
    flops: int  # estimated per forward pass for input (1,3,32,32)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def estimate_flops_conv_linear(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
) -> int:
    """
    FLOPs estimate (multiply-add counted as 2 FLOPs) for Conv2d + Linear only.
    Deterministic given model + input_shape.
    """
    flops_total = 0
    hooks = []

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal flops_total
        # out: (N, Cout, Hout, Wout)
        x = inp[0]
        N = x.shape[0]
        Cout = out.shape[1]
        Hout = out.shape[2]
        Wout = out.shape[3]

        # groups support
        Cin = m.in_channels
        kH, kW = m.kernel_size
        groups = m.groups

        # MACs per output element = (Cin/groups) * kH * kW
        macs = N * Cout * Hout * Wout * (Cin // groups) * kH * kW
        flops_total += int(2 * macs)  # multiply + add

    def linear_hook(m: nn.Linear, inp, out):
        nonlocal flops_total
        x = inp[0]
        # x: (N, in_features) possibly with extra dims, flatten last dim
        N = x.shape[0]
        in_f = m.in_features
        out_f = m.out_features
        macs = N * in_f * out_f
        flops_total += int(2 * macs)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    first_param = next(model.parameters(), None)
    device = next(model.parameters(), torch.tensor([])).device
    dummy = torch.zeros(*input_shape, device=device)
    _ = model(dummy)

    for h in hooks:
        h.remove()

    return int(flops_total)

@torch.no_grad()
def estimate_latency(model: torch.nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32)) -> float:
    """
    Latency estimation for a single forward pass.
    This will measure the time it takes for a model to make one forward pass.
    """
    device = next(model.parameters()).device  # Get the device of the model
    dummy_input = torch.zeros(*input_shape, device=device)  # Create a dummy input tensor
    
    start_time = time.time()  # Start time measurement
    _ = model(dummy_input)  # Perform the forward pass
    end_time = time.time()  # End time measurement

    latency = end_time - start_time  # Calculate the latency
    return latency * 1000  # Convert to milliseconds
