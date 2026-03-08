# utils/latency.py
import time
import torch
from typing import Tuple

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
