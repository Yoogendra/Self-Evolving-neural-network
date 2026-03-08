# pruning/prune_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def prune_layer(layer, pruning_percentage=0.2):
    """
    Prune the given layer by removing weights with the smallest magnitudes.
    """
    # Get the weight tensor
    weight = layer.weight.data.abs()

    # Calculate the threshold to prune weights below this value
    threshold = torch.kthvalue(weight.view(-1), int(weight.numel() * pruning_percentage)).values

    # Set weights below threshold to 0
    layer.weight.data[weight < threshold] = 0

def prune_model(model, pruning_percentage=0.1):
    """
    Prune the model by removing unimportant weights.
    :param model: The neural network model.
    :param pruning_percentage: The percentage of weights to prune in each layer.
    """
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):  # For Linear layers
            prune_layer(layer, pruning_percentage)
        # Add conditions for other layer types if needed (e.g., Conv2d)
        # Example for Conv2D layers (you can add more layers if needed)
        elif isinstance(layer, nn.Conv2d):  
            prune_layer(layer, pruning_percentage)
        
    return model
