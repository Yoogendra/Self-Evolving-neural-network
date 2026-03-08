import torch

def evaluate_model(model, val_loader, device):
    """
    Evaluate the pruned model's accuracy on the validation set.
    :param model: The pruned neural network model.
    :param val_loader: The validation data loader.
    :param device: The device to run the model on (CPU or GPU).
    :return: The accuracy of the model on the validation set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy
