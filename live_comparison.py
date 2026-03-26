import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Import necessary modules from the project
from config import Config
from evolution.dna_builder import build_model_from_dna
from evolution.dna_schema import ArchitectureDNA
from evaluate_best_model import get_latest_run_dir, get_best_arch_from_metrics
from utils.utils import get_device

# 1. Define the Baseline CNN (from our previous step)
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(), nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return self.fc_layer(x)

def load_best_senn_model(device):
    run_dir = get_latest_run_dir()
    if not run_dir:
        raise FileNotFoundError("Outputs directory not found")
        
    best_dna_dict = None
    best_arch_path = run_dir / "best_architecture.json"
    
    if best_arch_path.exists():
        with open(best_arch_path, "r") as f:
            best_dna_dict = json.load(f)
    else:
        best_arch_id = get_best_arch_from_metrics(run_dir)
        if best_arch_id:
            arch_json_path = run_dir / "population" / best_arch_id / "arch.json"
            if arch_json_path.exists():
                with open(arch_json_path, "r") as f:
                    best_dna_dict = json.load(f)
                    
    if not best_dna_dict:
        raise FileNotFoundError(f"Could not load best SENN architecture from {run_dir}")

    best_dna = ArchitectureDNA.from_dict(best_dna_dict)
    model = build_model_from_dna(best_dna).to(device)
    
    weights_path = run_dir / "best_phase1_model.pth"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded trained SENN weights from {weights_path}")
    else:
        print(f"Warning: Trained weights {weights_path} not found. Using untrained SENN model.")
        
    model.eval()
    return model

def main():
    # CIFAR-10 Class Names
    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    import warnings
    warnings.filterwarnings('ignore') # Ignore download warnings
    
    #torch.manual_seed(48)
    
    testset = torchvision.datasets.CIFAR10(root='./local_cifar_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True) # Get 5 random images

    # 3. Load Baseline Model
    baseline_model = BaselineCNN()
    # Load the weights we just saved
    baseline_model.load_state_dict(torch.load('models/baseline_weights.pth')) 
    baseline_model.eval()

    # 4. LOAD YOUR SENN MODEL HERE
    try:
        senn_model = load_best_senn_model(device)
    except Exception as e:
        print(f"Failed to load SENN model: {e}")
        return

    # 5. Get a batch of images
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # 6. Make Predictions
    with torch.no_grad():
        baseline_outputs = baseline_model(images)
        _, baseline_predicted = torch.max(baseline_outputs, 1)
        
        senn_outputs = senn_model(images)
        _, senn_predicted = torch.max(senn_outputs, 1)

    # 7. Visualize the Results
    # Move everything back to CPU for matplotlib
    images = images.cpu()
    labels = labels.cpu()
    baseline_predicted = baseline_predicted.cpu()
    senn_predicted = senn_predicted.cpu()

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle('Live Model Comparison: Baseline vs. SENN', fontsize=16, fontweight='bold')

    for i in range(5):
        # Un-normalize and format image for matplotlib
        img = images[i] / 2 + 0.5     
        npimg = img.numpy()
        axes[i].imshow(np.transpose(npimg, (1, 2, 0)))
        
        true_label = classes[labels[i]]
        base_pred = classes[baseline_predicted[i]]
        senn_pred = classes[senn_predicted[i]]

        # Color coding: Green if correct, Red if wrong
        base_color = 'green' if base_pred == true_label else 'red'
        senn_color = 'green' if senn_pred == true_label else 'red'

        axes[i].set_title(f"True: {true_label}", color='black', fontsize=10)
        
        # Add colored text for predictions
        axes[i].text(0.5, -0.15, f"Base: {base_pred}", color=base_color, ha='center', transform=axes[i].transAxes, fontweight='bold')
        axes[i].text(0.5, -0.30, f"SENN: {senn_pred}", color=senn_color, ha='center', transform=axes[i].transAxes, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)  # Ensure text below axes is not cropped
    plt.show()

if __name__ == "__main__":
    main()