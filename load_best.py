import torch
from models.model_space import build_model
from experiments.best_genome import BEST_GENOME
from utils.utils import get_device

device = get_device("cuda")
model = build_model(BEST_GENOME, num_classes=10).to(device)

state = torch.load("outputs/best_model.pth", map_location=device)
model.load_state_dict(state)
model.eval()

print("Loaded model on:", next(model.parameters()).device)
print("Ready for inference.")
