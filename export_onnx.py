import torch
from models.model_space import build_model
from experiments.best_genome import BEST_GENOME
from utils.utils import get_device

device = get_device("cuda")
model = build_model(BEST_GENOME, num_classes=10).to(device)
model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
model.eval()

dummy = torch.randn(1, 3, 32, 32).to(device)

torch.onnx.export(
    model, dummy, "outputs/best_model.onnx",
    input_names=["image"], output_names=["logits"],
    opset_version=12
)

print("Saved: outputs/best_model.onnx")
