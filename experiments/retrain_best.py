import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data.CIFAR10 import get_cifar10_loaders
from models.model_space import build_model
from evaluation.train_eval import evaluate
from utils.utils import get_device, set_seed, ensure_dir
from experiments.best_genome import BEST_GENOME

def train_epochs(model, train_loader, optimizer, device, epochs=20):
    import torch.nn as nn
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for ep in range(1, epochs+1):
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        for x, y in tqdm(train_loader, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += x.size(0)

        print(f"Epoch {ep:02d}: train_loss={total_loss/total:.4f} train_acc={correct/total:.4f}")

def main():
    ensure_dir("outputs")
    set_seed(42)
    device = get_device("cuda")
    torch.backends.cudnn.benchmark = True
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=256, num_workers=2)

    model = build_model(BEST_GENOME, num_classes=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=BEST_GENOME["opt"]["lr"], weight_decay=BEST_GENOME["opt"]["weight_decay"])

    # retrain for convergence
    train_epochs(model, train_loader, opt, device, epochs=20)

    # final evaluation
    _, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\nFINAL TEST ACCURACY: {test_acc:.4f}")

    # confusion matrix
    cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format="d")
    plt.title("CIFAR-10 Confusion Matrix (Best Evolved Model)")
    plt.savefig("outputs/confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("Saved: outputs/confusion_matrix.png")

    # save model weights
    torch.save(model.state_dict(), "outputs/best_model.pth")
    print("Saved: outputs/best_model.pth")

if __name__ == "__main__":
    main()
