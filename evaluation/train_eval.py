import torch
import torch.nn as nn
from tqdm import tqdm


def get_cosine_scheduler(optimizer, num_training_steps: int, warmup_steps: int = 0):
    """Cosine annealing LR. Step per batch. warmup_steps linear warmup."""
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(min(1.0, progress) * 3.14159265)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epochs_cosine(
    model, train_loader, val_loader, device,
    num_epochs: int, lr: float = 1e-3, weight_decay: float = 1e-4,
    warmup_epochs: int = 5, log_every: int = 10
):
    """
    Train model to convergence with cosine LR schedule.
    Returns (best_val_acc, final_test_metrics_dict).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    scheduler = get_cosine_scheduler(optimizer, total_steps, warmup_steps)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        train_acc = train_correct / train_total
        train_loss /= train_total
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    return best_val_acc


def train_one(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, val_loader, device, input_size):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    all_preds, all_y = [], []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(1)
        all_preds.append(pred.cpu())
        all_y.append(y.cpu())
        correct += int((pred == y).sum().item())
        total += x.size(0)
    acc = correct / total
    return total_loss / total, acc, torch.cat(all_y), torch.cat(all_preds)
