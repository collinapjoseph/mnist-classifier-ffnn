"""
MNIST Classification Pipeline
Feedforward Neural Network using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "batch_size": 128,
    "epochs": 10,
    "learning_rate": 1e-3,
    "hidden_sizes": [512, 256, 128],
    "dropout_rate": 0.3,
    "seed": 42,
}

torch.manual_seed(CONFIG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 2. Data Loading & Preprocessing
# ─────────────────────────────────────────────
def get_dataloaders(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_dataset):,}  |  Test samples: {len(test_dataset):,}")
    return train_loader, test_loader


# ─────────────────────────────────────────────
# 3. Model Definition
# ─────────────────────────────────────────────
class FeedforwardNN(nn.Module):
    """
    Fully-connected feedforward network for MNIST digit classification.
    Architecture: 784 → [hidden layers with BN + ReLU + Dropout] → 10
    """

    def __init__(self, input_dim: int, hidden_sizes: list, num_classes: int, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten 28×28 → 784
        return self.net(x)


def build_model() -> FeedforwardNN:
    return FeedforwardNN(
        input_dim=28 * 28,
        hidden_sizes=CONFIG["hidden_sizes"],
        num_classes=10,
        dropout=CONFIG["dropout_rate"],
    ).to(DEVICE)


# ─────────────────────────────────────────────
# 4. Training & Evaluation
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


# ─────────────────────────────────────────────
# 5. ONNX Export
# ─────────────────────────────────────────────
def export_onnx(model: FeedforwardNN, path: str = "best_model.onnx"):
    """Export trained model to ONNX so the Flask server can load it."""
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28, device=DEVICE)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )
    print(f"ONNX model exported → {path}")


# ─────────────────────────────────────────────
# 6. Visualisation helpers
# ─────────────────────────────────────────────
def plot_learning_curves(history: dict, save_path: str = "learning_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MNIST Feedforward NN – Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train")
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved learning curves → {save_path}")
    plt.close()


@torch.no_grad()
def plot_sample_predictions(model, loader, num_samples: int = 16, save_path: str = "predictions.png"):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:num_samples].to(DEVICE), labels[:num_samples]
    outputs = model(images)
    preds = outputs.argmax(dim=1).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=12, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap="gray")
        color = "green" if preds[i] == labels[i] else "red"
        ax.set_title(f"pred={preds[i].item()}  true={labels[i].item()}", color=color, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved sample predictions → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 7. Main Pipeline
# ─────────────────────────────────────────────
def main():
    # Data
    train_loader, test_loader = get_dataloaders(CONFIG["batch_size"])

    # Model
    model = build_model()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel architecture:\n{model}")
    print(f"Trainable parameters: {total_params:,}\n")

    # Loss, optimiser, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = "best_model.pt"

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>8.2f}%  {va_loss:>8.4f}  {va_acc:>6.2f}%  {elapsed:>5.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    _, final_acc = evaluate(model, test_loader, criterion)
    print(f"Final test accuracy (best model): {final_acc:.2f}%")

    # Export to ONNX for the Flask UI server
    export_onnx(model, path="best_model.onnx")

    # Plots
    plot_learning_curves(history, save_path="learning_curves.png")
    plot_sample_predictions(model, test_loader, save_path="predictions.png")


if __name__ == "__main__":
    main()
