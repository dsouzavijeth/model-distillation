"""
Standard training script — used for both TEACHER and BASELINE STUDENT.

No distillation here: plain cross-entropy training with cosine LR schedule.
Run this twice:
  1. Train the teacher   (python train_standard.py --model resnet56 --out teacher.pth)
  2. Train the baseline  (python train_standard.py --model resnet20 --out student_baseline.pth)

Both runs use identical hyperparameters so the comparison is clean.

Designed to run well on a Colab free-tier T4:
  - ResNet-56 for 200 epochs: ~45 minutes
  - ResNet-20 for 200 epochs: ~20 minutes

Hyperparameters follow the standard CIFAR recipe:
  - SGD with momentum 0.9 and Nesterov
  - Weight decay 5e-4
  - Initial LR 0.1 (yes, that high — BN + skip connections handle it)
  - Cosine annealing to 0 over all epochs
  - 200 epochs is standard; you can go 100 for speed with ~1% accuracy loss
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import resnet20, resnet56, count_parameters
from data import get_cifar100_loaders


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Top-1 accuracy on the test set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> tuple:
    """Train one epoch. Returns (avg_loss, train_acc)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return running_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet20", "resnet56"], required=True,
                        help="Which model to train")
    parser.add_argument("--out", type=str, required=True,
                        help="Output checkpoint path (e.g. teacher.pth)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # ── Model ────────────────────────────────────────────────────────
    model_fn = {"resnet20": resnet20, "resnet56": resnet56}[args.model]
    model = model_fn(num_classes=100).to(device)
    print(f"\nModel:    {args.model}")
    print(f"Params:   {count_parameters(model):,}")

    # ── Data ─────────────────────────────────────────────────────────
    train_loader, test_loader = get_cifar100_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # ── Optimizer + Scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training Loop ────────────────────────────────────────────────
    best_acc = 0.0
    start_time = time.time()

    print(f"\n{'Epoch':>5} {'LR':>8} {'TrainLoss':>10} {'TrainAcc':>9} {'TestAcc':>8} {'Best':>7} {'Time':>7}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        scheduler.step()

        test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - t0

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model,
                "test_acc": test_acc,
                "epoch": epoch,
            }, args.out)

        flag = "*" if is_best else " "
        print(f"{epoch:>5} {optimizer.param_groups[0]['lr']:>8.5f} "
              f"{train_loss:>10.4f} {train_acc:>8.2f}% {test_acc:>7.2f}% "
              f"{best_acc:>6.2f}%{flag} {epoch_time:>6.1f}s")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {args.out}")


if __name__ == "__main__":
    main()
