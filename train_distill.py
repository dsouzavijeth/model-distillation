"""
Knowledge Distillation Training.

Trains a student model using a frozen pretrained teacher.

Usage:
    python train_distill.py \
        --teacher-ckpt teacher.pth \
        --out student_distilled.pth \
        --temperature 4.0 --alpha 0.7 --beta 100.0

The three KD hyperparameters:
    temperature (T): how soft the teacher probabilities are.
                     T=1 → normal softmax (nearly one-hot)
                     T=4 → recommended default
                     T=10+ → very soft, may lose useful info

    alpha (α):       balance between CE loss and soft-label KD loss.
                     α=0.0 → pure CE (no KD from logits)
                     α=0.5 → equal weighting
                     α=0.9 → mostly KD (often works best when teacher is good)

    beta (β):        weight on feature distillation.
                     β=0    → no feature matching
                     β=100  → strong feature matching (our default)
                     Big number because feat MSE after L2 norm is tiny (~0.01).

Pro tip: start with (T=4, α=0.7, β=0) and confirm it beats baseline.
Then add feature KD (β=100) for extra gains.
"""

import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import resnet20, resnet56, count_parameters
from data import get_cifar100_loaders
from losses import DistillationLoss


def load_teacher(ckpt_path: str, device: str) -> nn.Module:
    """Load a trained teacher checkpoint and freeze it."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "resnet56")
    model_fn = {"resnet20": resnet20, "resnet56": resnet56}[model_name]
    teacher = model_fn(num_classes=100).to(device)
    teacher.load_state_dict(ckpt["model_state"])

    # FREEZE: no grads, eval mode (so BN uses running stats)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Loaded teacher: {model_name}")
    print(f"  Checkpoint accuracy: {ckpt.get('test_acc', 'N/A'):.2f}%")
    print(f"  Parameters: {count_parameters(teacher):,}")
    return teacher


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_one_epoch_distill(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: DistillationLoss,
    device: str,
) -> dict:
    """
    One epoch of distillation training.

    Per batch:
        1. Forward student  → (logits, features) with grads
        2. Forward teacher  → (logits, features) no grads, eval mode
        3. Combined loss    = CE + α·KD + β·feat
        4. Backprop through student only
    """
    student.train()
    teacher.eval()  # belt and suspenders

    agg = {"total": 0.0, "ce": 0.0, "kd": 0.0, "feat": 0.0}
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Student forward (with features for feature-KD)
        s_logits, s_feats = student(x, return_features=True)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_logits, t_feats = teacher(x, return_features=True)

        # Combined loss
        loss, parts = loss_fn(
            student_logits=s_logits,
            teacher_logits=t_logits,
            labels=y,
            student_features=s_feats,
            teacher_features=t_feats,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        for k in agg:
            agg[k] += parts[k] * y.size(0)
        correct += (s_logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    for k in agg:
        agg[k] /= total
    agg["train_acc"] = 100.0 * correct / total
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-json", type=str, default=None,
                        help="Optional path to dump per-epoch metrics as JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # ── Models ───────────────────────────────────────────────────────
    teacher = load_teacher(args.teacher_ckpt, device)

    student = resnet20(num_classes=100).to(device)
    print(f"\nStudent: resnet20")
    print(f"  Parameters: {count_parameters(student):,}")

    # ── Data ─────────────────────────────────────────────────────────
    train_loader, test_loader = get_cifar100_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
    )

    # ── Loss ─────────────────────────────────────────────────────────
    loss_fn = DistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        teacher_channels=teacher.feature_channels,
        student_channels=student.feature_channels,
    ).to(device)

    # Include adaptation-layer parameters in the optimizer.
    # (If teacher and student channels match, these are nn.Identity
    # and contribute no params — but we still sweep them in safely.)
    trainable_params = list(student.parameters()) + list(loss_fn.parameters())

    optimizer = optim.SGD(
        trainable_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nDistillation config:")
    print(f"  Temperature T: {args.temperature}")
    print(f"  Alpha α:       {args.alpha}")
    print(f"  Beta β:        {args.beta}")
    print(f"  Epochs:        {args.epochs}")

    # ── Training Loop ────────────────────────────────────────────────
    best_acc = 0.0
    start_time = time.time()
    log = []

    print(f"\n{'Ep':>4} {'LR':>7} {'Total':>7} {'CE':>6} {'KD':>6} {'Feat':>7} "
          f"{'Train':>7} {'Test':>7} {'Best':>7} {'T':>6}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        metrics = train_one_epoch_distill(
            student, teacher, train_loader, optimizer, loss_fn, device
        )
        scheduler.step()

        test_acc = evaluate(student, test_loader, device)
        epoch_time = time.time() - t0

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            torch.save({
                "model_state": student.state_dict(),
                "model_name": "resnet20",
                "test_acc": test_acc,
                "epoch": epoch,
                "kd_config": {
                    "temperature": args.temperature,
                    "alpha": args.alpha,
                    "beta": args.beta,
                },
            }, args.out)

        flag = "*" if is_best else " "
        print(f"{epoch:>4} {optimizer.param_groups[0]['lr']:>7.4f} "
              f"{metrics['total']:>7.3f} {metrics['ce']:>6.3f} "
              f"{metrics['kd']:>6.3f} {metrics['feat']:>7.4f} "
              f"{metrics['train_acc']:>6.2f}% {test_acc:>6.2f}% "
              f"{best_acc:>6.2f}%{flag} {epoch_time:>5.1f}s")

        log.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr'],
            **metrics,
            "test_acc": test_acc,
            "best_acc": best_acc,
        })

    total_time = time.time() - start_time
    print("-" * 80)
    print(f"Distillation complete in {total_time/60:.1f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {args.out}")

    if args.log_json:
        with open(args.log_json, "w") as f:
            json.dump({
                "config": vars(args),
                "best_acc": best_acc,
                "epochs": log,
            }, f, indent=2)
        print(f"Training log saved to: {args.log_json}")


if __name__ == "__main__":
    main()
