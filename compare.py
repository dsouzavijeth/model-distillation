"""
Compare teacher, baseline student, and distilled student.

Usage:
    python compare.py \
        --teacher teacher.pth \
        --baseline student_baseline.pth \
        --distilled student_distilled.pth

Reports:
  - Top-1 and top-5 accuracy on the CIFAR-100 test set
  - Parameter count
  - Model size (MB)
  - Inference latency (ms/image on current device)
  - Gap closure: how much of the teacher-student gap the distillation closed
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import resnet20, resnet56, count_parameters
from data import get_cifar100_loaders


def load_checkpoint(path: str, device: str) -> tuple:
    ckpt = torch.load(path, map_location=device)
    model_name = ckpt.get("model_name", "resnet20")
    model_fn = {"resnet20": resnet20, "resnet56": resnet56}[model_name]
    model = model_fn(num_classes=100).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_topk(
    model: nn.Module, loader: DataLoader, device: str, k: tuple = (1, 5)
) -> dict:
    """Compute top-1 and top-5 accuracy."""
    model.eval()
    max_k = max(k)
    correct = {kk: 0 for kk in k}
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # top-k predictions: [B, max_k]
        _, topk_pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        topk_pred = topk_pred.t()  # [max_k, B]
        match = topk_pred.eq(y.view(1, -1).expand_as(topk_pred))  # [max_k, B]

        for kk in k:
            correct[kk] += match[:kk].any(dim=0).sum().item()
        total += y.size(0)

    return {f"top{kk}": 100.0 * correct[kk] / total for kk in k}


def measure_latency(
    model: nn.Module, device: str, num_warmup: int = 20, num_runs: int = 100
) -> float:
    """Average inference time per image (ms) with batch size 1."""
    model.eval()
    dummy = torch.randn(1, 3, 32, 32, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs * 1000
    return elapsed


def model_size_mb(model: nn.Module) -> float:
    """Approximate size in MB assuming float32 weights."""
    n_params = count_parameters(model)
    return n_params * 4 / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--distilled", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    _, test_loader = get_cifar100_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=2,
        augment=False,
    )

    results = {}
    for name, path in [
        ("teacher", args.teacher),
        ("baseline", args.baseline),
        ("distilled", args.distilled),
    ]:
        print(f"\nEvaluating {name} from {path}...")
        model, ckpt = load_checkpoint(path, device)
        acc = evaluate_topk(model, test_loader, device)
        latency = measure_latency(model, device)

        results[name] = {
            "top1": acc["top1"],
            "top5": acc["top5"],
            "params": count_parameters(model),
            "size_mb": model_size_mb(model),
            "latency_ms": latency,
            "model_name": ckpt.get("model_name", "?"),
        }

    # ── Print comparison table ───────────────────────────────────────
    print("\n" + "=" * 76)
    print("COMPARISON: Teacher vs Baseline Student vs Distilled Student")
    print("=" * 76)
    header = f"{'Model':<20} {'Arch':<10} {'Top-1':>8} {'Top-5':>8} {'Params':>10} {'Size':>8} {'Latency':>10}"
    print(header)
    print("-" * 76)
    for name in ["teacher", "baseline", "distilled"]:
        r = results[name]
        print(
            f"{name:<20} {r['model_name']:<10} "
            f"{r['top1']:>7.2f}% {r['top5']:>7.2f}% "
            f"{r['params']:>10,} {r['size_mb']:>6.2f}MB "
            f"{r['latency_ms']:>8.2f}ms"
        )
    print("=" * 76)

    # ── Gap analysis ─────────────────────────────────────────────────
    t = results["teacher"]["top1"]
    b = results["baseline"]["top1"]
    d = results["distilled"]["top1"]
    gap = t - b
    improvement = d - b
    gap_closed = (improvement / gap * 100) if gap > 0 else 0.0

    print("\nGAP ANALYSIS (Top-1):")
    print(f"  Teacher accuracy:            {t:.2f}%")
    print(f"  Baseline student accuracy:   {b:.2f}%")
    print(f"  Distilled student accuracy:  {d:.2f}%")
    print(f"  Teacher-student gap:         {gap:.2f}%")
    print(f"  Distillation improvement:    {improvement:+.2f}%")
    print(f"  Gap closed:                  {gap_closed:.1f}%")

    print("\nCOMPRESSION (distilled vs teacher):")
    tp = results["teacher"]["params"]
    dp = results["distilled"]["params"]
    print(f"  Parameters: {tp/dp:.2f}x smaller")
    print(f"  Size:       {results['teacher']['size_mb']/results['distilled']['size_mb']:.2f}x smaller")
    print(f"  Latency:    {results['teacher']['latency_ms']/results['distilled']['latency_ms']:.2f}x faster")


if __name__ == "__main__":
    main()
