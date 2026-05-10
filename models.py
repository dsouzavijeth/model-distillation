"""
CIFAR ResNet models for knowledge distillation.

Two models, both canonical CIFAR architectures from the original ResNet paper
(He et al., 2016, "Deep Residual Learning for Image Recognition"):

  TEACHER: ResNet-56   (~855K params, ~72-73% on CIFAR-100)
  STUDENT: ResNet-20   (~278K params, ~68-69% on CIFAR-100 baseline)

Why these specific architectures?
  - Both are standard benchmarks — you can compare your results against
    published numbers in dozens of distillation papers.
  - ~3× compression is modest enough that the student has real capacity
    to learn, but large enough that distillation's benefit is clearly visible.
  - ResNet-20 is small enough to train quickly but not so small it hits
    an architectural ceiling on CIFAR-100.

CIFAR ResNet architecture (different from ImageNet ResNet!):
  stem: 3×3 conv, 16 channels
  stage1: N blocks, 16 channels, 32×32 spatial
  stage2: N blocks, 32 channels, 16×16 spatial
  stage3: N blocks, 64 channels,  8×8 spatial
  global avg pool → linear(num_classes)

  Total layers = 6N + 2
    N=3 → ResNet-20
    N=9 → ResNet-56
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BasicBlock(nn.Module):
    """
    Standard ResNet basic block for CIFAR.

        x ──┬──[3x3 Conv → BN → ReLU]──[3x3 Conv → BN]──(+)── ReLU── out
            └──────(1x1 Conv if dims mismatch)──────────┘

    The skip connection is the heart of ResNet: it lets gradients flow
    backward through very deep networks without vanishing, and lets
    each block learn a *residual* (delta) rather than a full mapping.
    """
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CIFARResNet(nn.Module):
    """
    Configurable CIFAR ResNet.

    Forward modes:
        model(x)                       → logits [B, num_classes]
        model(x, return_features=True) → (logits, [feat1, feat2, feat3])

    The three feature maps come from the end of each stage. They're what
    feature-based distillation uses to align teacher & student internally.
    """

    def __init__(self, num_blocks_per_stage: int, num_classes: int = 100):
        super().__init__()
        self.num_blocks_per_stage = num_blocks_per_stage

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(16, 16, num_blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(16, 32, num_blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(32, 64, num_blocks_per_stage, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        # Channel dims of each stage's output — used by feature distillation
        self.feature_channels: List[int] = [16, 32, 64]

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, n_blocks, stride):
        blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            blocks.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        f1 = self.stage1(x)   # [B, 16, 32, 32]
        f2 = self.stage2(f1)  # [B, 32, 16, 16]
        f3 = self.stage3(f2)  # [B, 64,  8,  8]
        pooled = self.pool(f3).flatten(1)
        logits = self.fc(pooled)

        if return_features:
            return logits, [f1, f2, f3]
        return logits


def resnet20(num_classes: int = 100) -> CIFARResNet:
    """ResNet-20: 3 blocks per stage, ~278K params. The STUDENT."""
    return CIFARResNet(num_blocks_per_stage=3, num_classes=num_classes)


def resnet56(num_classes: int = 100) -> CIFARResNet:
    """ResNet-56: 9 blocks per stage, ~855K params. The TEACHER."""
    return CIFARResNet(num_blocks_per_stage=9, num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    teacher = resnet56(num_classes=100)
    student = resnet20(num_classes=100)
    x = torch.randn(4, 3, 32, 32)

    t_logits, t_feats = teacher(x, return_features=True)
    s_logits, s_feats = student(x, return_features=True)

    print("=" * 60)
    print("MODEL CHECK")
    print("=" * 60)
    print(f"Teacher (ResNet-56) params:  {count_parameters(teacher):>10,}")
    print(f"Student (ResNet-20) params:  {count_parameters(student):>10,}")
    print(f"Compression ratio:           {count_parameters(teacher)/count_parameters(student):>10.2f}x")
    print()
    print(f"Teacher logits shape:        {tuple(t_logits.shape)}")
    print(f"Student logits shape:        {tuple(s_logits.shape)}")
    print()
    print("Feature map shapes:")
    for i, (tf, sf) in enumerate(zip(t_feats, s_feats)):
        print(f"  Stage {i+1}: teacher {tuple(tf.shape)}  student {tuple(sf.shape)}")
    print("=" * 60)
