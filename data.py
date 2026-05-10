"""
CIFAR-100 data loading.

CIFAR-100:
  - 60,000 color images, 32×32 pixels, 100 classes
  - 50,000 train / 10,000 test (500 train + 100 test per class)
  - Tiny compared to ImageNet but enough classes to make distillation
    meaningfully interesting (more classes → more dark knowledge)
  - Downloads automatically via torchvision (~170 MB)

Normalization stats below are computed on the CIFAR-100 training set.
Using the correct per-channel mean/std is important — it centers and
scales each input channel so the first conv layer sees well-conditioned
data, which speeds up convergence.

Augmentation recipe (standard for CIFAR):
  - Random crop with 4px padding (simulates small spatial shifts)
  - Random horizontal flip (mirror invariance)
  - To tensor + normalize

No augmentation on the test set — we evaluate on the true images.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


# Computed on CIFAR-100 train set (values are in [0,1] after ToTensor)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_cifar100_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test dataloaders for CIFAR-100.

    Args:
        data_root: where to download/cache the dataset
        batch_size: mini-batch size
        num_workers: parallel data loading workers (2-4 is plenty for CIFAR)
        augment: apply random crop + flip to the training set

    Returns:
        (train_loader, test_loader)
    """
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # cleaner BN stats
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_cifar100_loaders(batch_size=128)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    x, y = next(iter(train_loader))
    print(f"Batch image shape: {tuple(x.shape)}")
    print(f"Batch label shape: {tuple(y.shape)}")
    print(f"Image value range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Num classes: {len(train_loader.dataset.classes)}")
