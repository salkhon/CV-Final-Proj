"""SVHN 10 digit classes + one non-digit (background) class for patch rejection."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# 11th class: proposed regions that are not digits
NEGATIVE_CLASS = 10
NUM_CLASSES = 11


class SVHN11Dataset(Dataset):
    """
    Wraps a digit dataset (e.g. combined SVHN train+extra) and appends
    synthetic non-digit patches (class NEGATIVE_CLASS) for training.
    """

    def __init__(
        self,
        base: Dataset,
        include_negatives: bool,
        neg_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.base = base
        self.include_negatives = include_negatives
        self.neg_ratio = max(0.0, float(neg_ratio))
        self._rng = random.Random(seed)
        self._neg_count = int(len(base) * self.neg_ratio) if include_negatives else 0

    def __len__(self) -> int:
        return len(self.base) + self._neg_count

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        n_digit = len(self.base)
        if index < n_digit:
            img, y = self.base[index]
            if isinstance(y, torch.Tensor):
                y = int(y.item())
            return img, y
        g = torch.Generator()
        g.manual_seed(self._rng.randint(0, 2**31 - 1))
        noise = torch.rand(3, 32, 32, generator=g) * 0.5 + 0.25
        noise = noise + 0.08 * torch.randn(3, 32, 32, generator=g)
        noise = torch.clamp(noise, 0.0, 1.0)
        return noise, NEGATIVE_CLASS


def _default_transforms_train() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=15, fill=0.0),
            transforms.ColorJitter(brightness=0.4, contrast=0.3),
            transforms.Lambda(
                lambda x: torch.clamp(
                    x + 0.05 * torch.randn_like(x), 0.0, 1.0
                )
            ),
        ]
    )


def _default_transforms_eval() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


class _CombinedSVHN(Dataset):
    """Indexes into train then extra split of SVHN."""

    def __init__(
        self,
        train: datasets.SVHN,
        extra: datasets.SVHN,
        indices: List[int],
    ) -> None:
        self.train = train
        self.extra = extra
        self.n_train = len(train)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        j = self.indices[i]
        if j < self.n_train:
            img, y = self.train[j]
        else:
            img, y = self.extra[j - self.n_train]
        if isinstance(y, torch.Tensor):
            y = int(y.item())
        return img, y


def get_svhn_11_datasets(
    data_dir: str,
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
    neg_ratio: float = 0.2,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[SVHN11Dataset, SVHN11Dataset, SVHN11Dataset]:
    """
    Build train, val, and test SVHN-11 datasets. Val/test have no synthetic negatives.
    """
    tr_tf = train_transform or _default_transforms_train()
    te_tf = test_transform or _default_transforms_eval()

    train_raw = datasets.SVHN(
        data_dir, split="train", download=True, transform=tr_tf
    )
    extra_raw = datasets.SVHN(
        data_dir, split="extra", download=True, transform=tr_tf
    )
    test_raw = datasets.SVHN(
        data_dir, split="test", download=True, transform=te_tf
    )

    n_train = len(train_raw)
    n_extra = len(extra_raw)
    indices_all = list(range(n_train + n_extra))
    rng = random.Random(seed)
    rng.shuffle(indices_all)
    n_val = int(len(indices_all) * val_fraction)
    val_set_idx = set(indices_all[:n_val])
    tr_idx = [i for i in indices_all if i not in val_set_idx]
    va_idx = list(val_set_idx)

    train_combined = _CombinedSVHN(train_raw, extra_raw, tr_idx)
    val_combined = _CombinedSVHN(train_raw, extra_raw, va_idx)

    train_ds = SVHN11Dataset(
        train_combined, include_negatives=True, neg_ratio=neg_ratio, seed=seed
    )
    val_ds = SVHN11Dataset(
        val_combined, include_negatives=False, neg_ratio=0.0, seed=seed
    )
    test_ds = SVHN11Dataset(
        test_raw, include_negatives=False, neg_ratio=0.0, seed=seed
    )
    return train_ds, val_ds, test_ds


def get_svhn_11_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 0,
    neg_ratio: float = 0.2,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tr, va, te = get_svhn_11_datasets(
        data_dir,
        neg_ratio=neg_ratio,
        val_fraction=val_fraction,
        seed=seed,
    )
    return (
        DataLoader(
            tr, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        DataLoader(
            va, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        DataLoader(
            te, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    )
