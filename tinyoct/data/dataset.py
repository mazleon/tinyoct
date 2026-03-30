"""
Dataset classes for TinyOCT.
  - OCT2017Dataset: Primary training / evaluation (Kermany 2018, Kaggle)
  - OCTIDDataset:   Cross-scanner OOD validation (Cirrus HD-OCT)
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# OCTID uses different class names — map overlapping ones
OCTID_CLASS_MAP = {
    "NORMAL": "NORMAL",
    "NOR":    "NORMAL",
    "DR":     "DME",       # approximate mapping for OOD experiment
    "AMD":    "CNV",       # approximate mapping
    "CSR":    "DRUSEN",    # approximate mapping
    "MH":     None,        # exclude Macular Hole (no equivalent class)
}


class OCT2017Dataset(Dataset):
    """
    OCT2017 (Kermany2018) retinal OCT dataset.

    Expected folder structure:
        {root}/
          train/
            CNV/      *.jpeg
            DME/      *.jpeg
            DRUSEN/   *.jpeg
            NORMAL/   *.jpeg
          test/
            CNV/      ...
            ...

    Download: https://www.kaggle.com/datasets/paultimothymooney/kermany2018
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"OCT2017 split directory not found: {split_dir}\n"
                f"Download from: https://www.kaggle.com/datasets/paultimothymooney/kermany2018"
            )
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                # Also check lowercase
                class_dir = split_dir / class_name.lower()
            if not class_dir.exists():
                continue
            label = CLASS_TO_IDX[class_name]
            for img_path in class_dir.glob("*.jpeg"):
                samples.append((str(img_path), label))
            for img_path in class_dir.glob("*.jpg"):
                samples.append((str(img_path), label))
            for img_path in class_dir.glob("*.png"):
                samples.append((str(img_path), label))
        if len(samples) == 0:
            raise RuntimeError(
                f"No images found in {split_dir}. "
                f"Check folder names match: {CLASS_NAMES}"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> dict:
        counts = {c: 0 for c in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts


class OCTIDDataset(Dataset):
    """
    OCTID cross-scanner OOD validation dataset.
    Captured on Cirrus HD-OCT (different manufacturer from OCT2017).

    Expected folder structure:
        {root}/
          NORMAL/   *.jpg
          DR/       *.jpg
          AMD/      *.jpg
          CSR/      *.jpg
          MH/       *.jpg   (excluded — no equivalent class in training)

    Download: https://dataverse.scholarsportal.info/dataverse/OCTID
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        exclude_mh: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.exclude_mh = exclude_mh
        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        if not self.root.exists():
            raise FileNotFoundError(
                f"OCTID directory not found: {self.root}\n"
                f"Download from: https://dataverse.scholarsportal.info/dataverse/OCTID"
            )
        for octid_class, mapped_class in OCTID_CLASS_MAP.items():
            if mapped_class is None:
                continue  # skip MH
            class_dir = self.root / octid_class
            if not class_dir.exists():
                continue
            label = CLASS_TO_IDX[mapped_class]
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in class_dir.glob(ext):
                    samples.append((str(img_path), label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Ensures equal class representation per batch for SupCon balanced sampling.
    Each batch will have exactly (batch_size // num_classes) samples per class.
    """

    def __init__(self, dataset: OCT2017Dataset, batch_size: int, num_classes: int = 4):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples_per_class = batch_size // num_classes

        # Group indices by class
        self.class_indices = {i: [] for i in range(num_classes)}
        for idx, (_, label) in enumerate(dataset.samples):
            self.class_indices[label].append(idx)

        self.num_batches = min(
            len(v) // self.samples_per_class
            for v in self.class_indices.values()
        )

    def __iter__(self):
        import random
        # Shuffle within each class
        shuffled = {k: list(v) for k, v in self.class_indices.items()}
        for v in shuffled.values():
            random.shuffle(v)

        for batch_idx in range(self.num_batches):
            batch = []
            for class_id in range(self.num_classes):
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(shuffled[class_id][start:end])
            random.shuffle(batch)
            yield batch   # batch_sampler must yield a list per batch, not flat ints

    def __len__(self):
        return self.num_batches
