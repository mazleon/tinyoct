"""
MedMNIST Dataset wrapper for TinyOCT.

Wraps the OCTMNIST 224x224 dataset from the medmnist library.
  - 4 classes: Choroidal Neovascularization (0), Diabetic Macular Edema (1),
               Drusen (2), Normal (3)
  - Same label order as OCT2017 -- no remapping needed.

Reference:
  Yang et al., MedMNIST v2 -- A Large-Scale Lightweight Benchmark for 2D and 3D
  Biomedical Image Classification, Scientific Data 2023.
  https://medmnist.com
"""
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset


# OCTMNIST class index -> our CLASS_NAMES order
# medmnist OCTMNIST: {0: CNV, 1: DME, 2: DRUSEN, 3: NORMAL}
OCTMNIST_CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
OCTMNIST_CLASS_TO_IDX = {c: i for i, c in enumerate(OCTMNIST_CLASS_NAMES)}


class OCTMNISTDataset(Dataset):
    """
    Wrapper around medmnist.OCTMNIST (224x224).

    Delegates all data access to the medmnist library which handles
    npz loading internally. This avoids direct numpy npz parsing issues.

    Expected file location: {root}/octmnist_224.npz
    Download: python scripts/download_datasets.py

    Args:
        root:      Path to directory containing octmnist_224.npz
        split:     'train' | 'val' | 'test'
        transform: torchvision transform pipeline (applied on PIL images)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root      = Path(root)
        self.split     = split
        self.transform = transform
        self._use_medmnist = False

        npz_path = self.root / "octmnist_224.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"OCTMNIST .npz not found: {npz_path}\n"
                f"Download it with: python scripts/download_datasets.py\n"
                f"Or manually from: https://zenodo.org/records/10519652/files/octmnist_224.npz"
            )

        try:
            from medmnist import OCTMNIST
            # medmnist handles npz internally; we apply our own transform in __getitem__
            self._ds = OCTMNIST(
                split=split,
                size=224,
                transform=None,      # applied ourselves below
                download=False,
                root=str(self.root),
            )
            self._use_medmnist = True
        except Exception as e:
            # Graceful fallback: direct numpy read (may fail on Zip64 large files)
            print(f"[OCTMNISTDataset] medmnist load failed ({e}), falling back to numpy.")
            self._load_numpy(npz_path, split)

    def _load_numpy(self, npz_path: Path, split: str):
        """Direct numpy .npz read fallback."""
        import numpy as np
        data = np.load(str(npz_path), allow_pickle=True)
        img_key   = f"{split}_images"
        label_key = f"{split}_labels"
        self._images = data[img_key]                              # [N, 224, 224, 3] uint8
        self._labels = data[label_key].squeeze(-1).astype("int64")  # [N]

    def __len__(self) -> int:
        if self._use_medmnist:
            return len(self._ds)
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self._use_medmnist:
            # medmnist returns (PIL Image, ndarray label) when transform=None
            img, label = self._ds[idx]
            label = int(label.squeeze())
        else:
            import numpy as np
            from PIL import Image
            img   = Image.fromarray(self._images[idx].astype("uint8"))
            label = int(self._labels[idx])

        if hasattr(img, 'convert'):
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def class_counts(self) -> dict:
        """Return per-class sample count dict."""
        import numpy as np
        counts = {c: 0 for c in OCTMNIST_CLASS_NAMES}
        if self._use_medmnist:
            labels = self._ds.labels.squeeze()
        else:
            labels = self._labels
        for lbl in labels:
            counts[OCTMNIST_CLASS_NAMES[int(lbl)]] += 1
        return counts

    def __repr__(self) -> str:
        return (
            f"OCTMNISTDataset(split={self.split}, n={len(self)}, "
            f"backend={'medmnist' if self._use_medmnist else 'numpy'})"
        )
