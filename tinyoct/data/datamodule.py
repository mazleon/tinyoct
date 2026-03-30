"""
DataModule: wraps all dataset creation, splitting, and DataLoader setup.
Supports three dataset modes:
  - 'oct2017'  : Kermany 2018 (primary training set, ~83K images)
  - 'octmnist' : MedMNIST OCTMNIST 224×224 (~97K images from .npz)
  - 'octid'    : OCTID cross-scanner OOD validation
"""

from typing import Optional
from torch.utils.data import DataLoader, Dataset

from .dataset import OCT2017Dataset, OCTIDDataset, BalancedBatchSampler
from .medmnist_dataset import OCTMNISTDataset
from .transforms import get_train_transforms, get_val_transforms


class OCTDataModule:
    """
    Unified data module for all TinyOCT datasets.

    cfg.data.dataset selects the training source:
      - 'oct2017'   →  ./data/oct2017/  (folder hierarchy)
      - 'octmnist'  →  ./data/medmnist/octmnist_224.npz
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.train_ds: Optional[Dataset] = None
        self.val_ds:   Optional[Dataset] = None
        self.test_ds:  Optional[Dataset] = None
        self.ood_ds:   Optional[Dataset] = None

        # Which dataset to use (default: oct2017 for backwards compat)
        self.dataset_name = getattr(cfg.data, "dataset", "oct2017").lower()

    def setup(self, stage: Optional[str] = None):
        c = self.cfg.data
        train_tf = get_train_transforms(c.image_size)
        val_tf   = get_val_transforms(c.image_size)

        if self.dataset_name == "octmnist":
            self._setup_octmnist(c, train_tf, val_tf, stage)
        else:
            self._setup_oct2017(c, train_tf, val_tf, stage)

    # ------------------------------------------------------------------
    # OCT2017 (Kermany)
    # ------------------------------------------------------------------

    def _setup_oct2017(self, c, train_tf, val_tf, stage):
        if stage in ("fit", None):
            self.train_ds = OCT2017Dataset(
                root=c.oct2017_path, split="train", transform=train_tf
            )
            self.val_ds = OCT2017Dataset(
                root=c.oct2017_path, split="val", transform=val_tf
            )
        if stage in ("test", None):
            self.test_ds = OCT2017Dataset(
                root=c.oct2017_path, split="test", transform=val_tf
            )

    # ------------------------------------------------------------------
    # OCTMNIST (MedMNIST 224×224)
    # ------------------------------------------------------------------

    def _setup_octmnist(self, c, train_tf, val_tf, stage):
        root = c.octmnist_path
        if stage in ("fit", None):
            self.train_ds = OCTMNISTDataset(root=root, split="train", transform=train_tf)
            self.val_ds   = OCTMNISTDataset(root=root, split="val",   transform=val_tf)
        if stage in ("test", None):
            self.test_ds  = OCTMNISTDataset(root=root, split="test",  transform=val_tf)

    # ------------------------------------------------------------------
    # OOD (OCTID cross-scanner)
    # ------------------------------------------------------------------

    def setup_ood(self):
        """Load OCTID for cross-scanner OOD evaluation."""
        val_tf = get_val_transforms(self.cfg.data.image_size)
        self.ood_ds = OCTIDDataset(
            root=self.cfg.data.octid_path, transform=val_tf
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        c = self.cfg.train
        # Use BalancedBatchSampler only when SupCon loss is active
        if c.loss.supcon_weight > 0 and hasattr(self.train_ds, "samples"):
            # BalancedBatchSampler requires .samples attribute (OCT2017Dataset)
            sampler = BalancedBatchSampler(
                self.train_ds, batch_size=c.batch_size
            )
            return DataLoader(
                self.train_ds,
                batch_sampler=sampler,
                num_workers=self.cfg.data.num_workers,
                pin_memory=self.cfg.data.pin_memory,
            )
        # Standard loader (used for OCTMNIST which has no .samples)
        return DataLoader(
            self.train_ds,
            batch_size=c.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.eval.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.eval.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )

    def ood_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ood_ds,
            batch_size=self.cfg.eval.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )
