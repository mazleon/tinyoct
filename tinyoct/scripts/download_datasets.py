#!/usr/bin/env python3
"""
Dataset download helper for TinyOCT.
Handles OCTMNIST (auto-download via medmnist).
OCT2017 and OCTID require manual download — instructions printed here.

Usage:
    python scripts/download_datasets.py
"""

import os
from pathlib import Path

DATA_DIR = Path("./data")
MEDMNIST_ROOT = DATA_DIR / "medmnist"
MEDMNIST_ROOT.mkdir(parents=True, exist_ok=True)


def download_octmnist():
    """Download OCTMNIST via medmnist (auto)."""
    try:
        import medmnist
        from medmnist import OCTMNIST
        import torchvision.transforms as T

        tf = T.ToTensor()
        print("\nDownloading OCTMNIST (224×224)...")
        for split in ["train", "val", "test"]:
            OCTMNIST(split=split, size=224, transform=tf, download=True,
                     root=str(MEDMNIST_ROOT))
        print("✓ OCTMNIST ready at ./data/medmnist/")
    except ImportError:
        print("medmnist not installed. Run: pip install medmnist")


def print_oct2017_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  OCT2017 — Manual download required                            ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Go to: https://www.kaggle.com/datasets/paultimothymooney/  ║
║            kermany2018                                          ║
║  2. Sign in with a free Kaggle account                         ║
║  3. Click Download (~2 GB zip)                                  ║
║  4. Unzip and place at:  ./data/OCT2017/                       ║
║     Expected structure:                                         ║
║       ./data/OCT2017/train/CNV/*.jpeg                           ║
║       ./data/OCT2017/train/DME/*.jpeg                           ║
║       ./data/OCT2017/train/DRUSEN/*.jpeg                        ║
║       ./data/OCT2017/train/NORMAL/*.jpeg                        ║
║       ./data/OCT2017/test/...                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_octid_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  OCTID — Manual download required                              ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Go to: https://dataverse.scholarsportal.info/dataverse/    ║
║            OCTID                                                ║
║  2. Click Access Dataset (no account needed)                    ║
║  3. Download zip files by category                             ║
║  4. Place at: ./data/OCTID/                                    ║
║     Expected structure:                                         ║
║       ./data/OCTID/NORMAL/*.jpg                                 ║
║       ./data/OCTID/DR/*.jpg                                     ║
║       ./data/OCTID/AMD/*.jpg                                    ║
╚══════════════════════════════════════════════════════════════════╝
""")


def verify_oct2017():
    """Check if OCT2017 is present and report class counts."""
    oct_path = DATA_DIR / "OCT2017"
    if not oct_path.exists():
        print("✗ OCT2017 not found at ./data/OCT2017/")
        return False
    print("\nOCT2017 class counts:")
    for split in ["train", "test"]:
        for cls in ["CNV", "DME", "DRUSEN", "NORMAL"]:
            d = oct_path / split / cls
            if d.exists():
                n = len(list(d.glob("*.jpeg")) + list(d.glob("*.jpg")))
                print(f"  {split}/{cls}: {n:,} images")
    return True


if __name__ == "__main__":
    print("TinyOCT — Dataset Setup")
    print("=" * 50)
    download_octmnist()
    print_oct2017_instructions()
    print_octid_instructions()
    verify_oct2017()
