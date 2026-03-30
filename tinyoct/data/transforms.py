"""
Augmentation pipelines for TinyOCT v3.
Clinical constraint: only use anatomically-safe augmentations.
  - Horizontal flip: OK (OCT B-scans are symmetric left-right)
  - Brightness/contrast: OK (scanner intensity variation)
  - Rotation > ±10°: NOT OK (retinal layers must remain horizontal)
  - Vertical flip: NOT OK (inverts retinal anatomy)
  - Aggressive colour jitter: NOT OK (OCT is near-grayscale)
"""

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import random


def get_train_transforms(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size + 16, image_size + 16)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        T.RandomApply([SmallRotation(max_angle=5)], p=0.3),  # ±5° only
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def get_tta_transforms(image_size: int = 224):
    """Test-Time Augmentation: horizontal flip + mild brightness.
    Only clinically safe transforms. Returns a list of transform pipelines.
    """
    base = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    augmented = [
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    return [T.Compose(base), T.Compose(augmented)]


class SmallRotation:
    """Rotate by a small random angle — models patient head-tilt during OCT acquisition."""
    def __init__(self, max_angle: float = 5.0):
        self.max_angle = max_angle

    def __call__(self, img):
        angle = random.uniform(-self.max_angle, self.max_angle)
        return TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
