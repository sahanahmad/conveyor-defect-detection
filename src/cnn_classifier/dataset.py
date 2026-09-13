import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def label_from_path(p: Path) -> str:
    m = re.match(r'^(.+)_\d+$', p.stem)
    return m.group(1) if m else p.stem


def list_files_and_labels(root: Path):
    files = sorted(root.glob('*.jpg'))
    labels = [label_from_path(f) for f in files]
    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return files, labels, classes, class_to_idx


class NEUDataset(Dataset):
    """
    Loads NEU surface-defect images as single-channel grayscale.
    `indices` selects a subset of `files` (e.g. the train/val/test split).
    `mean`/`std` are optional per-pixel normalization stats, in [0, 1] scale,
    computed on the TRAIN split only and passed in for val/test to avoid leakage.
    `augment=True` applies random horizontal flip, vertical flip, and small
    rotation — intended for the TRAIN split only. Never set this True for
    val/test: those must reflect real, un-augmented data.
    """

    def __init__(self, files, labels, class_to_idx, indices, mean=None, std=None, augment=False):
        self.files = [files[i] for i in indices]
        self.labels = [labels[i] for i in indices]
        self.class_to_idx = class_to_idx
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _apply_augmentation(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        return img

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('L')  # single-channel grayscale

        if self.augment:
            img = self._apply_augmentation(img)

        arr = np.array(img, dtype=np.float32) / 255.0   # scale to [0, 1]

        if self.mean is not None and self.std is not None:
            arr = (arr - self.mean) / self.std

        arr = arr[np.newaxis, :, :]  # add channel dim: (1, H, W)
        label = self.class_to_idx[self.labels[idx]]
        return arr, label