import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.cnn_classifier.dataset import NEUDataset
from src.cnn_classifier.model import DefectCNN

# --- Step 1: list files and parse labels ---

root = Path('data/neu/IMAGES')
files = sorted(root.glob('*.jpg'))


def label_from_path(p: Path) -> str:
    m = re.match(r'^(.+)_\d+$', p.stem)
    return m.group(1) if m else p.stem


labels = [label_from_path(f) for f in files]
classes = sorted(set(labels))
class_to_idx = {c: i for i, c in enumerate(classes)}

print('classes:', classes)
print('class_to_idx:', class_to_idx)
print('counts:', Counter(labels))
print('first 3 (path, label, idx):')
for f, l in list(zip(files, labels))[:3]:
    print(' ', f.name, l, class_to_idx[l])

print()

# --- Step 2: stratified train/val/test split ---

SEED = 42
random.seed(SEED)

by_class = {c: [] for c in classes}
for i, l in enumerate(labels):
    by_class[l].append(i)

train_idx, val_idx, test_idx = [], [], []

for c, idxs in by_class.items():
    idxs = idxs.copy()
    random.shuffle(idxs)
    n = len(idxs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    # remainder goes to test, so rounding doesn't silently drop a sample
    train_idx += idxs[:n_train]
    val_idx += idxs[n_train:n_train + n_val]
    test_idx += idxs[n_train + n_val:]

print('total:', len(files))
print('train:', len(train_idx), 'val:', len(val_idx), 'test:', len(test_idx))
print('sum check:', len(train_idx) + len(val_idx) + len(test_idx) == len(files))

print()
print('per-class breakdown (train/val/test):')
for c in classes:
    idxs = by_class[c]
    t = sum(1 for i in idxs if i in set(train_idx))
    v = sum(1 for i in idxs if i in set(val_idx))
    te = sum(1 for i in idxs if i in set(test_idx))
    print(f'  {c}: {t}/{v}/{te}  (total {len(idxs)})')

print()

# --- Step 3: train-only normalization stats + Dataset sanity check ---

train_files_subset = [files[i] for i in train_idx]
pixel_sum, pixel_sq_sum, pixel_count = 0.0, 0.0, 0
for f in train_files_subset:
    arr = np.array(Image.open(f).convert('L'), dtype=np.float32) / 255.0
    pixel_sum += arr.sum()
    pixel_sq_sum += (arr ** 2).sum()
    pixel_count += arr.size

mean = pixel_sum / pixel_count
std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
print('train mean:', mean, 'train std:', std)

train_ds = NEUDataset(files, labels, class_to_idx, train_idx, mean=mean, std=std)
val_ds = NEUDataset(files, labels, class_to_idx, val_idx, mean=mean, std=std)

print('train_ds length:', len(train_ds))
sample_arr, sample_label = train_ds[0]
print('sample shape:', sample_arr.shape, 'dtype:', sample_arr.dtype)
print('sample label idx:', sample_label)
print('sample min/max after normalization:', sample_arr.min(), sample_arr.max())

print()

# --- Step 4: model shape sanity check ---

model = DefectCNN(num_classes=6)
dummy_batch = torch.randn(4, 1, 200, 200)  # batch of 4, matches Dataset output shape
output = model(dummy_batch)
print('output shape:', output.shape)  # expect (4, 6)
print('output dtype:', output.dtype)
print('num params:', sum(p.numel() for p in model.parameters()))