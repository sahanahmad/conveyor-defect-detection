import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from src.cnn_classifier.dataset import NEUDataset
from src.cnn_classifier.model import DefectCNN

# --- Reproduce the same split as scratch_v0.py ---

root = Path('data/neu/IMAGES')
files = sorted(root.glob('*.jpg'))


def label_from_path(p: Path) -> str:
    m = re.match(r'^(.+)_\d+$', p.stem)
    return m.group(1) if m else p.stem


labels = [label_from_path(f) for f in files]
classes = sorted(set(labels))
class_to_idx = {c: i for i, c in enumerate(classes)}

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
    train_idx += idxs[:n_train]
    val_idx += idxs[n_train:n_train + n_val]
    test_idx += idxs[n_train + n_val:]

# --- Train-only normalization stats ---

train_files_subset = [files[i] for i in train_idx]
pixel_sum, pixel_sq_sum, pixel_count = 0.0, 0.0, 0
for f in train_files_subset:
    arr = np.array(Image.open(f).convert('L'), dtype=np.float32) / 255.0
    pixel_sum += arr.sum()
    pixel_sq_sum += (arr ** 2).sum()
    pixel_count += arr.size

mean = pixel_sum / pixel_count
std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

# --- Datasets and loaders ---
# augment=True on train only — val must reflect real, un-augmented data

train_ds = NEUDataset(files, labels, class_to_idx, train_idx, mean=mean, std=std, augment=True)
val_ds = NEUDataset(files, labels, class_to_idx, val_idx, mean=mean, std=std, augment=False)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

# --- Device ---

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('using device:', device)

# --- Model, loss, optimizer ---

model = DefectCNN(num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training loop ---

NUM_EPOCHS = 20

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        train_correct += (logits.argmax(dim=1) == y).sum().item()
        train_total += x.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            val_loss += loss.item() * x.size(0)
            val_correct += (logits.argmax(dim=1) == y).sum().item()
            val_total += x.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f'epoch {epoch:2d} | train loss {train_loss:.4f} acc {train_acc:.4f} '
          f'| val loss {val_loss:.4f} acc {val_acc:.4f}')

# --- Save checkpoint ---

checkpoint_path = Path('checkpoints/cnn_classifier_v1.pt')
checkpoint_path.parent.mkdir(exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': class_to_idx,
    'train_mean': mean,
    'train_std': std,
}, checkpoint_path)
print('saved checkpoint to', checkpoint_path)