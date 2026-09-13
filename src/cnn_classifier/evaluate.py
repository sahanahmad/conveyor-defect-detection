import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.cnn_classifier.dataset import NEUDataset
from src.cnn_classifier.model import DefectCNN

# --- Reproduce the same split used in train.py (same seed, same logic) ---

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

# --- Load checkpoint ---

checkpoint_path = Path('checkpoints/cnn_classifier_v1.pt')
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Sanity check: catch a silent class-index mismatch before it corrupts every metric
saved_class_to_idx = checkpoint['class_to_idx']
assert saved_class_to_idx == class_to_idx, (
    f"class_to_idx mismatch! checkpoint has {saved_class_to_idx}, "
    f"recomputed {class_to_idx}. Do not proceed until this is resolved."
)
print('class_to_idx sanity check passed:', class_to_idx)

mean = checkpoint['train_mean']
std = checkpoint['train_std']

# --- Test dataset (augment=False — real, unaltered test data) ---

test_ds = NEUDataset(files, labels, class_to_idx, test_idx, mean=mean, std=std, augment=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# --- Load model ---

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = DefectCNN(num_classes=6).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Run inference on the test set — the ONE time this happens ---

all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# --- Overall accuracy ---

overall_acc = (all_preds == all_labels).mean()
print(f'\nTest set size: {len(all_labels)}')
print(f'Overall test accuracy: {overall_acc:.4f}')

# --- Confusion matrix (rows = true label, cols = predicted label) ---

num_classes = len(classes)
cm = np.zeros((num_classes, num_classes), dtype=int)
for true, pred in zip(all_labels, all_preds):
    cm[true, pred] += 1

# Short, fixed-width codes avoid the truncation/alignment bug from
# variable-length class names (e.g. 'rolled-in_scale' vs 'crazing')
codes = [f'C{i}' for i in range(num_classes)]

print('\nLegend:')
for code, c in zip(codes, classes):
    print(f'  {code} = {c}')

print('\nConfusion matrix (rows=true, cols=predicted):')
col_width = 6
header = ' ' * 6 + ''.join(f'{code:>{col_width}}' for code in codes)
print(header)
for i, code in enumerate(codes):
    row = f'{code:>6}' + ''.join(f'{cm[i, j]:>{col_width}}' for j in range(num_classes))
    print(row)

# --- Per-class precision, recall, F1 ---

print('\nPer-class metrics:')
print(f'{"class":>16} {"precision":>10} {"recall":>10} {"f1":>10} {"support":>8}')
for i, c in enumerate(classes):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    support = cm[i, :].sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f'{c:>16} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>8}')

# --- Save confusion matrix as a plot ---

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_yticklabels(classes)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix — Test Accuracy {overall_acc:.2%}')

# annotate each cell with its count
for i in range(num_classes):
    for j in range(num_classes):
        count = cm[i, j]
        color = 'white' if count > cm.max() / 2 else 'black'
        ax.text(j, i, str(count), ha='center', va='center', color=color)

fig.colorbar(im, ax=ax, label='count')
fig.tight_layout()

output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
plot_path = output_dir / 'confusion_matrix_v1.png'
fig.savefig(plot_path, dpi=150)
print(f'\nSaved confusion matrix plot to {plot_path}')