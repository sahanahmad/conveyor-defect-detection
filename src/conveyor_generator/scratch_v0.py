import cv2
import numpy as np
import os
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "neu" / "IMAGES"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scratch_v0_frames"
GT_PATH = PROJECT_ROOT / "outputs" / "ground_truth.csv"

belt = np.full((200, 400, 3), 90, dtype=np.uint8)

defect_class = "scratches"
defect_path = DATA_DIR / f"{defect_class}_1.jpg"
defect_raw = cv2.imread(str(defect_path))
if defect_raw is None:
    raise FileNotFoundError(f"Couldn't load {defect_path} — check the dataset is unzipped there.")
defect = cv2.resize(defect_raw, (40, 40))
h, w = defect.shape[:2]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ground_truth = []

frame_num = 0
for x in range(0, 360, 5):
    frame = belt.copy()
    frame[90:90+h, x:x+w] = defect
    cv2.imwrite(str(OUTPUT_DIR / f"frame_{frame_num:04d}.png"), frame)
    ground_truth.append([frame_num, x, 90, w, h, defect_class])
    frame_num += 1

with open(GT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_number", "x", "y", "width", "height", "defect_class"])
    writer.writerows(ground_truth)

print(f"Wrote {frame_num} frames to {OUTPUT_DIR}")
print(f"Wrote {len(ground_truth)} rows to {GT_PATH}")