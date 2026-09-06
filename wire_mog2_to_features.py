import cv2
from pathlib import Path
from src.feature_extraction.morphological_features import extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[0]  # adjust if this file
                                                     # doesn't live at repo root
FRAMES_DIR = PROJECT_ROOT / "outputs" / "scratch_v0_frames"

frame_paths = sorted(FRAMES_DIR.glob("*.png"))

if not frame_paths:
    raise FileNotFoundError(
        f"No frames found in {FRAMES_DIR}. Did Component 1 run and write PNGs there?"
    )

bg_subtractor = cv2.createBackgroundSubtractorMOG2()

print(f"{'frame':>5} {'#blobs':>7} {'w':>4} {'h':>4} {'area':>9} "
      f"{'aspect':>7} {'extent':>7} {'solidity':>9}")

for i, path in enumerate(frame_paths):
    frame = cv2.imread(str(path))
    if frame is None:
        raise IOError(f"Failed to read frame: {path}")

    raw_mask = bg_subtractor.apply(frame)
    _, solid_mask = cv2.threshold(raw_mask, 254, 255, cv2.THRESH_BINARY)

    features = extract_features(solid_mask)

    if not features:
        print(f"{i:>5} {0:>7} {'--':>4} {'--':>4} {'--':>9} {'--':>7} {'--':>7} {'--':>9}")
        continue

    # Largest by area = our best guess at "the real defect" for this
    # summary view. This is a display heuristic only, NOT a filtering
    # decision baked into the pipeline (see memory: noise rejection is
    # deferred to Component 4's Kalman gating, not decided here).
    largest = max(features, key=lambda f: f["area"])

    print(f"{i:>5} {len(features):>7} {largest['w']:>4} {largest['h']:>4} "
          f"{largest['area']:>9.1f} {largest['aspect_ratio']:>7.3f} "
          f"{largest['extent']:>7.3f} {largest['solidity']:>9.3f}")