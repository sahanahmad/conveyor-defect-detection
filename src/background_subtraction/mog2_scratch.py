"""
Day 3 — Component 2 scratch: Background Subtraction (MOG2)

Purpose: run MOG2 over the Component 1 synthetic frame sequence and inspect
how the foreground mask behaves over time. This is exploratory/diagnostic —
not yet wired into Component 3 (that's Day 4).

Update (this version): bounding-box measurement is now computed on a
thresholded mask (255 only), not the raw MOG2 output. The raw mask also
contains shadow pixels (value 127), which findContours treats as nonzero —
that was producing bogus bboxes (e.g. w=400,h=200) during warm-up frames
where white_pixels was 0. Confirmed via frame 0: white_pixels=0 but the
unthresholded bbox call still returned a huge box, which is only possible
if it was picking up shadow noise, not real foreground.
"""

import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRAMES_DIR = PROJECT_ROOT / "outputs" / "scratch_v0_frames"
MASK_OUT_DIR = PROJECT_ROOT / "outputs" / "mog2_masks"

MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)

frame_paths = sorted(FRAMES_DIR.glob("*.png"))

if not frame_paths:
    raise FileNotFoundError(
        f"No frames found in {FRAMES_DIR}. Did Component 1 run and write PNGs there?"
    )

bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Frames to save as mask images for visual inspection.
# Chosen to span: warm-up, mid-growth, and two points in the plateau region
# (based on the frame-count printout from the first run).
SAVE_FRAMES = {5, 20, 45, 71}

for i, path in enumerate(frame_paths):
    frame = cv2.imread(str(path))
    if frame is None:
        raise IOError(f"Failed to read frame: {path}")

    mask = bg_subtractor.apply(frame)
    white_pixels = int((mask == 255).sum())
    print(f"frame {i:02d}: {white_pixels} foreground pixels")

    # Threshold to pure foreground (255) before finding contours, so shadow
    # pixels (127) can't produce a bogus bounding box.
    _, solid_mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        print(f"  -> largest blob bbox: w={w}, h={h}")
    else:
        print("  -> no foreground blob")

    if i in SAVE_FRAMES:
        out_path = MASK_OUT_DIR / f"mask_frame_{i:02d}.png"
        cv2.imwrite(str(out_path), mask)