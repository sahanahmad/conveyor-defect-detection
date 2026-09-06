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