"""
Build a demo GIF from Component 7's annotated output frames
(outputs/pipeline_demo_frames/*.png).

Uses PIL directly -- no video codec involved, sidestepping the OpenCV
VideoWriter codec issue noted earlier in this project (PNG frame sequences
were already the chosen output format for exactly this reason).

Run from the repo root, after python -m src.pipeline.integrate has produced
outputs/pipeline_demo_frames/:
    python scripts/make_demo_gif.py
"""
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = PROJECT_ROOT / "outputs" / "pipeline_demo_frames"
GIF_OUT = PROJECT_ROOT / "docs" / "demo.gif"

DURATION_MS = 80  # per-frame display time; 72 frames * 80ms ~= 5.8s per loop


def main():
    frame_paths = sorted(FRAMES_DIR.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in {FRAMES_DIR}. "
            f"Run 'python -m src.pipeline.integrate' first."
        )

    frames = [Image.open(p).convert("RGB") for p in frame_paths]

    GIF_OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GIF_OUT,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        optimize=True,
    )

    size_kb = GIF_OUT.stat().st_size / 1024
    print(f"Wrote {len(frames)}-frame GIF to {GIF_OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()