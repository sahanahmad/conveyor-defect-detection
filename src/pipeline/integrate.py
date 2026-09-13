"""
Component 7: End-to-end pipeline integration.

Wires: MOG2 background subtraction -> morphological feature extraction ->
multi-track Kalman tracking (automatic track birth/death, gated by
Mahalanobis distance) -> CNN defect classification (once a track confirms,
re-run on every new largest-observed crop so the final label reflects the
most fully-formed view of the blob, not a partial MOG2 warm-up silhouette).

Run from the repo root:
    python -m src.pipeline.integrate
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.morphological_features import extract_features
from src.kalman_tracker.kalman_core import make_conveyor_kf
from src.kalman_tracker.track import Track
from src.cnn_classifier.model import DefectCNN

FRAMES_DIR = PROJECT_ROOT / "outputs" / "scratch_v0_frames"
DEMO_OUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_demo_frames"
LOG_OUT_PATH = PROJECT_ROOT / "outputs" / "pipeline_log.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_classifier_v1.pt"

# DefectCNN was trained on native 200x200 NEU crops. A live conveyor blob is
# ~40x40 (the synthetic generator's scratch crop size), so this is a real
# train/inference resolution mismatch -- documented, not silently absorbed.
CNN_INPUT_SIZE = 200

# Real scratch blob area ranges 123.0-1150.5 across all 72 frames (grows as
# MOG2 converges); every other contour ever observed is <=11.0 (measured via
# quick_area_diagnostic.py against the actual frame sequence -- not a guess).
# 20 sits in the ~112-unit gap between the two clusters with margin on both sides.
MIN_BLOB_AREA = 20


def blob_centroid(feature):
    """(x + w/2, y + h/2) -- matches the convention used to generate
    outputs/frame_5_39_centroids.txt, the data Component 4 was validated
    against, so live tracking stays consistent with that validation."""
    return (feature["x"] + feature["w"] / 2, feature["y"] + feature["h"] / 2)


def filter_blobs_by_area(features, min_area=MIN_BLOB_AREA):
    """Drop contours too small to be the real defect. Measured against all
    72 frames of the real pipeline (see quick_area_diagnostic.py): every
    noise contour observed is <=11.0, every real-blob observation is
    >=123.0 -- min_area=20 sits cleanly in that gap. Without this, MOG2's
    warm-up noise spawns dozens of spurious tracks (see Day 8 pipeline_log.csv)."""
    return [f for f in features if f["area"] >= min_area]


def is_new_best_crop(track_best_area, track_id, area):
    """
    True if `area` is the largest area seen for this track so far, and
    updates track_best_area in place accordingly.

    Classification is re-run whenever this returns True, so the final label
    always comes from the most complete view of the blob observed -- early
    frames, while MOG2 is still converging, only see a partial silhouette.
    Concretely: a real scratch blob was captured as a 5x33-pixel sliver at
    frame 8 (the moment its track first confirmed) and misclassified as
    'inclusion' at 97% confidence. Classifying at first confirmation instead
    of at peak-observed-size was the root cause; this fixes that by always
    preferring the largest crop seen so far.
    """
    if area > track_best_area.get(track_id, -1):
        track_best_area[track_id] = area
        return True
    return False


def match_and_advance_tracks(active_tracks, detections, chi2_threshold, frame_idx):
    """
    Advance every non-dead track by one frame against this frame's
    detections. Mutates each Track via its own .update() (predicts always,
    corrects on a gated match, per step.py). Kills are handled inside
    Track itself; this function does not remove dead tracks from the list.

    Returns:
        dict[track_id -> detection] for every track that got a match this
        frame. A caller can build the "claimed" set from this dict's values
        and can look up which detection any given track just matched.
    """
    matches = {}
    for track in active_tracks:
        if track.state == "dead":
            continue
        _state, chosen, _dist = track.update(detections, chi2_threshold, frame_idx=frame_idx)
        if chosen is not None:
            matches[track.id] = chosen
    return matches


def spawn_tracks_from_unclaimed(detections, claimed, next_track_id):
    """
    Give every detection not claimed by an existing track a brand-new
    tentative Track. A detection is only "claimed" if some track's gated
    match landed exactly on it this frame -- duplicate centroids are the
    one edge case where two blobs could be indistinguishable by value, but
    that requires two blobs at the literal same centroid, which the real
    conveyor data has never produced.

    Returns:
        (new_tracks: list[Track], updated next_track_id: int)
    """
    new_tracks = []
    for det in detections:
        if det in claimed:
            continue
        kf = make_conveyor_kf(*det)
        new_tracks.append(Track(kf, track_id=next_track_id))
        next_track_id += 1
    return new_tracks, next_track_id


def load_classifier(device):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    model = DefectCNN(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, idx_to_class, checkpoint["train_mean"], checkpoint["train_std"]


def classify_blob(frame_bgr, feature, model, idx_to_class, mean, std, device):
    """
    Crop the matched blob out of the ORIGINAL color frame (not the MOG2
    mask), convert to grayscale, resize to CNN_INPUT_SIZE, normalize with
    the checkpoint's own train_mean/train_std (never recompute stats at
    inference time -- that would be leakage in spirit even at eval time),
    and run through DefectCNN. Returns (label, confidence).

    Resize uses NEAREST, not BILINEAR/LANCZOS. Diagnosed via
    quick_interp_check.py against the one real scratch blob available:
    BILINEAR and LANCZOS both smeared the thin scratch line into a soft
    gradient on the 5x upsample (40x37 -> 200x200), which the model then
    read as 'inclusion' (94.8% confidence, wrong). NEAREST preserves the
    sharp edge instead of blending it, and correctly predicted 'scratches'
    (87.1% confidence). Caveat: validated on n=1 real blob only -- NEAREST's
    blockiness could plausibly hurt a defect class that depends on smooth
    gradients rather than sharp edges (e.g. 'patches'); not yet tested
    against those since the synthetic pipeline only ever produces a scratch.
    """
    x, y, w, h = feature["x"], feature["y"], feature["w"], feature["h"]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, frame_bgr.shape[1]), min(y + h, frame_bgr.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None, 0.0  # degenerate crop, e.g. blob clipped at a frame edge

    crop_bgr = frame_bgr[y0:y1, x0:x1]
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    crop_resized = Image.fromarray(crop_gray).resize(
        (CNN_INPUT_SIZE, CNN_INPUT_SIZE), Image.NEAREST
    )
    arr = np.array(crop_resized, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr[np.newaxis, np.newaxis, :, :]).float().to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        pred_idx = int(probs.argmax())

    return idx_to_class[pred_idx], float(probs[pred_idx])


def run_pipeline(chi2_threshold=2.45, save_frames=True):
    frame_paths = sorted(FRAMES_DIR.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in {FRAMES_DIR}. Did Component 1 run and write PNGs there?"
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, idx_to_class, mean, std = load_classifier(device)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    if save_frames:
        DEMO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    active_tracks = []
    next_track_id = 0
    track_best_area = {}  # track_id -> largest matched-blob area seen so far
    track_labels = {}      # track_id -> (label, confidence), from the best crop so far
    log_rows = []

    for i, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise IOError(f"Failed to read frame: {path}")

        raw_mask = bg_subtractor.apply(frame)
        _, solid_mask = cv2.threshold(raw_mask, 254, 255, cv2.THRESH_BINARY)
        features = extract_features(solid_mask)
        features = filter_blobs_by_area(features)
        detections = [blob_centroid(f) for f in features]

        matches = match_and_advance_tracks(active_tracks, detections, chi2_threshold, i)
        claimed = set(matches.values())

        # Re-classify whenever a confirmed track's matched blob is the
        # largest we've seen for it so far -- not just once at first
        # confirmation, since the earliest confirmed frame may still be a
        # partial MOG2 warm-up silhouette (see is_new_best_crop docstring).
        for track in active_tracks:
            if track.state != "confirmed" or track.id not in matches:
                continue
            chosen = matches[track.id]
            matched_feature = features[detections.index(chosen)]
            if is_new_best_crop(track_best_area, track.id, matched_feature["area"]):
                track_labels[track.id] = classify_blob(
                    frame, matched_feature, model, idx_to_class, mean, std, device
                )

        active_tracks = [t for t in active_tracks if t.state != "dead"]
        new_tracks, next_track_id = spawn_tracks_from_unclaimed(detections, claimed, next_track_id)
        active_tracks.extend(new_tracks)

        for track in active_tracks:
            label, conf = track_labels.get(track.id, (None, 0.0))
            chosen = matches.get(track.id)
            bbox = None
            if chosen is not None:
                bbox = tuple(features[detections.index(chosen)][k] for k in ("x", "y", "w", "h"))

            log_rows.append({
                "frame": i,
                "track_id": track.id,
                "state": track.state,
                "bbox": bbox,
                "predicted_class": label,
                "confidence": round(conf, 4) if label else "",
            })

            if save_frames:
                if bbox is not None:
                    # Real match this frame: draw the actual detection bbox.
                    x, y, w, h = bbox
                    color = (0, 255, 0) if track.state == "confirmed" else (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                    text = f"id={track.id} {track.state}"
                    if label:
                        text += f" {label} ({conf:.2f})"
                    cv2.putText(frame, text, (x, max(y - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                else:
                    # Coasting: no real detection this frame. Draw only the
                    # predicted point, not a fabricated box -- attaching a
                    # box to an unrelated blob would misrepresent the match.
                    px, py = int(track.kf.x[0]), int(track.kf.x[1])
                    cv2.drawMarker(frame, (px, py), (0, 0, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=8)

        if save_frames:
            cv2.imwrite(str(DEMO_OUT_DIR / f"frame_{i:03d}.png"), frame)

    with open(LOG_OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame", "track_id", "state", "bbox", "predicted_class", "confidence"]
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Processed {len(frame_paths)} frames.")
    if save_frames:
        print(f"Annotated frames written to {DEMO_OUT_DIR}")
    print(f"Log written to {LOG_OUT_PATH}")

    print("\nFinal per-track classification (from each track's largest-observed crop):")
    for tid, (label, conf) in track_labels.items():
        print(f"  track {tid}: {label} (confidence {conf:.4f}, crop area={track_best_area[tid]:.1f})")

    return log_rows


if __name__ == "__main__":
    run_pipeline()