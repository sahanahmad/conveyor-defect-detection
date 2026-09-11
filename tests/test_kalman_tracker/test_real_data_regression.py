from collections import defaultdict
from pathlib import Path

import pytest

from src.kalman_tracker.kalman_core import make_conveyor_kf
from src.kalman_tracker.track import Track

ROOT = Path(__file__).resolve().parents[2]
CENTROIDS_PATH = ROOT / "outputs" / "frame_5_39_centroids.txt"


def _load_frames(path):
    frames = defaultdict(list)
    with open(path) as f:
        next(f)  # header
        for line in f:
            frame, cx, cy, area = line.split()
            frames[int(frame)].append((float(cx), float(cy), float(area)))
    return frames


def _largest_blob(blobs):
    return max(blobs, key=lambda b: b[2])


@pytest.fixture(scope="module")
def real_frames():
    if not CENTROIDS_PATH.exists():
        pytest.skip(f"real data file not found at {CENTROIDS_PATH}")
    return _load_frames(CENTROIDS_PATH)


@pytest.fixture(scope="module")
def tracked_result(real_frames):
    """Run the real tracker once against the real log; shared across assertions."""
    frame_ids = sorted(real_frames.keys())
    gt0 = _largest_blob(real_frames[frame_ids[0]])
    kf = make_conveyor_kf(gt0[0], gt0[1])
    track = Track(kf, track_id="real_frame5_39")

    per_frame = []
    for frame_id in frame_ids[1:]:
        blobs = real_frames[frame_id]
        detections = [(cx, cy) for cx, cy, _area in blobs]
        gt_xy = (_largest_blob(blobs)[0], _largest_blob(blobs)[1])
        x, chosen, dist = track.update(detections, frame_idx=frame_id)
        per_frame.append({
            "frame_id": frame_id,
            "n_det": len(blobs),
            "chosen": chosen,
            "gt_xy": gt_xy,
            "correct": chosen == gt_xy,
            "state": track.state,
        })
    return {"per_frame": per_frame, "track": track}


def test_real_log_has_expected_clutter_peaks(real_frames):
    assert len(real_frames[11]) == 16
    assert len(real_frames[12]) == 16
    assert len(real_frames[29]) == 18


def test_tracker_matches_ground_truth_every_frame(tracked_result):
    per_frame = tracked_result["per_frame"]
    mismatches = [f for f in per_frame if not f["correct"]]
    assert not mismatches, (
        f"{len(mismatches)}/{len(per_frame)} frames picked the wrong blob: "
        f"{[(m['frame_id'], m['chosen'], m['gt_xy']) for m in mismatches]}"
    )


def test_tracker_correct_through_clutter_peaks(tracked_result):
    by_frame = {f["frame_id"]: f for f in tracked_result["per_frame"]}
    for frame_id in (11, 12, 29):
        assert by_frame[frame_id]["correct"], (
            f"frame {frame_id} (clutter peak, {by_frame[frame_id]['n_det']} blobs): "
            f"chose {by_frame[frame_id]['chosen']}, expected {by_frame[frame_id]['gt_xy']}"
        )


def test_track_stays_confirmed_and_alive_through_real_log(tracked_result):
    track = tracked_result["track"]
    assert track.state == "confirmed"
    assert track.death_log is None