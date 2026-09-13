import pytest
from src.kalman_tracker.kalman_core import make_conveyor_kf
from src.kalman_tracker.track import Track
from src.pipeline.integrate import (
    blob_centroid,
    filter_blobs_by_area,
    is_new_best_crop,
    match_and_advance_tracks,
    spawn_tracks_from_unclaimed,
)


def make_feature(x, y, w=10, h=10):
    return {"x": x, "y": y, "w": w, "h": h}


def test_blob_centroid_matches_frame_5_39_convention():
    # (x + w/2, y + h/2) -- must match wire_mog2_to_features.py's centroid
    # dump exactly, since Component 4 was validated against that data.
    f = make_feature(x=10, y=20, w=6, h=8)
    assert blob_centroid(f) == (13.0, 24.0)


def test_filter_blobs_by_area_drops_noise_keeps_real_blob():
    noise = [make_feature(0, 0, w=1, h=1) for _ in range(5)]
    for f, area in zip(noise, [0.0, 1.0, 2.0, 3.5, 11.0]):
        f["area"] = area
    real_blob = make_feature(100, 60, w=15, h=8)
    real_blob["area"] = 123.0

    kept = filter_blobs_by_area(noise + [real_blob], min_area=20)
    assert kept == [real_blob]


def test_filter_blobs_by_area_default_threshold():
    borderline_noise = make_feature(0, 0)
    borderline_noise["area"] = 11.0  # highest noise value ever observed
    real = make_feature(100, 60)
    real["area"] = 123.0  # lowest real-blob value ever observed

    kept = filter_blobs_by_area([borderline_noise, real])
    assert kept == [real]


def test_is_new_best_crop_true_on_first_observation():
    best_areas = {}
    assert is_new_best_crop(best_areas, track_id=0, area=123.5) is True
    assert best_areas[0] == 123.5


def test_is_new_best_crop_true_when_area_grows():
    best_areas = {0: 123.5}
    assert is_new_best_crop(best_areas, track_id=0, area=585.0) is True
    assert best_areas[0] == 585.0


def test_is_new_best_crop_false_when_area_shrinks_or_ties():
    best_areas = {0: 1150.5}
    assert is_new_best_crop(best_areas, track_id=0, area=900.0) is False
    assert is_new_best_crop(best_areas, track_id=0, area=1150.5) is False
    assert best_areas[0] == 1150.5  # unchanged


def test_is_new_best_crop_tracks_are_independent():
    best_areas = {0: 500.0}
    assert is_new_best_crop(best_areas, track_id=1, area=10.0) is True
    assert best_areas == {0: 500.0, 1: 10.0}


def test_spawn_creates_one_tentative_track_per_unclaimed_detection():
    detections = [(100.0, 60.0), (200.0, 60.0)]
    new_tracks, next_id = spawn_tracks_from_unclaimed(detections, claimed=set(), next_track_id=0)
    assert [t.id for t in new_tracks] == [0, 1]
    assert all(t.state == "tentative" for t in new_tracks)
    assert next_id == 2


def test_spawn_skips_claimed_detections():
    detections = [(100.0, 60.0), (200.0, 60.0)]
    new_tracks, next_id = spawn_tracks_from_unclaimed(
        detections, claimed={(100.0, 60.0)}, next_track_id=5
    )
    assert len(new_tracks) == 1
    assert new_tracks[0].id == 5
    assert next_id == 6


def test_match_and_advance_returns_only_tracks_that_matched():
    real_track = Track(make_conveyor_kf(100, 60), track_id=0)
    detections = [(105.0, 60.0)]  # within gate of the real track's prediction
    matches = match_and_advance_tracks([real_track], detections, chi2_threshold=2.45, frame_idx=1)
    assert matches == {0: (105.0, 60.0)}


def test_match_and_advance_skips_dead_tracks():
    dead_track = Track(make_conveyor_kf(100, 60), track_id=0)
    dead_track.state = "dead"
    matches = match_and_advance_tracks([dead_track], detections=[(100.0, 60.0)],
                                        chi2_threshold=2.45, frame_idx=1)
    assert matches == {}


def test_spurious_clutter_track_dies_without_stealing_the_real_tracks_slot():
    """End-to-end miniature: one real moving blob + one one-off clutter blob
    that never reappears. The clutter track must die on its first miss
    (tentative_max_miss=1 default) while the real track keeps accumulating
    hits toward confirmation, uninterrupted."""
    frames = [
        [make_feature(100, 60)],
        [make_feature(105, 60), make_feature(300, 300, w=4, h=4)],
        [make_feature(110, 60)],
        [make_feature(115, 60)],
    ]

    active_tracks, next_id = [], 0
    for i, feats in enumerate(frames):
        dets = [blob_centroid(f) for f in feats]
        matches = match_and_advance_tracks(active_tracks, dets, 2.45, frame_idx=i)
        claimed = set(matches.values())
        active_tracks = [t for t in active_tracks if t.state != "dead"]
        new_tracks, next_id = spawn_tracks_from_unclaimed(dets, claimed, next_id)
        active_tracks.extend(new_tracks)

    assert len(active_tracks) == 1
    assert active_tracks[0].id == 0
    assert active_tracks[0].state == "confirmed"