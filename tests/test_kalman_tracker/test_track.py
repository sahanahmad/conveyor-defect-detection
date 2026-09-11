import pytest
from src.kalman_tracker.kalman_core import make_conveyor_kf
from src.kalman_tracker.track import Track


def make_track(**kwargs):
    return Track(make_conveyor_kf(100, 60), track_id="T", **kwargs)


def test_track_starts_tentative():
    t = make_track()
    assert t.state == "tentative"
    assert t.hits == 0 and t.misses == 0


def test_track_confirms_after_confirm_hits_consecutive_hits():
    t = make_track(confirm_hits=3)
    true_positions = [(100 + 5 * k, 60) for k in range(1, 4)]
    for i, xy in enumerate(true_positions, start=1):
        t.update([xy], frame_idx=i)
        if i < 3:
            assert t.state == "tentative", f"promoted too early at hit {i}"
    assert t.state == "confirmed"
    assert t.hits == 3


def test_track_tentative_dies_on_default_single_miss():
    t = make_track()  # tentative_max_miss=1 default
    t.update([], frame_idx=1)  # immediate miss, never matched
    assert t.state == "dead"
    assert t.death_log["reason"] == "tentative_timeout"


def test_track_tentative_survives_within_higher_miss_allowance():
    t = make_track(tentative_max_miss=2)
    t.update([], frame_idx=1)  # 1 miss, allowance is 2
    assert t.state == "tentative"
    assert t.death_log is None


def test_track_confirmed_survives_coast_within_cap():
    t = make_track(confirm_hits=3, confirmed_max_coast=5)
    true_positions = [(100 + 5 * k, 60) for k in range(1, 4)]
    for i, xy in enumerate(true_positions, start=1):
        t.update([xy], frame_idx=i)
    assert t.state == "confirmed"
    for i in range(4, 4 + 5):  # exactly 5 misses -- at the cap, not over it
        t.update([], frame_idx=i)
        assert t.state == "confirmed", f"died early at miss count {t.misses}"
    assert t.death_log is None


def test_track_confirmed_dies_after_exceeding_coast_cap():
    t = make_track(confirm_hits=3, confirmed_max_coast=5)
    true_positions = [(100 + 5 * k, 60) for k in range(1, 4)]
    for i, xy in enumerate(true_positions, start=1):
        t.update([xy], frame_idx=i)
    for i in range(4, 4 + 6):  # 6 misses -- one past the cap of 5
        t.update([], frame_idx=i)
    assert t.state == "dead"
    assert t.death_log["reason"] == "confirmed_coast_exceeded"
    assert t.death_log["misses_at_death"] == 6


def test_track_death_log_has_expected_fields():
    t = make_track()
    t.update([], frame_idx=7)
    log = t.death_log
    assert log["track_id"] == "T"
    assert log["frame_idx"] == 7
    assert log["reason"] == "tentative_timeout"
    assert log["age_at_death"] == 1
    assert log["misses_at_death"] == 1


def test_track_raises_if_updated_after_death():
    t = make_track()
    t.update([], frame_idx=1)
    assert t.state == "dead"
    with pytest.raises(RuntimeError):
        t.update([(999, 999)], frame_idx=2)


def test_track_hit_resets_miss_streak():
    """A confirmed track that misses a few frames then hits again should
    have its miss counter reset, not carry the streak forward."""
    t = make_track(confirm_hits=3, confirmed_max_coast=5)
    true_positions = [(100 + 5 * k, 60) for k in range(1, 4)]
    for i, xy in enumerate(true_positions, start=1):
        t.update([xy], frame_idx=i)
    t.update([], frame_idx=4)
    t.update([], frame_idx=5)
    assert t.misses == 2
    t.update([(115, 60)], frame_idx=6)  # real detection returns
    assert t.misses == 0
    assert t.hits == 1
    assert t.state == "confirmed"