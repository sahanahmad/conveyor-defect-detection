import numpy as np
import pytest
from src.kalman_tracker.kalman_core import make_conveyor_kf
from src.kalman_tracker.step import step


def test_step_predicts_every_frame_no_lag():
    """Regression test for the Day 6 bug: skipping predict() on detection
    frames caused est_x to lag true_x on every frame, not just during dropouts."""
    true_positions = [(100 + 5 * t, 60) for t in range(10)]
    kf = make_conveyor_kf(*true_positions[0])
    for t, (tx, ty) in enumerate(true_positions):
        if t == 0:
            continue  # frame 0 is just init, nothing to predict/update yet
        x, chosen, dist = step(kf, [(tx, ty)])
        assert chosen is not None
        # by frame 3 the filter has converged enough that lag should be under 1px
        if t >= 3:
            assert abs(x[0] - tx) < 1.0, f"frame {t}: est_x={x[0]:.2f} lagging true_x={tx}"


def test_step_returns_chosen_detection_when_matched():
    kf = make_conveyor_kf(100, 60)
    x, chosen, dist = step(kf, [(105, 60)])
    assert chosen == (105, 60)
    assert dist is not None


def test_step_returns_none_when_no_detections():
    kf = make_conveyor_kf(100, 60)
    kf.predict()  # warm it up once so P isn't at its initial (huge) value
    x, chosen, dist = step(kf, [])
    assert chosen is None
    assert dist is None


def test_step_rejects_detection_outside_gate():
    """A detection far outside the chi-square gate should be rejected --
    step() should behave as if no detection existed (predict-only)."""
    kf = make_conveyor_kf(100, 60)
    for _ in range(5):
        step(kf, [(kf.x[0] + 5, 60)])  # a few clean frames to converge P down
    x_before = kf.x.copy()
    x_after, chosen, dist = step(kf, [(9999, 9999)])
    assert chosen is None
    assert dist > 2.45
    # state should have moved by the motion model alone, not jumped toward (9999, 9999)
    assert abs(x_after[0] - x_before[0]) < 20


def test_step_uncertainty_grows_during_dropout_and_shrinks_after():
    true_positions = [(100 + 5 * t, 60) for t in range(15)]
    kf = make_conveyor_kf(*true_positions[0])
    dropout_frames = {5, 6, 7, 8}
    p_traces = []
    for t, (tx, ty) in enumerate(true_positions):
        dets = [] if t in dropout_frames else [(tx, ty)]
        step(kf, dets)
        p_traces.append(np.trace(kf.P))

    # P must strictly increase across the dropout window
    dropout_trace = [p_traces[t] for t in sorted(dropout_frames)]
    assert all(a < b for a, b in zip(dropout_trace, dropout_trace[1:])), \
        f"P_trace did not grow monotonically during dropout: {dropout_trace}"

    # and collapse back down once a detection returns (frame 9, right after the dropout)
    assert p_traces[9] < p_traces[max(dropout_frames)]


def test_step_coasts_along_velocity_during_dropout():
    """During a dropout, position should keep advancing at ~the learned
    velocity, not freeze or jump."""
    true_positions = [(100 + 5 * t, 60) for t in range(10)]
    kf = make_conveyor_kf(*true_positions[0])
    for t in range(4):  # let it learn velocity ~5px/frame first
        step(kf, [true_positions[t]])
    x_before, _, _ = step(kf, [])  # frame 4: dropout starts
    x_after, _, _ = step(kf, [])   # frame 5: still dropped out
    assert 3.0 < (x_after[0] - x_before[0]) < 7.0  # advanced roughly one belt-step, not frozen/jumped