import numpy as np
try:
    # works when imported via pytest from the repo root (from src.kalman_tracker.step import step)
    from src.kalman_tracker.kalman_core import make_conveyor_kf, gated_nearest_by_mahalanobis
except ImportError:
    # works when run standalone (python src/kalman_tracker/step.py)
    from kalman_core import make_conveyor_kf, gated_nearest_by_mahalanobis


def step(kf, detections, chi2_threshold=2.45):
    """
    Advance a track by one frame.

    Always predicts. Only updates (corrects toward a measurement) when
    gated_nearest_by_mahalanobis accepts a candidate.

    Returns:
        state (np.ndarray): kf.x after this frame's predict (+ update if matched)
        chosen (tuple or None): the accepted detection, or None if no match
        dist (float or None): Mahalanobis distance of the best candidate
            (populated even on a rejected/no-match frame, for diagnostics)
    """
    kf.predict()
    chosen, dist, _S = gated_nearest_by_mahalanobis(kf, detections, chi2_threshold)
    if chosen is not None:
        kf.update(np.array(chosen))
    return kf.x.copy(), chosen, dist


if __name__ == "__main__":
    # Coast-on-no-detection demo: 4-frame dropout, watch P_trace grow then collapse
    true_positions = [(100 + 5 * t, 60) for t in range(15)]
    kf = make_conveyor_kf(*true_positions[0])
    detections = [
        [] if t in (5, 6, 7, 8) else [true_positions[t]]
        for t in range(len(true_positions))
    ]

    print(f"{'frame':>5} {'had_det':>8} {'est_x':>8} {'est_y':>8} {'true_x':>8} {'P_trace':>9}")
    for t, dets in enumerate(detections):
        x, chosen, dist = step(kf, dets)
        tx, ty = true_positions[t]
        print(f"{t:>5} {str(chosen is not None):>8} {x[0]:8.2f} {x[1]:8.2f} {tx:8.2f} {np.trace(kf.P):9.2f}")