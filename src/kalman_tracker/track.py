import logging
try:
    from src.kalman_tracker.step import step
    from src.kalman_tracker.kalman_core import make_conveyor_kf
except ImportError:
    from step import step
    from kalman_core import make_conveyor_kf

logger = logging.getLogger("kalman_tracker")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


class Track:
    def __init__(self, kf, track_id, confirm_hits=3, tentative_max_miss=1, confirmed_max_coast=5):
        self.kf = kf
        self.id = track_id
        self.state = "tentative"
        self.hits = 0     # consecutive hits
        self.misses = 0   # consecutive misses
        self.age = 0
        self.death_log = None  # populated by _kill(); None while alive

        self.confirm_hits = confirm_hits
        self.tentative_max_miss = tentative_max_miss
        self.confirmed_max_coast = confirmed_max_coast

    def update(self, detections, chi2_threshold=2.45, frame_idx=None):
        if self.state == "dead":
            raise RuntimeError(
                f"Track {self.id} is dead (frame {frame_idx}); "
                f"caller should have removed it from the active track list"
            )

        x, chosen, dist = step(self.kf, detections, chi2_threshold)
        self.age += 1

        if chosen is not None:
            self.hits += 1
            self.misses = 0
            if self.state == "tentative" and self.hits >= self.confirm_hits:
                self.state = "confirmed"
        else:
            self.hits = 0
            self.misses += 1
            if self.state == "tentative" and self.misses >= self.tentative_max_miss:
                self._kill("tentative_timeout", frame_idx)
            elif self.state == "confirmed" and self.misses > self.confirmed_max_coast:
                self._kill("confirmed_coast_exceeded", frame_idx)

        return x, chosen, dist

    def _kill(self, reason, frame_idx):
        self.state = "dead"
        self.death_log = {
            "track_id": self.id,
            "reason": reason,
            "frame_idx": frame_idx,
            "age_at_death": self.age,
            "misses_at_death": self.misses,
        }
        logger.info(
            "Track %s died at frame %s: reason=%s, age=%d, misses=%d",
            self.id, frame_idx, reason, self.age, self.misses,
        )


if __name__ == "__main__":
    import numpy as np
    np.random.seed(7)

    def far_clutter_blobs(true_xy, n_clutter, min_offset=80, max_offset=150):
        tx, ty = true_xy
        out = []
        for _ in range(n_clutter):
            dx = np.random.uniform(min_offset, max_offset) * np.random.choice([-1, 1])
            dy = np.random.uniform(min_offset, max_offset) * np.random.choice([-1, 1])
            out.append((tx + dx, ty + dy))
        return out

    def clutter_blobs(true_xy, n_clutter, spread=60):
        tx, ty = true_xy
        return [
            (tx + np.random.uniform(-spread, spread), ty + np.random.uniform(-spread, spread))
            for _ in range(n_clutter)
        ]

    def run(track, frames_dets, label):
        print(f"=== {label} ===")
        print(f"{'frame':>5} {'n_det':>6} {'matched':>8} {'hits':>5} {'miss':>5} {'state':>10}")
        for t, dets in enumerate(frames_dets):
            if track.state == "dead":
                print(f"      -> {track.id} already dead, stopping")
                break
            x, chosen, dist = track.update(dets, frame_idx=t)
            matched = "yes" if chosen is not None else "no"
            print(f"{t:>5} {len(dets):>6} {matched:>8} {track.hits:>5} {track.misses:>5} {track.state:>10}")
        print()

    # Scenario A: genuine track, 18-blob clutter peak + survivable 4-frame dropout
    true_positions_A = [(100 + 5 * t, 60) for t in range(20)]
    trackA = Track(make_conveyor_kf(*true_positions_A[0]), track_id="A")
    frames_A = []
    for t, true_xy in enumerate(true_positions_A):
        if t in (10, 11):
            frames_A.append(clutter_blobs(true_xy, 17) + [true_xy])
        elif t in (14, 15, 16, 17):
            frames_A.append(clutter_blobs(true_xy, 5))
        else:
            frames_A.append([true_xy])
    run(trackA, frames_A, "Scenario A: clutter peak + 4-frame dropout (survives)")

    # Scenario B: spurious track on a one-off noise blob, dies fast
    trackB = Track(make_conveyor_kf(300, 90), track_id="B")
    frames_B = [clutter_blobs((320 + 5 * t, 95), 6) for t in range(1, 5)]
    run(trackB, frames_B, "Scenario B: spurious birth (tentative_timeout)")

    # Scenario C: confirmed track that coasts past its cap
    true_positions_C = [(200 + 5 * t, 60) for t in range(15)]
    trackC = Track(make_conveyor_kf(*true_positions_C[0]), track_id="C")
    frames_C = []
    for t, true_xy in enumerate(true_positions_C):
        if t >= 4:
            frames_C.append(far_clutter_blobs(true_xy, 4))
        else:
            frames_C.append([true_xy])
    run(trackC, frames_C, "Scenario C: confirmed track coasts past cap (confirmed_coast_exceeded)")

    print("Death logs collected:")
    for tr in (trackA, trackB, trackC):
        print(f"  {tr.id}: {tr.death_log}")