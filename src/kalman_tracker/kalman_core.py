import numpy as np
from filterpy.kalman import KalmanFilter

def make_conveyor_kf(x0, y0, dt=1.0):
    kf = KalmanFilter(dim_x=4, dim_z=2)

    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ])

    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    kf.x = np.array([x0, y0, 0., 0.])
    kf.P = np.diag([10., 10., 100., 100.])
    kf.R = np.diag([4., 4.])
    kf.Q = np.diag([0.5, 0.5, 0.5, 0.5])
    return kf

def innovation_covariance(kf):
    #Quantify how far an incoming measurement deviates from expectation
    return kf.H @ kf.P @ kf.H.T + kf.R

def mahalanobis_distance(kf, S, detection_xy):
    predicted_xy = kf.H @ kf.x
    residual = np.array(detection_xy, dtype=float) - predicted_xy
    S_inv = np.linalg.inv(S)
    d_squared = residual.T @ S_inv @ residual
    return float(np.sqrt(d_squared))

def gated_nearest_by_mahalanobis(kf, detections, chi2_threshold=2.45):
    #Calculates how many SD away an incoming detection is from the predicted location
    #chi2 2.45 -> outside of 95% confidence region
    if not detections:
        return None, None, innovation_covariance(kf)

    S = innovation_covariance(kf)
    scored = [(d, mahalanobis_distance(kf, S, d)) for d in detections]
    scored.sort(key=lambda pair: pair[1])
    best_det, best_dist = scored[0]

    if best_dist > chi2_threshold:
        return None, best_dist, S
    return best_det, best_dist, S

if __name__ == "__main__":
    true_positions = [(100 + 5 * t, 60) for t in range(10)]
    kf = make_conveyor_kf(*true_positions[0])

    print(f"{'frame':>5} {'measured':>12} {'gated_choice':>14} {'maha_dist':>10}")
    for t, (mx, my) in enumerate(true_positions):
        if t > 0:
            kf.predict()
            chosen, dist, _ = gated_nearest_by_mahalanobis(kf, [(mx, my)])
            if chosen is not None:
                kf.update(np.array(chosen))
            print(f"{t:>5} {str((mx, my)):>12} {str(chosen):>14} {round(dist, 2) if dist else '--':>10}")
        else:
            print(f"{t:>5} {str((mx, my)):>12} {'(init)':>14} {'--':>10}")

    print()
    print("Final learned velocity (vx, vy):", (round(kf.x[2], 2), round(kf.x[3], 2)))
        