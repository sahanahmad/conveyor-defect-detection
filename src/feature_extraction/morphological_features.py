import cv2


def extract_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect_ratio = w / h
        extent = area / (w * h)

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        # Guard against single-pixel / degenerate blobs (hull_area == 0),
        # which real background-subtraction noise produces regularly.
        # Without this, a stray noise pixel would crash the whole pipeline.
        solidity = area / hull_area if hull_area > 0 else 0

        features.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "solidity": solidity,
        })

    return features


if __name__ == "__main__":
    import numpy as np

    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:100, 60:180] = 255

    for f in extract_features(mask):
        print(f)