import numpy as np
import cv2


def extract_palette(image_rgb, k=5):
    pixels = image_rgb.reshape(-1, 3)
    pixels = pixels[np.newaxis, :, :]
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB)
    lab_float = np.float32(lab)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(
        lab_float, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)
    sorted_idx = np.argsort(-counts)

    palette = []
    for idx in sorted_idx:
        lab_color = centers[idx]
        rgb = cv2.cvtColor(
            np.array([[lab_color]]).astype(np.uint8), cv2.COLOR_LAB2RGB
        )[0][0]
        palette.append(tuple(int(c) for c in rgb))

    return palette
