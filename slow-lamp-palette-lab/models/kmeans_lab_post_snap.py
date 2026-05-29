import numpy as np
import cv2

def extract_palette(image_rgb, k=5):
    pixels_rgb = image_rgb.reshape(-1, 3)
    lab = cv2.cvtColor(pixels_rgb[np.newaxis, :, :], cv2.COLOR_RGB2LAB)
    lab_flat = lab.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, _ = cv2.kmeans(
        lab_flat, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)

    a = lab_flat[:, 1] - 128
    b = lab_flat[:, 2] - 128
    chroma = np.sqrt(a * a + b * b)

    palette = []
    for idx in np.argsort(-counts):
        mask = labels == idx
        best_pixel = pixels_rgb[mask][np.argmax(chroma[mask])]
        palette.append(tuple(int(c) for c in best_pixel))
    return palette