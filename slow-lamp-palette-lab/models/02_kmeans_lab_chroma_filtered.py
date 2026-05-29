import numpy as np
import cv2

CHROMA_FLOOR = 20

def extract_palette(image_rgb, k=5):
    pixels_rgb = image_rgb.reshape(-1, 3)
    lab = cv2.cvtColor(pixels_rgb[np.newaxis, :, :], cv2.COLOR_RGB2LAB)
    lab_flat = lab.reshape(-1, 3).astype(np.float32)

    a = lab_flat[:, 1] - 128
    b = lab_flat[:, 2] - 128
    chroma = np.sqrt(a * a + b * b)

    chromatic = lab_flat[chroma > CHROMA_FLOOR]
    if len(chromatic) < k * 10:
        chromatic = lab_flat  # fallback for near-monochrome scenes

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(
        chromatic, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)

    palette = []
    for idx in np.argsort(-counts):
        rgb = cv2.cvtColor(
            np.clip(np.array([[centers[idx]]]), 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2RGB
        )[0][0]
        palette.append(tuple(int(c) for c in rgb))
    return palette