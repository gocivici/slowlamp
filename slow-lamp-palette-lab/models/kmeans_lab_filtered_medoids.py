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

    mask = chroma > CHROMA_FLOOR
    chromatic_lab = lab_flat[mask]
    chromatic_rgb = pixels_rgb[mask]

    if len(chromatic_lab) < k * 10:
        chromatic_lab = lab_flat
        chromatic_rgb = pixels_rgb

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(
        chromatic_lab, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)

    palette = []
    for idx in np.argsort(-counts):
        cluster_mask = labels == idx
        cluster_lab = chromatic_lab[cluster_mask]
        cluster_rgb = chromatic_rgb[cluster_mask]
        dists = np.linalg.norm(cluster_lab - centers[idx], axis=1)
        medoid = cluster_rgb[np.argmin(dists)]
        palette.append(tuple(int(c) for c in medoid))
    return palette