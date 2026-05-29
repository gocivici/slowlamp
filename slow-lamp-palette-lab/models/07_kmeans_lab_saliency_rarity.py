# kmeans_lab_rarity.py
import numpy as np
import cv2

N_SAMPLE = 10000
OVERSAMPLE_FACTOR = 3
MIN_SPREAD_DIST = 12

def extract_palette(image_rgb, k=5):
    pixels_rgb = image_rgb.reshape(-1, 3)
    lab = cv2.cvtColor(pixels_rgb[np.newaxis, :, :], cv2.COLOR_RGB2LAB)
    lab_flat = lab.reshape(-1, 3).astype(np.float32)

    bins = (lab_flat / np.array([2.56, 2.56, 2.56])).astype(np.int32)
    bins = np.clip(bins, 0, 99)
    keys = bins[:, 0] * 10000 + bins[:, 1] * 100 + bins[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    importance = (1.0 / counts[inverse]).astype(np.float32)
    importance /= importance.sum()

    n = min(len(lab_flat), N_SAMPLE)
    sample_idx = np.random.choice(len(lab_flat), size=n, replace=True, p=importance)
    sampled_lab = lab_flat[sample_idx]
    sampled_rgb = pixels_rgb[sample_idx]

    k_over = max(k + 1, k * OVERSAMPLE_FACTOR)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(
        sampled_lab, k_over, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k_over)

    candidates_lab, candidates_rgb = [], []
    for idx in np.argsort(-counts):
        mask = labels == idx
        cluster_lab = sampled_lab[mask]
        cluster_rgb = sampled_rgb[mask]
        dists = np.linalg.norm(cluster_lab - centers[idx], axis=1)
        medoid_idx = np.argmin(dists)
        candidates_lab.append(cluster_lab[medoid_idx])
        candidates_rgb.append(cluster_rgb[medoid_idx])

    candidates_lab = np.array(candidates_lab)
    candidates_rgb = np.array(candidates_rgb)

    selected = [0]
    remaining = list(range(1, len(candidates_lab)))
    while len(selected) < k and remaining:
        best, best_dist = -1, -1
        for i in remaining:
            min_d = min(np.linalg.norm(candidates_lab[i] - candidates_lab[j]) for j in selected)
            if min_d > best_dist:
                best_dist, best = min_d, i
        if best_dist >= MIN_SPREAD_DIST:
            selected.append(best)
            remaining.remove(best)
        else:
            selected.append(remaining[0])
            remaining.pop(0)

    return [tuple(int(c) for c in candidates_rgb[i]) for i in selected]