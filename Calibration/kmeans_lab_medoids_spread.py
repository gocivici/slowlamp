import numpy as np
import cv2

OVERSAMPLE_FACTOR = 3
MIN_SPREAD_DIST = 12  # LAB units — below this, fall back to most-frequent

def extract_palette(image_rgb, k=5):
    pixels_rgb = image_rgb.reshape(-1, 3)
    lab = cv2.cvtColor(pixels_rgb[np.newaxis, :, :], cv2.COLOR_RGB2LAB)
    lab_flat = lab.reshape(-1, 3).astype(np.float32)

    k_over = max(k + 1, k * OVERSAMPLE_FACTOR)

    try:

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
        _, labels, centers = cv2.kmeans(
            lab_flat, k_over, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        labels = labels.flatten()
        counts = np.bincount(labels, minlength=k_over)

        # Get medoid (real pixel) for each oversampled cluster, sorted by count
        candidates_lab = []
        candidates_rgb = []
        sorted_indices = np.argsort(-counts)
        for idx in sorted_indices:
            mask = labels == idx
            cluster_lab = lab_flat[mask]
            cluster_rgb = pixels_rgb[mask]
            dists = np.linalg.norm(cluster_lab - centers[idx], axis=1)
            medoid_idx = np.argmin(dists)
            candidates_lab.append(cluster_lab[medoid_idx])
            candidates_rgb.append(cluster_rgb[medoid_idx])

        candidates_lab = np.array(candidates_lab)
        candidates_rgb = np.array(candidates_rgb)

        # Greedy max-spread selection with minimum distance gate.
        # If the best spread candidate is closer than MIN_SPREAD_DIST to any
        # already-selected colour, fall back to the next most-frequent cluster
        # instead — prevents over-spreading on monochromatic inputs.
        selected = [0]
        remaining = list(range(1, len(candidates_lab)))

        while len(selected) < k and remaining:
            best, best_dist = -1, -1
            for i in remaining:
                min_d = min(
                    np.linalg.norm(candidates_lab[i] - candidates_lab[j])
                    for j in selected
                )
                if min_d > best_dist:
                    best_dist, best = min_d, i

            if best_dist >= MIN_SPREAD_DIST:
                selected.append(best)
                remaining.remove(best)
            else:
                # All remaining candidates are within threshold — take most frequent
                selected.append(remaining[0])
                remaining.pop(0)

        DominantColors = np.array([tuple(int(c) for c in candidates_rgb[i]) for i in selected])
        dominant_counts = np.array([counts[sorted_indices[i]] for i in selected])

        sorted_indices = np.argsort(-dominant_counts)

        # Sort center and label_counts accordingly
        DominantColors_sorted = DominantColors[sorted_indices]
        dominant_counts_sorted = dominant_counts[sorted_indices]

        return DominantColors_sorted, dominant_counts_sorted
    
    except:
        DominantColors = np.array([[255, 255, 255, 0]] * k)
        label_counts_sorted = [0]*k
        return DominantColors, label_counts_sorted