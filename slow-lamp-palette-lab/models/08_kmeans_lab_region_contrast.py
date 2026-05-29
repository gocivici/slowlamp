import numpy as np
import cv2
from skimage.segmentation import slic

OVERSAMPLE_FACTOR = 3
MIN_SPREAD_DIST   = 12     # LAB units — minimum inter-colour distance gate
N_SEGMENTS        = 300    # target superpixel count for SLIC
SPATIAL_SIGMA     = 0.4    # Gaussian bandwidth on normalised [0,1] centroid coords
N_SAMPLE          = 10_000 # pixels drawn for k-means after saliency weighting


def extract_palette(image_rgb, k=5):
    pixels_rgb = image_rgb.reshape(-1, 3)

    # ── Saliency-weighted sampling (2-D input only) ───────────────────────────
    # Subtraction mode passes a flat (M, 3) array with no spatial layout, so
    # SLIC is not possible — fall back to uniform sampling in that case.
    if image_rgb.ndim == 3:
        H, W = image_rgb.shape[:2]

        segments = slic(image_rgb, n_segments=N_SEGMENTS, compactness=10,
                        start_label=0, channel_axis=-1)

        lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        seg_ids = np.unique(segments)
        n_segs  = len(seg_ids)

        seg_colors    = np.zeros((n_segs, 3), dtype=np.float32)
        seg_centroids = np.zeros((n_segs, 2), dtype=np.float32)  # [y/H, x/W]
        seg_areas     = np.zeros(n_segs, dtype=np.float32)

        for i, sid in enumerate(seg_ids):
            mask = segments == sid
            seg_colors[i]    = lab_image[mask].mean(axis=0)
            ys, xs           = np.where(mask)
            seg_centroids[i] = [ys.mean() / H, xs.mean() / W]
            seg_areas[i]     = mask.sum()

        # Region-contrast saliency (Cheng et al. 2011):
        # S(i) = Σ_j  area(j) · ΔE(i,j) · exp(−‖p_i−p_j‖² / 2σ²)
        # Self-term (i=j) contributes 0 via color_dist=0, no exclusion needed.
        saliency = np.zeros(n_segs, dtype=np.float32)
        for i in range(n_segs):
            color_dists   = np.linalg.norm(seg_colors[i] - seg_colors, axis=1)
            spatial_dists = np.linalg.norm(seg_centroids[i] - seg_centroids, axis=1)
            spatial_prox  = np.exp(-(spatial_dists ** 2) / (2 * SPATIAL_SIGMA ** 2))
            saliency[i]   = (seg_areas * color_dists * spatial_prox).sum()

        s_min, s_max = saliency.min(), saliency.max()
        if s_max > s_min:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency[:] = 1.0

        # LUT maps each segment ID → its saliency score
        saliency_lut = np.zeros(int(seg_ids.max()) + 1, dtype=np.float32)
        for i, sid in enumerate(seg_ids):
            saliency_lut[int(sid)] = saliency[i]

        pixel_weights = saliency_lut[segments].flatten()
        probs         = pixel_weights / pixel_weights.sum()
        n             = min(len(pixels_rgb), N_SAMPLE)
        sample_idx    = np.random.choice(len(pixels_rgb), size=n, replace=True, p=probs)
    else:
        n          = min(len(pixels_rgb), N_SAMPLE)
        sample_idx = np.random.choice(len(pixels_rgb), size=n, replace=True)

    sampled_rgb = pixels_rgb[sample_idx]

    # ── Oversample → medoid → max-spread selection ───────────────────────────
    lab      = cv2.cvtColor(sampled_rgb[np.newaxis, :, :], cv2.COLOR_RGB2LAB)
    lab_flat = lab.reshape(-1, 3).astype(np.float32)

    k_over   = max(k + 1, k * OVERSAMPLE_FACTOR)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(
        lab_flat, k_over, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k_over)

    candidates_lab = []
    candidates_rgb = []
    for idx in np.argsort(-counts):
        mask        = labels == idx
        cluster_lab = lab_flat[mask]
        cluster_rgb = sampled_rgb[mask]
        dists       = np.linalg.norm(cluster_lab - centers[idx], axis=1)
        medoid_idx  = np.argmin(dists)
        candidates_lab.append(cluster_lab[medoid_idx])
        candidates_rgb.append(cluster_rgb[medoid_idx])

    candidates_lab = np.array(candidates_lab)
    candidates_rgb = np.array(candidates_rgb)

    selected  = [0]
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
            selected.append(remaining[0])
            remaining.pop(0)

    return [tuple(int(c) for c in candidates_rgb[i]) for i in selected]
