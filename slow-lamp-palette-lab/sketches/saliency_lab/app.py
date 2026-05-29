"""
Saliency Lab — standalone visualiser for region-contrast palette extraction.
Run: python sketches/saliency_lab/app.py
Port: 5002
"""

import base64
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from skimage.segmentation import mark_boundaries, slic

app = Flask(__name__, static_folder="static", static_url_path="/static")

_MAX_EDGE       = 1024
_N_SAMPLE       = 10_000
_OVERSAMPLE     = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_png(image_np: np.ndarray, mode: str = "RGB") -> str:
    pil = Image.fromarray(image_np, mode=mode)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _resize(image: Image.Image) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= _MAX_EDGE:
        return image
    scale = _MAX_EDGE / longest
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ── Subtraction mask (copied from main app logic — do not import) ─────────────

def _subtraction_mask(image_np: np.ndarray, ref_stream, threshold: int) -> np.ndarray:
    """Return a (H, W) uint8 foreground mask using the same method as the main app."""
    reference = Image.open(ref_stream).convert("RGB")
    reference = reference.resize(
        (image_np.shape[1], image_np.shape[0]), Image.LANCZOS
    )
    ref_np = np.array(reference, dtype=np.uint8)

    diff1 = cv2.subtract(image_np, ref_np)
    diff2 = cv2.subtract(ref_np, image_np)
    diff  = diff1 + diff2

    diff[np.abs(diff) < threshold] = 0

    gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray[gray < 5] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph  = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return morph  # 0 = background, >0 = foreground


# ── Pipeline (self-contained copy — do not import from palette lab) ───────────

def _run_pipeline(
    image_np: np.ndarray,
    n_segments: int,
    spatial_sigma: float,
    k: int,
    min_spread_dist: float,
    fg_mask: np.ndarray = None,
) -> dict:
    H, W = image_np.shape[:2]
    pixels_rgb = image_np.reshape(-1, 3)

    # ── SLIC superpixels ──────────────────────────────────────────────────────
    segments = slic(image_np, n_segments=n_segments, compactness=10,
                    start_label=0, channel_axis=-1)

    # ── Per-segment stats ─────────────────────────────────────────────────────
    lab_image  = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    seg_ids    = np.unique(segments)
    n_segs     = len(seg_ids)

    seg_colors    = np.zeros((n_segs, 3), dtype=np.float32)
    seg_centroids = np.zeros((n_segs, 2), dtype=np.float32)  # [y/H, x/W]
    seg_areas     = np.zeros(n_segs,      dtype=np.float32)

    for i, sid in enumerate(seg_ids):
        mask            = segments == sid
        seg_colors[i]   = lab_image[mask].mean(axis=0)
        ys, xs          = np.where(mask)
        seg_centroids[i] = [ys.mean() / H, xs.mean() / W]
        seg_areas[i]    = mask.sum()

    # ── Region-contrast saliency (Cheng et al. 2011) ─────────────────────────
    # S(i) = Σ_j  area(j) · ΔE(i,j) · exp(−‖p_i−p_j‖² / 2σ²)
    saliency = np.zeros(n_segs, dtype=np.float32)
    for i in range(n_segs):
        color_dists   = np.linalg.norm(seg_colors[i] - seg_colors,    axis=1)
        spatial_dists = np.linalg.norm(seg_centroids[i] - seg_centroids, axis=1)
        spatial_prox  = np.exp(-(spatial_dists ** 2) / (2 * spatial_sigma ** 2))
        saliency[i]   = (seg_areas * color_dists * spatial_prox).sum()

    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency[:] = 1.0

    # ── Per-pixel saliency map ────────────────────────────────────────────────
    saliency_lut = np.zeros(int(seg_ids.max()) + 1, dtype=np.float32)
    for i, sid in enumerate(seg_ids):
        saliency_lut[int(sid)] = saliency[i]
    pixel_saliency = saliency_lut[segments]  # (H, W)

    # ── Visualisations ────────────────────────────────────────────────────────
    heatmap_bgr  = cv2.applyColorMap((pixel_saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    boundaries   = mark_boundaries(image_np, segments, color=(1, 1, 0), mode="outer")
    segments_u8  = (boundaries * 255).astype(np.uint8)

    # ── Gate heatmap with subtraction mask (if provided) ─────────────────────
    if fg_mask is not None:
        mask_f             = (fg_mask > 0).astype(np.float32)
        gated              = pixel_saliency * mask_f
        gated_bgr          = cv2.applyColorMap((gated * 255).astype(np.uint8), cv2.COLORMAP_JET)
        masked_heatmap_enc = _encode_png(cv2.cvtColor(gated_bgr, cv2.COLOR_BGR2RGB))
        # Palette input: foreground pixels in real colour on a dark background
        dark_bg           = np.full_like(image_np, [38, 38, 36], dtype=np.uint8)
        palette_input_enc = _encode_png(np.where(fg_mask[:, :, np.newaxis] > 0, image_np, dark_bg))
        # Sampling pool: foreground pixels only — explicitly excludes background
        mask_bool  = fg_mask.flatten() > 0
        pool_rgb   = pixels_rgb[mask_bool]
        pool_sal   = pixel_saliency.flatten()[mask_bool]
    else:
        masked_heatmap_enc = None
        palette_input_enc  = _encode_png(image_np)
        pool_rgb   = pixels_rgb
        pool_sal   = pixel_saliency.flatten()

    # ── Saliency-weighted sampling ────────────────────────────────────────────
    total      = pool_sal.sum()
    probs      = pool_sal / total if total > 0 else (
                     np.ones(len(pool_sal)) / len(pool_sal))
    n          = min(len(pool_rgb), _N_SAMPLE)
    sample_idx = np.random.choice(len(pool_rgb), size=n, replace=True, p=probs)
    sampled_rgb = pool_rgb[sample_idx]
    sampled_sal = pool_sal[sample_idx]

    # ── Oversample k-means ────────────────────────────────────────────────────
    lab      = cv2.cvtColor(sampled_rgb[np.newaxis, :, :], cv2.COLOR_RGB2LAB)
    lab_flat = lab.reshape(-1, 3).astype(np.float32)
    k_over   = max(k + 1, k * _OVERSAMPLE)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(
        lab_flat, k_over, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()

    # ── Medoids, ordered by mean cluster saliency ─────────────────────────────
    candidates_lab = []
    candidates_rgb = []
    candidates_sal = []

    for idx in range(k_over):
        mask = labels == idx
        if not mask.any():
            continue
        cluster_lab = lab_flat[mask]
        cluster_rgb = sampled_rgb[mask]
        dists       = np.linalg.norm(cluster_lab - centers[idx], axis=1)
        medoid      = np.argmin(dists)
        candidates_lab.append(cluster_lab[medoid])
        candidates_rgb.append(cluster_rgb[medoid])
        candidates_sal.append(float(sampled_sal[mask].mean()))

    order          = np.argsort(-np.array(candidates_sal))
    candidates_lab = np.array(candidates_lab)[order]
    candidates_rgb = np.array(candidates_rgb)[order]

    # ── Max-spread selection with min-distance gate ───────────────────────────
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

        if best_dist >= min_spread_dist:
            selected.append(best)
            remaining.remove(best)
        else:
            selected.append(remaining[0])
            remaining.pop(0)

    palette = [tuple(int(c) for c in candidates_rgb[i]) for i in selected]

    return {
        "original":       _encode_png(image_np),
        "heatmap":        _encode_png(heatmap_rgb),
        "segments":       _encode_png(segments_u8),
        "masked_heatmap": masked_heatmap_enc,
        "palette_input":  palette_input_enc,
        "palette":        palette,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/process", methods=["POST"])
def api_process():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        image = Image.open(request.files["file"].stream).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Unable to read image: {exc}"}), 400

    image    = _resize(image)
    image_np = np.array(image, dtype=np.uint8)

    try:
        n_segments      = max(10,  min(1000, int(  request.form.get("n_segments",     300))))
        spatial_sigma   = max(0.05, min(5.0, float(request.form.get("spatial_sigma",  0.4))))
        k               = max(2,   min(20,   int(  request.form.get("k",              5  ))))
        min_spread_dist = max(0.0, min(50.0, float(request.form.get("min_spread_dist",12 ))))
    except ValueError as exc:
        return jsonify({"error": f"Invalid parameter: {exc}"}), 400

    fg_mask = None
    if "subtract_file" in request.files:
        try:
            threshold = max(0, min(100, int(request.form.get("threshold", 13))))
            fg_mask = _subtraction_mask(image_np, request.files["subtract_file"].stream, threshold)
        except Exception as exc:
            return jsonify({"error": f"Subtraction error: {exc}"}), 400

    try:
        result = _run_pipeline(image_np, n_segments, spatial_sigma, k, min_spread_dist, fg_mask=fg_mask)
    except Exception as exc:
        return jsonify({"error": f"Pipeline error: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
