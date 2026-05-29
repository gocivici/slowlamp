# Palette Lab

A local web tool for experimenting with color palette extraction algorithms against uploaded images. Built for iterating quickly on new models — drop a Python file into `models/` and the server picks it up live without restarting.

---

## Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd slow-lamp-palette-lab

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python server.py
```

Open **http://localhost:5001** in your browser.

---

## Dependencies

| Package | Purpose |
|---|---|
| `flask` | Local web server and API |
| `numpy` | Array operations on pixel data |
| `opencv-python` | Color space conversion (RGB → LAB), Gaussian blur, pixel diff |
| `pillow` | Image loading, resizing, PNG encoding for previews |
| `scikit-image` | SLIC superpixel segmentation (used by `08_kmeans_lab_region_contrast`) |

---

## Project Structure

```
slow-lamp-palette-lab/
├── server.py             # Flask routes — thin, no business logic
├── model_registry.py     # Discovers and hot-reloads models from models/
├── preprocessing.py      # Blur, subtraction, preview generation
├── requirements.txt
├── models/               # Add your own .py files here (prefix = sidebar order)
│   ├── 01_kmeans_lab.py
│   ├── 02_kmeans_lab_chroma_filtered.py
│   ├── 03_kmeans_lab_chroma_weighted.py
│   ├── 04_kmeans_lab_medoids.py
│   ├── 05_kmeans_lab_filtered_medoids.py
│   ├── 06_kmeans_lab_medoids_spread.py
│   ├── 07_kmeans_lab_saliency_rarity.py
│   └── 08_kmeans_lab_region_contrast.py
├── sketches/
│   └── saliency_lab/     # Standalone SLIC saliency visualiser — port 5002
│       ├── app.py
│       ├── requirements.txt
│       └── static/index.html
└── static/
    ├── index.html        # App shell (structure only)
    ├── style.css         # All styles
    ├── canvas.js         # p5.js rendering layer
    └── app.js            # State, events, API calls
```

---

## Models

Model files are numbered so the sidebar lists them in creation order.

| Model | Color output | Pixel filtering | Notes |
|---|---|---|---|
| `01_kmeans_lab` | Cluster centers | None | Baseline — k-means in LAB with random init |
| `02_kmeans_lab_chroma_filtered` | Cluster centers | Drops low-chroma pixels (chroma < 20) | Skips near-gray pixels before clustering; falls back to all pixels for monochrome scenes |
| `03_kmeans_lab_chroma_weighted` | Cluster centers | Weighted sampling by chroma² | Vivid colors overrepresented in the sample; does not hard-exclude grays |
| `04_kmeans_lab_medoids` | Real pixels | None | Like baseline but returns the actual pixel closest to each center (medoid) |
| `05_kmeans_lab_filtered_medoids` | Real pixels | Drops low-chroma pixels (chroma < 20) | Chroma filter + medoid output combined |
| `06_kmeans_lab_medoids_spread` | Real pixels | None | Oversamples at 3× k, greedily selects k medoids maximally spread in LAB space; anchors on the largest cluster first |
| `07_kmeans_lab_saliency_rarity` | Real pixels | None | Weights pixels inversely by color-bin frequency in LAB space — rare colors are oversampled before clustering |
| `08_kmeans_lab_region_contrast` | Real pixels | None | SLIC superpixels + Cheng et al. region-contrast saliency weighting; falls back to uniform sampling for flat pixel arrays (subtraction mode) |

---

## Adding a New Model

Create a `.py` file anywhere inside `models/`. It must expose a single function:

```python
def extract_palette(image_rgb, k=5):
    """
    Args:
        image_rgb: np.ndarray, dtype uint8.
                   Either (H, W, 3) for the full image
                   or (M, 3) for a pre-filtered set of pixels
                   (used in subtraction mode — see below).
                   Both reshape to (N, 3) via image_rgb.reshape(-1, 3).
        k:         int, number of palette colors to return.

    Returns:
        list of k (R, G, B) tuples, sorted by dominance (most common first).
    """
    ...
```

The server hot-reloads every 8 seconds and on every browser focus event, so you can edit a model and see results without restarting.

### Minimal example

```python
import numpy as np
import cv2

def extract_palette(image_rgb, k=5):
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(pixels[np.newaxis], cv2.COLOR_RGB2LAB)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    palette = []
    for idx in np.argsort(-counts):
        rgb = cv2.cvtColor(np.array([[centers[idx]]]).astype(np.uint8), cv2.COLOR_LAB2RGB)[0][0]
        palette.append(tuple(int(c) for c in rgb))
    return palette
```

---

## Adding a New Preprocessing Step

All image manipulation lives in `preprocessing.py`. Each function takes and returns a `(H, W, 3)` uint8 numpy array. To add a step:

1. Write a function in `preprocessing.py`
2. Call it in `server.py` inside `api_extract()`, between `apply_blur` and the model call

```python
# server.py — preprocessing pipeline
image_np = preprocessing.resize_to_limit(image)
image_np = preprocessing.apply_blur(image_np, blur_sigma)
image_np = preprocessing.your_new_step(image_np, ...)   # ← add here
```

---

## Preprocessing Controls

### Colors k
Number of colors to extract. Adjustable from 2 to 12 in the sidebar (the API accepts up to 20). The swatch row expands horizontally to fit any count.

### Blur σ
Gaussian blur applied before palette extraction. Suppresses fine texture and noise so the model clusters broad color regions rather than individual pixels. Range 0–20.

### Subtraction Mode
Upload a reference image (e.g. the same scene with different lighting or a background plate). The tool computes the per-channel absolute difference `|main − reference|` and extracts the palette only from **pixels in the original image** that differ from the reference above the threshold. This means the palette reflects real colors from changed regions, not difference vectors.

**Threshold (0–100):** Minimum per-pixel difference (max across R, G, B channels) for a pixel to be included. Raise this to filter JPEG compression noise; lower it to include subtle changes. Pixels below the threshold are fully transparent in the canvas preview.

### Show Original Image
Toggle to compare the processed preview (blur / subtraction result) against the unmodified uploaded image.

---

## API

Two endpoints used by the frontend. Both are simple to call from other tools or notebooks if needed.

### `GET /api/models`
Returns a JSON array of available model names.

```json
["kmeans_lab", "kmeans_lab_chroma_filtered", ...]
```

### `POST /api/extract`
Runs palette extraction and returns colors + optional preview.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | image file | yes | Main image |
| `model` | string | yes | Model name from `/api/models` |
| `blur` | float | no | Gaussian sigma, default `0.0` |
| `k` | int | no | Number of palette colors, 2–20, default `5` |
| `subtract_file` | image file | no | Reference image for subtraction mode |
| `threshold` | int | no | Subtraction threshold 0–100, default `10` |

**Response:**

```json
{
  "model": "kmeans_lab",
  "palette": [[120, 80, 60], [200, 190, 175], ...],
  "preview": "data:image/png;base64,..."
}
```

`preview` is `null` when no preprocessing is active. In subtraction mode the preview PNG is RGBA — transparent where pixels are identical, opaque where they differ.

---

## Sketches

Standalone tools for deeper inspection, each self-contained and independent of the main app.

### `sketches/saliency_lab/`

Interactive visualiser for the region-contrast saliency pipeline. Runs on **port 5002**.

```bash
cd slow-lamp-palette-lab
python3 sketches/saliency_lab/app.py
```

**Controls:** n_segments (50–600), spatial σ (0.1–1.0), k (2–10), min spread (0–30). All sliders debounce 200 ms and re-run the full pipeline on change.

**Image views:** Original · Heatmap · Segments · Palette input · Masked heatmap (subtraction mode only)

**Subtraction mode:** Toggle on, upload a reference image, set a threshold (0–100). The saliency heatmap is gated by the foreground mask after it is computed on the full frame — SLIC and region contrast still see the full image. Only the gated pixels feed the clustering.

**`POST /api/process`**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | image file | yes | Main image |
| `n_segments` | int | no | SLIC target segment count, default `300` |
| `spatial_sigma` | float | no | Gaussian bandwidth on normalised centroid distance, default `0.4` |
| `k` | int | no | Palette size, default `5` |
| `min_spread_dist` | float | no | Minimum LAB distance gate for spread selection, default `12` |
| `subtract_file` | image file | no | Reference image for subtraction masking |
| `threshold` | int | no | Subtraction threshold 0–100, default `13` |

Response keys: `original`, `heatmap`, `segments`, `palette_input`, `masked_heatmap` (null when subtraction off), `palette`.
