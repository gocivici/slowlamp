"""
preprocessing.py
----------------
Image preprocessing steps applied to the uploaded image before palette extraction.

Pipeline order:
    1. resize_to_limit   — keeps memory and compute bounded
    2. apply_blur        — Gaussian smoothing to suppress fine texture noise
    3. apply_subtraction — (optional) isolate pixels that differ from a reference image

Adding a new preprocessing step:
    - Write a function here that takes/returns np.ndarray (H, W, 3) uint8.
    - Call it in server.py between apply_blur and the model invocation.
"""

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def resize_to_limit(image: Image.Image, max_edge: int = 1024) -> Image.Image:
    """Downscale so the longest edge is at most max_edge pixels."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def apply_blur(image_np: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with the given sigma. Returns image unchanged if sigma <= 0."""
    if sigma <= 0:
        return image_np
    return cv2.GaussianBlur(image_np, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_subtraction(
    image_np: np.ndarray,
    subtract_stream,
    threshold: int = 13,
) -> tuple:
    """
    Subtract a reference image from image_np to isolate regions of color change.
    Matches the Pi implementation in fgc_yumeng.py:
      - Double cv2.subtract (saturated, captures both directions)
      - Per-channel threshold, then second grayscale threshold at 5
      - Morphological closing (MORPH_ELLIPSE 5x5) to fill holes in the mask
      - bitwise_and to produce a clean foreground image for preview

    Returns (fgimg, diff_alpha, pixels_for_model) where:
      fgimg            — (H, W, 3) original image with background zeroed out
      diff_alpha       — (H, W)   closed foreground mask (uint8, 0=background)
      pixels_for_model — (M, 3)   original colors from foreground pixels
    """
    reference = Image.open(subtract_stream).convert("RGB")
    reference = reference.resize(
        (image_np.shape[1], image_np.shape[0]), Image.LANCZOS
    )
    reference_np = np.array(reference, dtype=np.uint8)

    # Double saturated subtract — equivalent to absdiff but matches Pi approach
    diff1 = cv2.subtract(image_np, reference_np)
    diff2 = cv2.subtract(reference_np, image_np)
    diff = diff1 + diff2  # safe: only one side is ever non-zero per channel

    # First threshold: suppress small per-channel differences
    diff[np.abs(diff) < threshold] = 0

    # Second threshold: convert to grayscale, suppress residual noise
    gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray[gray < 5] = 0
    fgmask = gray.astype(np.uint8)

    # Morphological closing: fill holes so the mask covers connected regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Foreground image for preview: original colours, background zeroed
    fgimg = cv2.bitwise_and(image_np, image_np, mask=morph)

    mask_bool = morph > 0
    pixels_for_model = image_np[mask_bool]  # may be empty if nothing changed

    return fgimg, morph, pixels_for_model


def generate_preview(image_np: np.ndarray, diff_alpha: np.ndarray = None) -> str:
    """
    Encode image_np as a base64 PNG data-URI for display in the browser.

    If diff_alpha is provided, the PNG is RGBA: transparent where the diff is
    zero, opaque where colors diverge. This lets the canvas background show
    through identical regions so students can see exactly what the model sees.
    """
    if diff_alpha is not None:
        rgba_np = np.dstack([image_np, diff_alpha])
        pil_image = Image.fromarray(rgba_np, mode="RGBA")
    else:
        pil_image = Image.fromarray(image_np)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded
