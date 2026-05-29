"""
model_registry.py
-----------------
Discovers and hot-reloads palette-extraction models from the models/ directory.

Each model is a .py file that exposes a single function:

    extract_palette(image_rgb: np.ndarray, k: int = 5) -> list[tuple[int, int, int]]

`image_rgb` is either:
  - an (H, W, 3) uint8 array  — the full image
  - an (M, 3) uint8 array     — a pre-filtered set of pixels (subtraction mode)

Both shapes flatten to (N, 3) via reshape(-1, 3), so models handle them identically.
The function returns k RGB tuples sorted by dominance (most common cluster first).

To add a new model: drop a .py file into models/ and reload the page.
"""

import importlib.util
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent / "models"

_models: dict = {}
_snapshot: dict = {}


def load_models() -> None:
    """Scan models/ and reload any file that is new or has changed on disk."""
    global _snapshot

    current = {}
    for path in _MODELS_DIR.glob("*.py"):
        try:
            current[path] = path.stat().st_mtime
        except OSError:
            continue

    if current == _snapshot and _models:
        return

    _snapshot = current
    _models.clear()
    for path in sorted(current):
        name = path.stem
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _models[name] = getattr(module, "extract_palette")
        except Exception as exc:
            print(f"[model_registry] Skipping {path.name}: {exc}")


def get_model_names() -> list[str]:
    """Return alphabetically sorted names of all successfully loaded models."""
    return sorted(_models.keys())


def get_model(name: str):
    """Return the extract_palette function for `name`, or None if not found."""
    return _models.get(name)
