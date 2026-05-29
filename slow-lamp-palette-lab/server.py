"""
server.py
---------
Flask application entry point. Contains only route handlers.

All image processing lives in preprocessing.py.
All model loading lives in model_registry.py.
"""

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import model_registry
import preprocessing

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/models")
def api_models():
    model_registry.load_models()
    return jsonify(model_registry.get_model_names())


@app.route("/api/extract", methods=["POST"])
def api_extract():
    model_registry.load_models()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    if "model" not in request.form:
        return jsonify({"error": "No model selected"}), 400

    model_name = request.form["model"]
    extract_fn = model_registry.get_model(model_name)
    if extract_fn is None:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    try:
        image = Image.open(request.files["file"].stream).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Unable to read image: {exc}"}), 400

    try:
        blur_sigma = max(0.0, min(20.0, float(request.form.get("blur", 0.0))))
    except ValueError:
        blur_sigma = 0.0

    try:
        k = max(2, min(20, int(request.form.get("k", 5))))
    except ValueError:
        k = 5

    # ── Preprocessing pipeline ────────────────────────────────────────────────
    image = preprocessing.resize_to_limit(image)
    image_np = np.array(image, dtype=np.uint8)
    image_np = preprocessing.apply_blur(image_np, blur_sigma)

    pixels_for_model = image_np
    preview_data = None

    if "subtract_file" in request.files:
        try:
            threshold = max(0, min(100, int(request.form.get("threshold", 13))))
            diff_np, diff_alpha, pixels_for_model = preprocessing.apply_subtraction(
                image_np, request.files["subtract_file"].stream, threshold=threshold
            )
            preview_data = preprocessing.generate_preview(diff_np, diff_alpha)
        except Exception as exc:
            return jsonify({"error": f"Unable to process subtraction image: {exc}"}), 400

        if len(pixels_for_model) == 0:
            return jsonify({
                "model": model_name,
                "palette": [],
                "preview": preview_data,
                "warning": "No pixels differ above the threshold.",
            })
    elif blur_sigma > 0:
        preview_data = preprocessing.generate_preview(image_np)

    # ── Palette extraction ────────────────────────────────────────────────────
    try:
        palette = extract_fn(pixels_for_model, k=k)
    except Exception as exc:
        return jsonify({"error": f"Model error: {exc}"}), 500

    return jsonify({
        "model": model_name,
        "palette": [list(color) for color in palette],
        "preview": preview_data,
    })


if __name__ == "__main__":
    model_registry.load_models()
    app.run(host="0.0.0.0", port=5001, debug=True)
