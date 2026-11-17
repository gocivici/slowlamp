"""
Slow Lamp replay visualizer — Flask backend.

    python3 server.py   ->   http://localhost:5002
"""

import base64
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request, send_from_directory

import engine

PORT = 5002
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BATCH = 256

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")

# One sample and one playback at a time.
state = {"records": None, "playback": None}


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/load", methods=["POST"])
def load():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="No file uploaded"), 400

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, os.path.basename(f.filename))
        f.save(path)
        try:
            kind, records, skipped = engine.parse_sample(f.filename, path)
        except Exception as e:
            return jsonify(error=f"Could not read {f.filename}: {e}"), 400
    if not records:
        return jsonify(error=f"No usable records in {f.filename}"), 400

    if state["playback"]:
        state["playback"].cancel()
    state.update(records=records, playback=None)

    return jsonify(
        name=f.filename,
        kind=kind,
        skipped=skipped,
        records=records,
        layout=engine.panel_layout(),
        default_interval=engine.DEFAULT_INTERVAL,
    )


@app.route("/api/playback", methods=["POST"])
def playback():
    records = state["records"]
    if records is None:
        return jsonify(error="No sample loaded"), 400
    opts = request.get_json(force=True)

    try:
        indices, info = engine.select(records, opts)
    except Exception as e:
        return jsonify(error=f"Sampling failed: {e}"), 400
    if not indices:
        return jsonify(error="No records match this sampling"), 400

    interval = float(opts.get("interval") or engine.DEFAULT_INTERVAL)
    layout = opts.get("layout") if opts.get("layout") in engine.CLUSTER_LAYOUTS else "pcb"

    if state["playback"]:
        state["playback"].cancel()
    engine.set_cluster_layout(layout)
    pb = engine.Playback(records, indices, engine.frames_for_interval(interval))
    state["playback"] = pb

    return jsonify(
        indices=indices,
        info=info,
        interval=interval,
        fps=engine.PLAYBACK_FPS,
        frames_per_transition=pb.frames_per_transition,
        transitions=pb.total,
        leds=int(engine.anim.num_leds),
        layout=layout,
    )


@app.route("/api/progress")
def progress():
    pb = state["playback"]
    if pb is None:
        return jsonify(ready=0, total=0)
    return jsonify(ready=pb.ready(), total=pb.total, error=pb.error)


@app.route("/api/batch")
def batch():
    pb = state["playback"]
    if pb is None:
        return jsonify(error="No playback"), 400
    start = request.args.get("start", type=int, default=0)
    count = max(0, min(MAX_BATCH, request.args.get("count", type=int, default=64)))
    items = pb.batch(start, count)
    return jsonify(
        start=start,
        count=len(items),
        frames=base64.b64encode(b"".join(drive for drive, _ in items)).decode("ascii"),
        swatch=[sw for _, sw in items],
    )


if __name__ == "__main__":
    print(f"Slow Lamp visualizer on http://localhost:{PORT}")
    app.run(port=PORT, debug=False, threaded=True)
