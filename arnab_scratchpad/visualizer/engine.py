"""
Loads field samples (trace-log .txt or cover archive .png) and replays a selection of
records through the frozen production animation code in pipeline_snapshot/: keyframes
from generate_frame_from_trace_tracks, OKLab fades from interpolate_two_frames. Each
pair of records fades over a fixed interval at 60 fps.
"""

import io
import os
import re
import sys
import time
import random
import threading
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

# The Pi runs on lamp local time; moon_helper uses datetime.fromtimestamp() without a zone.
os.environ["TZ"] = "America/Vancouver"
time.tzset()

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_snapshot")
sys.path.insert(0, SNAPSHOT_DIR)

with redirect_stdout(io.StringIO()):
    import helper_classes
    import cover
    import fgc_arnab_animation as anim
    from moon_helper import MoonHelper

import workers

LAMP_TZ = ZoneInfo("America/Vancouver")
MOON_TABLE = os.path.join(SNAPSHOT_DIR, "hourly_moon_table_25-26.csv")

PLAYBACK_FPS = 60
DEFAULT_INTERVAL = 0.3  # seconds per fade
SEED = 0

# LED positions the clustering runs on. The panel is the PCB rings; fgc_arnab.py line 159
# currently uses the 23 cm grid (switched back from the PCB rings in 30cc068).
CLUSTER_LAYOUTS = {
    "pcb": helper_classes.get_led_points_pcb,
    "grid23": helper_classes.get_led_points_23cm,
}

# Badly encoded chunk: a long run of records whose 5 colours all have blue pinned at 255.
BAD_RUN_MIN = 24


# ── Parsing ──────────────────────────────────────────────────────────────────

_TRACE_RE = re.compile(
    r"main_color \(rgb\) #(?P<mc>\d+): \((?P<mr>\d+), (?P<mg>\d+), (?P<mb>\d+)\);"
    r"\s*supplemental_colors: \[(?P<supp>.*?)\];\s*@ (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
)
_SUPP_RE = re.compile(r"#(\d+) \((\d+), (\d+), (\d+)\)")


def _record(t, colors, counts):
    counts = [int(c) for c in counts]
    return {
        "t": int(t),
        "hour": int(t // 3600),  # what cover.py stores as the timestamp
        "label": datetime.fromtimestamp(t, LAMP_TZ).strftime("%Y-%m-%d %H:%M"),
        "colors": [[int(c) for c in col[:3]] for col in colors],
        "counts": counts,
    }


def parse_trace_log(text):
    """vibrant, then largest, color3, color4, color5. Empty captures (white, count 0)
    are skipped like light_simulation/record_to_year_new.py does; they never reach archive.png."""
    records, empty, unparsed = [], 0, 0
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _TRACE_RE.search(line)
        supp = _SUPP_RE.findall(m.group("supp")) if m else []
        if not m or len(supp) != 4:
            unparsed += 1
            continue
        if (m.group("mr"), m.group("mg"), m.group("mb")) == ("255", "255", "255"):
            empty += 1
            continue
        colors = [(m.group("mr"), m.group("mg"), m.group("mb"))] + [s[1:] for s in supp]
        counts = [m.group("mc")] + [s[0] for s in supp]
        local = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=LAMP_TZ)
        records.append(_record(local.timestamp(), colors, counts))
    return records, {"empty": empty, "unparsed": unparsed}


def parse_cover_png(path):
    with redirect_stdout(io.StringIO()):
        blocks = cover.retrieve(filename=path)
    records = []
    for b in blocks:
        colors = [b["vc"], b["ac1"], b["ac2"], b["ac3"], b["ac4"]]
        counts = [b["vc_px"], b["ac1_px"], b["ac2_px"], b["ac3_px"], b["ac4_px"]]
        records.append(_record(b["timestamp"] * 3600, colors, counts))
    return records, {}


def drop_bad_runs(records):
    def pinned(r):
        return all(c[2] == 255 for c in r["colors"])

    keep, bad, i = [], 0, 0
    while i < len(records):
        j = i
        while j < len(records) and pinned(records[j]):
            j += 1
        if j - i >= BAD_RUN_MIN:
            bad += j - i
        else:
            keep.extend(records[i:j])
        if j == i:
            keep.append(records[i])
            j += 1
        i = j
    return keep, bad


def parse_sample(filename, path):
    if filename.lower().endswith(".png"):
        kind, (records, skipped) = "cover", parse_cover_png(path)
    else:
        with open(path, "r", errors="replace") as f:
            kind, (records, skipped) = "trace_log", parse_trace_log(f.read())
    records, skipped["bad_encoding"] = drop_bad_runs(records)
    return kind, records, skipped


# ── Sampling ─────────────────────────────────────────────────────────────────

_moon_helper = None


def select(records, opts):
    """Pick which records go into the moon queue. Returns (indices, info)."""
    mode = opts.get("mode", "all")

    if mode == "moon":
        global _moon_helper
        if _moon_helper is None:
            _moon_helper = MoonHelper(MOON_TABLE)
        current = datetime.strptime(opts["moon_time"], "%Y-%m-%dT%H:%M").replace(tzinfo=LAMP_TZ)
        hours = [r["hour"] for r in records]
        with redirect_stdout(io.StringIO()):
            illum, idx = _moon_helper.find_matched_indices(current, hours)
        return [int(i) for i in idx], {"illumination": float(illum)}

    idx = list(range(len(records)))
    if mode == "filter":
        lo, hi, hour = opts.get("from"), opts.get("to"), opts.get("hour")
        if lo:
            idx = [i for i in idx if records[i]["label"][:10] >= lo]
        if hi:
            idx = [i for i in idx if records[i]["label"][:10] <= hi]
        if hour not in (None, ""):
            idx = [i for i in idx if int(records[i]["label"][11:13]) == int(hour)]
        idx = idx[:: max(1, int(opts.get("step") or 1))]
    return idx, {}


def frames_for_interval(interval_s):
    return max(1, round(interval_s * PLAYBACK_FPS))


def set_cluster_layout(name):
    """Point the snapshot's module globals (fgc_arnab.py lines 159-167) at another LED
    layout, as if line 159 called a different get_led_points_* function."""
    with redirect_stdout(io.StringIO()):
        pts = CLUSTER_LAYOUTS[name]()
    anim.led_points = pts
    anim.num_leds = len(pts)
    anim.num_per_group = anim.num_leds // 5
    anim.header = ["frame"]
    for i in range(anim.num_leds):
        anim.header.extend([f"light_{i}_Wt", f"light_{i}_R", f"light_{i}_G", f"light_{i}_B"])


def panel_layout():
    """Physical LED positions on the PCB (mm), in data-chain order."""
    with redirect_stdout(io.StringIO()):
        return helper_classes.get_led_points_pcb().tolist()


# ── Playback ─────────────────────────────────────────────────────────────────

_pool = None


def pool():
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))
    return _pool


# Interpolated frames kept ahead of the playhead (bounded by memory, not by count).
WINDOW_BYTES = 96 * 2**20
MAX_WINDOW = 256


class Playback:
    """Keyframes are generated in order on one thread (the colour tracks carry state).
    Interpolations run in the process pool, only for a window of transitions ahead of
    where the client is reading. Transition k fades selected record k into record k+1;
    a single-record selection is one hold frame."""

    def __init__(self, records, indices, frames_per_fade, seed=SEED):
        self.records = [records[i] for i in indices]
        self.n = len(self.records)
        self.animation_length = frames_per_fade
        self.total = max(1, self.n - 1)
        self.frames_per_transition = 1 if self.n == 1 else frames_per_fade
        bytes_per_transition = self.frames_per_transition * len(anim.led_points) * 8
        self.window = max(2, min(MAX_WINDOW, WINDOW_BYTES // bytes_per_transition))
        self.seed = seed
        self.keyframes = []
        self.results = {}
        self.submitted = {}
        self.window_start = 0
        self.cancelled = False
        self.error = None
        self.lock = threading.Lock()
        threading.Thread(target=self._run_safe, daemon=True).start()

    def _run_safe(self):
        try:
            self._run()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def _trace(self, rec):
        return helper_classes.Trace(
            np.array(rec["colors"][0], dtype=np.uint8), rec["counts"][0],
            traces_storing_mode="vaooo",
            supplemental_colors=np.array(rec["colors"][1:], dtype=np.uint8),
            supplemental_counts=rec["counts"][1:],
        )

    def _keyframe(self, k, tracks):
        kf = anim.generate_frame_from_trace_tracks(self._trace(self.records[k]), tracks)
        # frame[0] is int(time.time()) on the lamp, minutes apart. interpolate_two_frames
        # lerps over [t0, t1+1], so evenly spaced, far-apart stamps give an even fade
        # (record timestamps can repeat within an hour in archive.png).
        kf[0] = k * 3600
        return kf

    def _store(self, k, future):
        if future.cancelled():
            return
        if future.exception() is not None:
            self.error = f"transition {k}: {future.exception()}"
            return
        with self.lock:
            self.submitted.pop(k, None)
            if k >= self.window_start - 2:
                self.results[k] = future.result()
        self._schedule()

    def _schedule(self):
        """Submit interpolations for the transitions in the window that have keyframes."""
        if self.cancelled:
            return
        with self.lock:
            hi = min(self.window_start + self.window, len(self.keyframes) - 1, self.total)
            new = []
            for k in range(self.window_start, hi):
                if k in self.results or k in self.submitted:
                    continue
                if self.n == 1:
                    fut = pool().submit(workers.hold_job, self.keyframes[0], self.records[0])
                else:
                    fut = pool().submit(workers.transition_job, self.keyframes[k], self.keyframes[k + 1],
                                        self.animation_length, self.records[k], self.records[k + 1])
                self.submitted[k] = fut
                new.append((k, fut))
        for k, fut in new:
            fut.add_done_callback(lambda f, k=k: self._store(k, f))

    def _run(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        tracks = []
        with redirect_stdout(io.StringIO()):
            anim.init_tracks(self._trace(self.records[0]), tracks)
            self.keyframes.append(self._keyframe(0, tracks))
            if self.n == 1:
                self.keyframes.append(self.keyframes[0])  # lets _schedule see one "transition"
            for k in range(1, self.n):
                if self.cancelled:
                    return
                self.keyframes.append(self._keyframe(k, tracks))
                if len(self.keyframes) - 2 < self.window_start + self.window:
                    self._schedule()
        self._schedule()

    def ready(self):
        """First transition at or after the read position that isn't computed yet."""
        with self.lock:
            k = self.window_start
            while k in self.results:
                k += 1
        return k

    def batch(self, start, count):
        """Read transitions from `start`; this also moves the window to `start`."""
        with self.lock:
            self.window_start = start
            for k in [k for k in self.results if k < start - 2 or k >= start + self.window]:
                del self.results[k]
            items = []
            for k in range(start, start + count):
                if k not in self.results:
                    break
                items.append(self.results[k])
        self._schedule()
        return items

    def cancel(self):
        self.cancelled = True
        with self.lock:
            for f in self.submitted.values():
                f.cancel()

