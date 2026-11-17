/**
 * app.js — sample loading, sampling, and the playback clock
 *
 * Selected records play back to back: each pair of records fades over `interval`
 * seconds at 60 fps. The server runs the production keyframe and OKLab
 * interpolation code; nothing here does colour math.
 *
 * Shared with sketch.js:
 *   sample, playback, selected (Set of record indices), view, diffuser,
 *   applyBrightness, frameDirty, currentDrive (Uint16Array, leds*4:
 *   brightness, r16, g16, b16), currentSwatch ({colors, counts}), currentRecord
 */

let sample = null;
let sampleFile = null;
let playback = null;
let selected = new Set();
let mode = "all";

let playing = false;
let view = "both";
let diffuser = true;
let applyBrightness = true;

let currentDrive = null;
let currentSwatch = null;
let currentRecord = -1;
let frameDirty = false;

// playback clock and frame cache
let clock = null;

const $ = (id) => document.getElementById(id);

function newClock() {
  return {
    t: 0,               // seconds of playback
    shown: -1,          // global frame index on the LEDs
    state: "paused",
    ready: 0,           // transitions the server has finished
    cache: new Map(),   // transition -> {frames, swatch}
    nextFetch: 0,
    fetching: false,
    generation: 0,
  };
}

// ── Loading ──────────────────────────────────────────────────────────────

async function loadSample(file) {
  setPlaying(false);
  $("sampleInfo").textContent = `Loading ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/load", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) {
    $("sampleInfo").textContent = data.error || "Load failed.";
    return;
  }

  sample = data;
  sampleFile = file;
  playback = null;
  selected = new Set();
  currentDrive = null;
  currentSwatch = null;
  currentRecord = -1;

  const r = sample.records;
  $("fromDate").value = r[0].label.slice(0, 10);
  $("toDate").value = r[r.length - 1].label.slice(0, 10);
  $("moonTime").value = r[r.length - 1].label.slice(0, 13).replace(" ", "T") + ":00";
  if (!$("interval").value) $("interval").value = sample.default_interval;

  renderSampleInfo();
  $("emptyState").style.display = "none";
  $("buildBtn").disabled = false;
  onSampleLoaded();
  buildPlayback();
}

function renderSampleInfo() {
  const r = sample.records;
  const s = sample.skipped;
  const kind = sample.kind === "cover" ? "cover archive" : "trace log";
  const notes = [];
  if (s.bad_encoding) notes.push(`${s.bad_encoding} badly encoded (blue-pinned run) removed`);
  if (s.empty) notes.push(`${s.empty} empty captures skipped`);
  if (s.unparsed) notes.push(`${s.unparsed} unparsed lines`);
  $("sampleInfo").innerHTML = [
    `<b>${sample.name}</b>`,
    `${kind} · ${r.length} records`,
    `${r[0].label} → ${r[r.length - 1].label}`,
    ...notes,
  ].join("<br>");
}

// ── Building a playback ──────────────────────────────────────────────────

async function buildPlayback() {
  if (!sample) return;
  setPlaying(false);
  const body = {
    mode,
    from: $("fromDate").value,
    to: $("toDate").value,
    hour: $("hourSelect").value,
    step: $("stepInput").value,
    moon_time: $("moonTime").value,
    interval: $("interval").value,
    layout: $("layoutSelect").value,
  };
  $("playbackInfo").textContent = "Building…";
  const res = await fetch("/api/playback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    $("playbackInfo").textContent = data.error || "Build failed.";
    return;
  }

  playback = data;
  selected = new Set(data.indices);
  const gen = clock ? clock.generation + 1 : 0;
  clock = newClock();
  clock.generation = gen;
  currentDrive = null;
  currentSwatch = null;
  currentRecord = data.indices[0];
  frameDirty = true;

  renderPlaybackInfo();
  $("playBtn").disabled = false;
  $("restartBtn").disabled = false;
  onSelectionChanged();
  pollProgress(gen);
}

function totalFrames() {
  return playback.transitions * playback.frames_per_transition;
}

function renderPlaybackInfo() {
  const p = playback;
  const secs = totalFrames() / p.fps;
  const lines =
    p.indices.length === 1
      ? ["1 record · shown as a still frame"]
      : [
          `${p.indices.length} records → ${p.transitions} fade${p.transitions === 1 ? "" : "s"}`,
          `${p.frames_per_transition} frames per fade at ${p.fps} fps · ${formatDuration(secs)} total`,
        ];
  if (p.leds !== sample.layout.length) {
    lines.push(`code drives ${p.leds} LEDs; the PCB has ${sample.layout.length}`);
  }
  if (p.info && p.info.illumination !== undefined) {
    lines.unshift(`moon illumination ${(p.info.illumination * 100).toFixed(1)}%`);
  }
  $("playbackInfo").innerHTML = lines.join("<br>");
}

async function pollProgress(gen) {
  while (clock && clock.generation === gen) {
    try {
      const p = await (await fetch("/api/progress")).json();
      if (!clock || clock.generation !== gen) return;
      if (p.error) {
        clock.state = `error: ${p.error}`;
        setPlaying(false);
        return;
      }
      clock.ready = p.ready;
      fetchAhead();
    } catch (e) {
      /* server restarting; keep polling */
    }
    await new Promise((r) => setTimeout(r, 400));
  }
}

function transitionBytes() {
  return playback.frames_per_transition * playback.leds * 8;
}

async function fetchAhead() {
  const c = clock;
  if (!c || c.fetching) return;
  const current = Math.max(0, Math.floor(Math.max(c.shown, 0) / playback.frames_per_transition));
  // drop what's behind us, keep a bounded window ahead
  for (const k of c.cache.keys()) if (k < current - 2) c.cache.delete(k);
  const aheadLimit = Math.max(2, Math.min(64, Math.floor((64 * 2 ** 20) / transitionBytes())));
  if (c.nextFetch >= c.ready || c.nextFetch >= current + aheadLimit) return;

  c.fetching = true;
  const start = c.nextFetch;
  const perRequest = Math.max(1, Math.floor((8 * 2 ** 20) / transitionBytes()));
  const count = Math.min(128, perRequest, c.ready - start, current + aheadLimit - start);
  try {
    const b = await (await fetch(`/api/batch?start=${start}&count=${count}`)).json();
    if (clock !== c) return;
    if (b.count === 0) {
      c.ready = start; // evicted on the server (e.g. after a rewind); wait for the next poll
      return;
    }
    const bytes = Uint8Array.from(atob(b.frames), (ch) => ch.charCodeAt(0));
    const all = new Uint16Array(bytes.buffer);
    const size = playback.frames_per_transition * playback.leds * 4;
    for (let i = 0; i < b.count; i++) {
      c.cache.set(start + i, { frames: all.subarray(i * size, (i + 1) * size), swatch: b.swatch[i] });
    }
    c.nextFetch = start + b.count;
  } finally {
    c.fetching = false;
  }
  if (c.nextFetch < c.ready) fetchAhead();
}

// ── Clock ────────────────────────────────────────────────────────────────

function showGlobalFrame(g) {
  const L = playback.frames_per_transition;
  const k = Math.floor(g / L);
  const f = g % L;
  const tr = clock.cache.get(k);
  if (!tr) return false;
  const size = playback.leds * 4;
  currentDrive = tr.frames.subarray(f * size, (f + 1) * size);
  const sw = tr.swatch;
  const si = L > 1 ? Math.round((f / (L - 1)) * (sw.colors.length - 1)) : 0;
  currentSwatch = { colors: sw.colors[si], counts: sw.counts[si] };
  currentRecord = playback.indices[Math.min(k + 1, playback.indices.length - 1)];
  clock.shown = g;
  frameDirty = true;
  return true;
}

// Runs on wall time, not draw(): browsers throttle animation frames.
let lastWall = performance.now();

function tickClock() {
  const now = performance.now();
  const dt = (now - lastWall) / 1000;
  lastWall = now;
  if (!playback || !clock) return;

  const total = totalFrames();
  if (clock.shown < 0 && clock.cache.has(0)) showGlobalFrame(0);
  if (!playing) {
    if (!clock.state.startsWith("error")) clock.state = "paused";
    return;
  }

  const L = playback.frames_per_transition;
  let t = clock.t + dt;
  let g = Math.floor(t * playback.fps);
  if (g >= total) {
    g = total - 1;
    t = total / playback.fps;
    clock.state = "ended";
    setPlaying(false);
  } else {
    clock.state = "playing";
  }

  const k = Math.floor(g / L);
  if (!clock.cache.has(k)) {
    // hold at the start of the transition that isn't ready yet
    t = (k * L) / playback.fps;
    g = k * L;
    clock.state = "preparing";
    fetchAhead();
  }
  clock.t = t;
  if (g !== clock.shown) showGlobalFrame(g);
  if (k + 300 > clock.nextFetch) fetchAhead();
}

// ── Status panel ─────────────────────────────────────────────────────────

function formatDuration(s) {
  if (!isFinite(s)) return "—";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const ms = `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  return h > 0 ? `${h}:${ms}` : ms;
}

function renderStatus() {
  if (!playback || !clock) {
    $("status").innerHTML = "";
    return;
  }
  const L = playback.frames_per_transition;
  const g = Math.max(clock.shown, 0);
  const k = Math.floor(g / L);
  const n = playback.indices.length;
  const recs = sample.records;
  const from = recs[playback.indices[k]];
  const to = recs[playback.indices[Math.min(k + 1, n - 1)]];
  const rows = [
    ["time", `${formatDuration(clock.t)} / ${formatDuration(totalFrames() / playback.fps)}`],
    ["state", clock.state],
    ["fade", n > 1 ? `${k + 1} / ${playback.transitions}` : "single record"],
    ["from", from.label],
    ["to", to.label],
    ["frame", `${(g % L) + 1} / ${L}`],
    ["ready ahead", `${Math.max(0, Math.min(clock.ready, playback.transitions) - k)} fades`],
  ];
  $("status").innerHTML = rows.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("");
}

// ── Controls ─────────────────────────────────────────────────────────────

function setPlaying(on) {
  playing = on;
  $("playBtn").textContent = on ? "Pause" : "Play";
}

function bindSegmented(groupId, attr, onChange) {
  const group = $(groupId);
  group.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    group.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
    onChange(btn.dataset[attr]);
  });
}

for (let h = 0; h < 24; h++) {
  const opt = document.createElement("option");
  opt.value = String(h);
  opt.textContent = `${String(h).padStart(2, "0")}:00`;
  $("hourSelect").appendChild(opt);
}

$("fileInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) loadSample(file);
  e.target.value = "";
});

$("buildBtn").addEventListener("click", buildPlayback);

function rewind() {
  if (!clock) return;
  clock.t = 0;
  clock.shown = -1;
  clock.cache.clear();
  clock.nextFetch = 0;
  clock.state = "paused";
  fetchAhead();
}

$("playBtn").addEventListener("click", () => {
  if (clock && clock.state === "ended") rewind();
  setPlaying(!playing);
});

$("restartBtn").addEventListener("click", rewind);

bindSegmented("modeGroup", "mode", (v) => {
  mode = v;
  $("filterPanel").hidden = v !== "filter";
  $("moonPanel").hidden = v !== "moon";
});
bindSegmented("viewGroup", "view", (v) => {
  view = v;
  layoutAreas();
});

$("diffuserToggle").addEventListener("change", (e) => {
  diffuser = e.target.checked;
  frameDirty = true;
});

$("brightnessToggle").addEventListener("change", (e) => {
  applyBrightness = e.target.checked;
  frameDirty = true;
});

setInterval(tickClock, 16);
setInterval(renderStatus, 250);
