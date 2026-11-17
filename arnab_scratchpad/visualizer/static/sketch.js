/**
 * sketch.js — p5.js rendering layer
 *
 * Panel view: the LEDs at their PCB ring positions (get_led_points_pcb(), mm, in
 * data-chain order), coloured from the exact uint16 values the lamp sends over SPI,
 * optionally blurred to approximate the diffuser.
 * Swatch view: the current record's 5 colours cross-faded, plus a strip of every
 * record in the sample with a playhead.
 *
 * Reads globals from app.js; the lamp clock itself runs on a timer in app.js.
 */

const BG = "#262624";
const MUTED = "#B5B3AC";
const BORDER = "#3A3937";
const ACCENT = "#D97757";
const LAMP_RADIUS_MM = 115; // 23 cm diffuser
const DIFFUSER_BLUR_MM = 14;
const DIFFUSER_GAIN = 4; // blurring spreads the light thin; add it back up
const DRIVE_FULL = 255 * 255; // interpolate_two_frames sends sRGB 0-255 scaled by 255
const MAX_BRIGHTNESS = 6; // clip ceiling in interpolate_two_frames

const SWATCH_NAMES = ["vibrant", "ac1", "ac2", "ac3", "ac4"];

let areas = {};
let ledPg = null;
let glowPg = null;
let stripPg = null;
let ledScreen = []; // [x, y] per LED in canvas coords
let ledDot = 0;

function setup() {
  const frame = document.getElementById("canvasFrame");
  const c = createCanvas(frame.clientWidth, frame.clientHeight);
  c.parent("canvasFrame");
  c.elt.addEventListener("mouseleave", hideTooltip);
  pixelDensity(Math.min(2, window.devicePixelRatio || 1));
  layoutAreas();
}

function windowResized() {
  const frame = document.getElementById("canvasFrame");
  resizeCanvas(frame.clientWidth, frame.clientHeight);
  layoutAreas();
}

function onSampleLoaded() {
  layoutAreas();
}

function draw() {
  tickClock();
  background(BG);
  if (!sample) return;

  if (frameDirty) {
    renderLeds();
    frameDirty = false;
  }

  if (areas.lamp) drawLamp();
  if (areas.swatch) drawSwatch();
  if (areas.strip) drawStrip();
}

// ── Layout ───────────────────────────────────────────────────────────────

function layoutAreas() {
  const pad = 24;
  const w = width - pad * 2;
  const h = height - pad * 2;
  areas = {};

  if (view === "panel") {
    areas.lamp = { x: pad, y: pad, w, h };
  } else if (view === "swatch") {
    areas.swatch = { x: pad, y: pad, w, h: h * 0.42 };
    areas.strip = { x: pad, y: pad + h * 0.42 + 36, w, h: h * 0.58 - 36 };
  } else {
    const lw = w * 0.52;
    const rw = w - lw - pad;
    areas.lamp = { x: pad, y: pad, w: lw, h };
    areas.swatch = { x: pad + lw + pad, y: pad, w: rw, h: h * 0.42 };
    areas.strip = { x: pad + lw + pad, y: pad + h * 0.42 + 36, w: rw, h: h * 0.58 - 36 };
  }

  if (sample && areas.lamp) buildLampGeometry();
  if (sample && areas.strip) buildStrip();
  frameDirty = true;
}

function buildLampGeometry() {
  const a = areas.lamp;
  const pts = sample.layout;
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const scale = (Math.min(a.w, a.h) / (LAMP_RADIUS_MM * 2)) * 0.94;

  a.cx = a.x + a.w / 2;
  a.cy = a.y + a.h / 2;
  a.scale = scale;
  ledScreen = pts.map((p) => [a.cx + (p[0] - cx) * scale, a.cy + (p[1] - cy) * scale]);
  ledDot = 5.5 * scale; // 5050-size LEDs, ~9.6 mm apart on the outer ring

  const d = pixelDensity();
  ledPg = createGraphics(width, height);
  ledPg.pixelDensity(d);
  glowPg = createGraphics(width, height);
  glowPg.pixelDensity(d);
}

// ── LED colour ───────────────────────────────────────────────────────────

const srgbToLinear = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
const linearToSrgb = (v) => (v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);

// HD108 drive values -> on-screen sRGB 0-255
function driveToScreen(b, r16, g16, b16) {
  const out = [r16, g16, b16].map((x) => Math.min(1, x / DRIVE_FULL));
  if (!applyBrightness) return out.map((v) => v * 255);
  const gain = Math.min(1, b / MAX_BRIGHTNESS);
  return out.map((v) => linearToSrgb(srgbToLinear(v) * gain) * 255);
}

function renderLeds() {
  if (!areas.lamp || !ledPg || !currentDrive) return;
  ledPg.clear();
  ledPg.noStroke();
  for (let i = 0; i < ledScreen.length; i++) {
    const o = i * 4;
    const [r, g, b] = driveToScreen(currentDrive[o], currentDrive[o + 1], currentDrive[o + 2], currentDrive[o + 3]);
    ledPg.fill(r, g, b);
    ledPg.circle(ledScreen[i][0], ledScreen[i][1], ledDot);
  }

  glowPg.clear();
  const ctx = glowPg.drawingContext;
  // the diffuser smears each LED over a few centimetres
  ctx.filter = `blur(${Math.max(4, DIFFUSER_BLUR_MM * areas.lamp.scale)}px)`;
  glowPg.image(ledPg, 0, 0, width, height);
  ctx.filter = "none";
}

// ── Lamp panel ───────────────────────────────────────────────────────────

function drawLamp() {
  const a = areas.lamp;
  const r = LAMP_RADIUS_MM * a.scale;

  noStroke();
  fill(12);
  circle(a.cx, a.cy, r * 2);

  if (currentDrive && ledPg) {
    drawingContext.save();
    drawingContext.beginPath();
    drawingContext.arc(a.cx, a.cy, r, 0, Math.PI * 2);
    drawingContext.clip();
    if (diffuser) {
      blendMode(ADD);
      for (let i = 0; i < DIFFUSER_GAIN; i++) image(glowPg, 0, 0, width, height);
      blendMode(BLEND);
    } else {
      image(ledPg, 0, 0, width, height);
    }
    drawingContext.restore();
  } else {
    fill(MUTED);
    textAlign(CENTER, CENTER);
    textSize(12);
    text(playback ? "preparing…" : "", a.cx, a.cy);
  }

  noFill();
  stroke(BORDER);
  strokeWeight(1);
  circle(a.cx, a.cy, r * 2);
}

// ── Swatch ───────────────────────────────────────────────────────────────

function drawSwatch() {
  const a = areas.swatch;
  noStroke();
  fill(MUTED);
  textSize(11);
  textAlign(LEFT, TOP);
  text("CURRENT RECORD · OKLab cross-fade", a.x, a.y);

  const barY = a.y + 22;
  const barH = a.h - 22 - 44;

  if (!currentSwatch) {
    fill(BORDER);
    rect(a.x, barY, a.w, barH, 8);
    return;
  }

  const counts = currentSwatch.counts;
  const total = counts.reduce((s, c) => s + c, 0);
  if (total === 0) {
    fill(BORDER);
    rect(a.x, barY, a.w, barH, 8);
    fill(MUTED);
    textAlign(CENTER, CENTER);
    text("no pixel counts", a.x + a.w / 2, barY + barH / 2);
    return;
  }

  let x = a.x;
  drawingContext.save();
  drawingContext.beginPath();
  drawingContext.roundRect(a.x, barY, a.w, barH, 8);
  drawingContext.clip();
  const segs = [];
  for (let i = 0; i < 5; i++) {
    const w = (counts[i] / total) * a.w;
    fill(...currentSwatch.colors[i]);
    rect(x, barY, w + 0.5, barH);
    segs.push([x, w]);
    x += w;
  }
  drawingContext.restore();

  textSize(10);
  textAlign(LEFT, TOP);
  for (let i = 0; i < 5; i++) {
    const [sx, w] = segs[i];
    if (w < 46) continue;
    fill(i === 0 ? ACCENT : MUTED);
    text(SWATCH_NAMES[i], sx + 4, barY + barH + 6);
    fill(MUTED);
    text(currentSwatch.colors[i].join(","), sx + 4, barY + barH + 20);
  }
}

// ── Record strip ─────────────────────────────────────────────────────────

function buildStrip() {
  const a = areas.strip;
  const recs = sample.records;
  stripPg = createGraphics(Math.max(1, Math.floor(a.w)), Math.max(1, Math.floor(a.h - 20)));
  stripPg.pixelDensity(pixelDensity());
  stripPg.noStroke();
  stripPg.background(BG);

  const colW = stripPg.width / recs.length;
  const dimUnselected = selected.size > 0 && selected.size < recs.length;
  recs.forEach((r, k) => {
    const x = k * colW;
    const total = r.counts.reduce((s, c) => s + c, 0) || 1;
    let y = 0;
    for (let i = 0; i < 5; i++) {
      const h = (r.counts[i] / total) * stripPg.height;
      stripPg.fill(...r.colors[i]);
      stripPg.rect(x, y, colW + 0.5, h + 0.5);
      y += h;
    }
    if (dimUnselected && !selected.has(k)) {
      stripPg.fill(38, 38, 36, 215);
      stripPg.rect(x, 0, colW + 0.5, stripPg.height);
    }
  });
}

function onSelectionChanged() {
  if (sample && areas.strip) buildStrip();
}

function drawStrip() {
  const a = areas.strip;
  if (!stripPg) return;
  const n = sample.records.length;

  noStroke();
  fill(MUTED);
  textSize(11);
  textAlign(LEFT, BOTTOM);
  const sel = selected.size && selected.size < n ? ` · ${selected.size} selected` : "";
  text(`ALL RECORDS · ${n}${sel}`, a.x, a.y - 8);

  image(stripPg, a.x, a.y, stripPg.width, stripPg.height);

  // sparse selections (e.g. moon matches) are hard to see as columns; mark them
  if (selected.size && selected.size <= 200) {
    fill(ACCENT);
    for (const k of selected) {
      const x = a.x + ((k + 0.5) / n) * stripPg.width;
      triangle(x - 3, a.y - 6, x + 3, a.y - 6, x, a.y - 1);
    }
  }

  // month ticks
  const recs = sample.records;
  textSize(10);
  textAlign(LEFT, TOP);
  let lastMonth = "";
  recs.forEach((r, k) => {
    const month = r.label.slice(0, 7);
    if (month === lastMonth) return;
    lastMonth = month;
    const x = a.x + (k / n) * stripPg.width;
    stroke(BORDER);
    line(x, a.y + stripPg.height, x, a.y + stripPg.height + 4);
    noStroke();
    fill(MUTED);
    text(month, x + 2, a.y + stripPg.height + 5);
  });

  // record currently being faded into
  if (currentRecord >= 0) {
    const x = a.x + ((currentRecord + 0.5) / n) * stripPg.width;
    stroke(ACCENT);
    strokeWeight(2);
    line(x, a.y - 4, x, a.y + stripPg.height + 4);
    strokeWeight(1);
  }
  noStroke();
}

// ── Hover ────────────────────────────────────────────────────────────────

function mouseMoved() {
  if (!sample) return hideTooltip();

  if (areas.lamp && currentDrive && ledScreen.length) {
    let best = -1;
    let bestD = (ledDot * 0.8) ** 2;
    for (let i = 0; i < ledScreen.length; i++) {
      const d = (ledScreen[i][0] - mouseX) ** 2 + (ledScreen[i][1] - mouseY) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best >= 0) {
      const o = best * 4;
      const v = Array.from(currentDrive.subarray(o, o + 4));
      const scr = driveToScreen(...v).map((x) => Math.round(x));
      return showTooltip(
        `LED ${best}\nbrightness ${v[0]} / 31\nR16 ${v[1]}  G16 ${v[2]}  B16 ${v[3]}\nscreen rgb(${scr.join(", ")})`
      );
    }
  }

  const s = areas.strip;
  if (s && stripPg && mouseX >= s.x && mouseX < s.x + stripPg.width && mouseY >= s.y && mouseY < s.y + stripPg.height) {
    const k = Math.floor(((mouseX - s.x) / stripPg.width) * sample.records.length);
    const r = sample.records[k];
    if (r) {
      const lines = r.colors.map((c, i) => `${SWATCH_NAMES[i].padEnd(7)} (${c.join(", ")})  #${r.counts[i]}`);
      const tag = selected.has(k) ? " · selected" : "";
      return showTooltip(`record ${k + 1} · ${r.label}${tag}\n${lines.join("\n")}`);
    }
  }
  hideTooltip();
}

function showTooltip(textContent) {
  const tip = document.getElementById("tooltip");
  tip.textContent = textContent;
  tip.style.display = "block";
  const x = Math.min(mouseX + 14, width - tip.offsetWidth - 8);
  const y = Math.min(mouseY + 14, height - tip.offsetHeight - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function hideTooltip() {
  document.getElementById("tooltip").style.display = "none";
}
