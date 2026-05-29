/**
 * app.js — application state, event wiring, API calls, and UI helpers
 *
 * This file owns all shared state that canvas.js reads, and handles every
 * user interaction. The split is simple: if it touches p5.js, it lives in
 * canvas.js; everything else lives here.
 */

// ── State ─────────────────────────────────────────────────────────────────

let uploadedImage    = null;  // p5.Image — the main uploaded image
let imageFile        = null;  // File object for the main image
let originalImageCtx = null;  // OffscreenCanvas 2D context for fast pixel reads
let previewImage     = null;  // p5.Image — post-processing preview (blur / diff)
let previewDataUrl   = null;  // base64 data-URI of the preview
let imageBounds      = null;  // {x,y,w,h,imgW,imgH} — set by canvas.js each draw
let currentModel     = null;  // name of the selected extraction model
let blurValue        = 0.0;
let showOriginal     = false;
let blurDebounce     = null;
let subtractionMode      = false;
let subtractFile         = null;
let subtractionThreshold = 13;

// ── DOM References ────────────────────────────────────────────────────────

const modelList           = document.getElementById("modelList");
const imageInput          = document.getElementById("imageInput");
const blurSlider          = document.getElementById("blurSlider");
const blurValueText       = document.getElementById("blurValueText");
const previewToggle       = document.getElementById("previewToggle");
const canvasPlaceholder   = document.getElementById("canvasPlaceholder");
const swatchRow           = document.getElementById("swatchRow");
const errorMessage        = document.getElementById("errorMessage");
const imageTooltip        = document.getElementById("imageTooltip");
const subtractionToggle    = document.getElementById("subtractionToggle");
const subtractionInput     = document.getElementById("subtractionInput");
const subtractionUploadRow = document.getElementById("subtractionUploadRow");
const subtractionFileName  = document.getElementById("subtractionFileName");
const subtractionPreview   = document.getElementById("subtractionPreview");
const thresholdInput       = document.getElementById("thresholdInput");

// ── Blur ──────────────────────────────────────────────────────────────────

function setBlurValue(value) {
  blurValue = parseFloat(parseFloat(value).toFixed(1));
  blurValueText.textContent = blurValue.toFixed(1);
  updateSliderBackground();
  try { redraw(); } catch (e) { /* p5 may not be ready on first call */ }
}

function updateSliderBackground() {
  blurSlider.style.setProperty("--fill", `${(blurValue / 20) * 100}%`);
}

function scheduleExtract() {
  if (!imageFile) return;
  clearTimeout(blurDebounce);
  blurDebounce = setTimeout(extractPalette, 200);
}

blurSlider.addEventListener("input", (event) => {
  setBlurValue(parseFloat(event.target.value));
  scheduleExtract();
});

// ── Preview toggle ────────────────────────────────────────────────────────

previewToggle.addEventListener("click", () => {
  showOriginal = previewToggle.getAttribute("aria-pressed") !== "true";
  previewToggle.setAttribute("aria-pressed", showOriginal.toString());
  if (!showOriginal && blurValue > 0 && imageFile && !previewDataUrl) {
    extractPalette();
  } else {
    redraw();
  }
});

// ── Subtraction mode ──────────────────────────────────────────────────────

subtractionToggle.addEventListener("click", () => {
  subtractionMode = subtractionToggle.getAttribute("aria-pressed") !== "true";
  subtractionToggle.setAttribute("aria-pressed", subtractionMode.toString());
  subtractionUploadRow.style.display = subtractionMode ? "block" : "none";
  if (!subtractionMode) {
    subtractFile = null;
    subtractionInput.value = "";
    subtractionFileName.textContent = "Choose reference image…";
    subtractionPreview.src = "";
    subtractionPreview.style.display = "none";
    clearError();
  }
  if (imageFile) extractPalette();
});

thresholdInput.addEventListener("change", (event) => {
  subtractionThreshold = Math.min(100, Math.max(0, parseInt(event.target.value, 10) || 0));
  thresholdInput.value = subtractionThreshold;
  if (imageFile && subtractFile) extractPalette();
});

subtractionInput.addEventListener("change", (event) => {
  subtractFile = event.target.files[0] || null;
  subtractionFileName.textContent = subtractFile ? subtractFile.name : "Choose reference image…";
  if (subtractFile) {
    const url = URL.createObjectURL(subtractFile);
    subtractionPreview.onload = () => URL.revokeObjectURL(url);
    subtractionPreview.src = url;
    subtractionPreview.style.display = "block";
  } else {
    subtractionPreview.src = "";
    subtractionPreview.style.display = "none";
  }
  if (imageFile) extractPalette();
});

// ── Main image upload ─────────────────────────────────────────────────────

imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  imageFile = file;

  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result;

    // Build an offscreen canvas for fast per-pixel reads in the hover handler.
    const tempImage = new Image();
    tempImage.onload = () => {
      const offscreen = document.createElement("canvas");
      offscreen.width = tempImage.naturalWidth;
      offscreen.height = tempImage.naturalHeight;
      originalImageCtx = offscreen.getContext("2d");
      originalImageCtx.drawImage(tempImage, 0, 0);
    };
    tempImage.src = dataUrl;

    uploadedImage = loadImage(dataUrl, () => {
      canvasPlaceholder.style.display = "none";
      redraw();
    });
  };
  reader.readAsDataURL(file);
  extractPalette();
});

// ── API ───────────────────────────────────────────────────────────────────

async function fetchModels() {
  try {
    const response = await fetch("/api/models");
    const models = await response.json();
    populateModels(models);
  } catch {
    showError("Unable to load models.");
  }
}

function populateModels(models) {
  if (!models.length) {
    modelList.innerHTML = "<div class='error-message'>No models available.</div>";
    return;
  }

  const previousModel = currentModel;
  models.sort();
  const selectedModel =
    previousModel && models.includes(previousModel) ? previousModel : models[0];

  modelList.innerHTML = "";
  models.forEach((name) => {
    const label = document.createElement("label");
    label.className = "model-option";
    label.dataset.name = name;

    const input = document.createElement("input");
    input.type = "radio";
    input.name = "model";
    input.value = name;
    input.checked = name === selectedModel;
    if (input.checked) currentModel = name;
    input.addEventListener("change", (e) => {
      currentModel = e.target.value;
      if (imageFile) extractPalette();
    });

    const radioVis = document.createElement("span");
    radioVis.className = "radio-visual";

    const span = document.createElement("span");
    span.textContent = name;

    label.appendChild(input);
    label.appendChild(radioVis);
    label.appendChild(span);
    modelList.appendChild(label);
  });

  if (imageFile && previousModel && selectedModel !== previousModel) {
    extractPalette();
  }
}

async function extractPalette() {
  if (!currentModel || !imageFile) return;

  if (subtractionMode && !subtractFile) {
    showError("Upload a reference image to enable subtraction.");
    return;
  }

  setBusy(true);
  clearError();

  try {
    const form = new FormData();
    form.append("file", imageFile);
    form.append("model", currentModel);
    form.append("blur", blurValue.toString());
    if (subtractionMode && subtractFile) {
      form.append("subtract_file", subtractFile);
      form.append("threshold", subtractionThreshold.toString());
    }

    const response = await fetch("/api/extract", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Extraction failed");

    if (data.warning) showError(data.warning);
    updateSwatches(data.palette);

    if (data.preview) {
      previewDataUrl = data.preview;
      previewImage = loadImage(previewDataUrl, () => {
        if (!showOriginal) redraw();
      });
    } else {
      previewDataUrl = null;
      previewImage = null;
    }
  } catch (err) {
    showError(err.message);
    updateSwatches([]);
  } finally {
    setBusy(false);
  }
}

// ── Swatches ──────────────────────────────────────────────────────────────

function updateSwatches(palette) {
  Array.from(swatchRow.children).forEach((block, index) => {
    const colorBox = block.querySelector(".swatch-color");
    const label    = block.querySelector(".swatch-hex");
    if (palette[index]) {
      const [r, g, b] = palette[index];
      const rgbText = `${r} ${g} ${b}`;
      const hex = rgbToHex(r, g, b);
      colorBox.style.background = hex;
      label.textContent = rgbText;
      block.style.cursor = "pointer";
      label.onclick = () => copyToClipboard(label, rgbText);
    } else {
      colorBox.style.background = "var(--placeholder)";
      label.textContent = "R G B";
      block.style.cursor = "default";
      label.onclick = null;
    }
  });
}

// ── Utilities ─────────────────────────────────────────────────────────────

function rgbToHex(r, g, b) {
  const h = (v) => v.toString(16).padStart(2, "0");
  return `#${h(r)}${h(g)}${h(b)}`.toUpperCase();
}

async function copyToClipboard(label, text) {
  try {
    await navigator.clipboard.writeText(text);
    label.classList.add("flash");
    setTimeout(() => label.classList.remove("flash"), 600);
  } catch { /* clipboard access denied */ }
}

function setBusy(state) {
  swatchRow.classList.toggle("dimmed", state);
  blurSlider.classList.toggle("busy", state);
}

function showError(message) {
  errorMessage.textContent = message;
}

function clearError() {
  errorMessage.textContent = "";
}

// ── Init ──────────────────────────────────────────────────────────────────

window.addEventListener("load", () => {
  setBlurValue(0.0);
  fetchModels();
  setInterval(fetchModels, 8000);
});

window.addEventListener("focus", fetchModels);
