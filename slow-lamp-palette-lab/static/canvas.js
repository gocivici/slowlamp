/**
 * canvas.js — p5.js rendering layer
 *
 * Owns the canvas and everything that draws to it.
 * Reads shared state set by app.js; writes imageBounds back so app.js
 * can map hover coordinates to pixel positions.
 *
 * Globals consumed (defined in app.js):
 *   uploadedImage, previewImage, originalImageCtx
 *   blurValue, showOriginal, subtractionMode, subtractFile
 *   imageBounds (written here, read in app.js for hover)
 *   imageTooltip (DOM element)
 */

// ── Helpers ───────────────────────────────────────────────────────────────

function shouldShowPreview() {
  return previewImage && !showOriginal && (blurValue > 0 || (subtractionMode && subtractFile));
}

// ── p5.js lifecycle ───────────────────────────────────────────────────────

function setup() {
  const frame = document.getElementById("canvasFrame");
  const canvas = createCanvas(frame.clientWidth, frame.clientHeight);
  canvas.parent("canvasFrame");
  canvas.elt.addEventListener("mousemove", handleCanvasHover);
  canvas.elt.addEventListener("mouseleave", hideTooltip);
  noLoop();
  background("#262624");
}

function windowResized() {
  const frame = document.getElementById("canvasFrame");
  resizeCanvas(frame.clientWidth, frame.clientHeight);
  redraw();
}

function draw() {
  background("#262624");
  imageBounds = null;

  const displaySource = shouldShowPreview() ? previewImage : uploadedImage;

  if (!displaySource) return;

  imageMode(CENTER);
  const scale = min(width / displaySource.width, height / displaySource.height, 1);
  const drawW = displaySource.width * scale;
  const drawH = displaySource.height * scale;
  const x = width / 2 - drawW / 2;
  const y = height / 2 - drawH / 2;

  image(displaySource, width / 2, height / 2, drawW, drawH);
  imageBounds = { x, y, w: drawW, h: drawH, imgW: displaySource.width, imgH: displaySource.height };
}

// ── Hover tooltip ─────────────────────────────────────────────────────────

function handleCanvasHover(event) {
  if (!imageBounds || !uploadedImage) {
    hideTooltip();
    return;
  }

  const rect = this.getBoundingClientRect();
  const localX = event.clientX - rect.left;
  const localY = event.clientY - rect.top;

  const inBounds =
    localX >= imageBounds.x &&
    localY >= imageBounds.y &&
    localX <= imageBounds.x + imageBounds.w &&
    localY <= imageBounds.y + imageBounds.h;

  if (!inBounds) {
    hideTooltip();
    return;
  }

  const relX = (localX - imageBounds.x) / imageBounds.w;
  const relY = (localY - imageBounds.y) / imageBounds.h;
  const pixelX = Math.min(imageBounds.imgW - 1, Math.max(0, Math.floor(relX * imageBounds.imgW)));
  const pixelY = Math.min(imageBounds.imgH - 1, Math.max(0, Math.floor(relY * imageBounds.imgH)));

  const displaySource = shouldShowPreview() ? previewImage : uploadedImage;

  const [r, g, b] = getAverageRgb(pixelX, pixelY, 3, displaySource);
  imageTooltip.textContent = `${r} ${g} ${b}`;
  imageTooltip.style.display = "block";
  imageTooltip.style.left = `${Math.min(Math.max(localX, 20), rect.width - 20)}px`;
  imageTooltip.style.top = `${Math.max(localY, 26)}px`;
}

/**
 * Average the RGB values in a (2*radius+1) neighbourhood around (cx, cy).
 * Uses originalImageCtx (an offscreen canvas) for the main image so pixel
 * reads are fast; falls back to p5's get() for other sources.
 */
function getAverageRgb(cx, cy, radius, sourceImage) {
  const half = Math.floor(radius / 2);
  let totalR = 0, totalG = 0, totalB = 0, count = 0;

  for (let dy = -half; dy <= half; dy++) {
    for (let dx = -half; dx <= half; dx++) {
      const x = Math.min(Math.max(cx + dx, 0), imageBounds.imgW - 1);
      const y = Math.min(Math.max(cy + dy, 0), imageBounds.imgH - 1);
      let pixel;
      if (sourceImage === uploadedImage && originalImageCtx) {
        pixel = originalImageCtx.getImageData(x, y, 1, 1).data;
      } else {
        pixel = sourceImage.get(x, y);
      }
      totalR += pixel[0];
      totalG += pixel[1];
      totalB += pixel[2];
      count++;
    }
  }

  return [
    Math.round(totalR / count),
    Math.round(totalG / count),
    Math.round(totalB / count),
  ];
}

function hideTooltip() {
  imageTooltip.style.display = "none";
}
