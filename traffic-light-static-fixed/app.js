"use strict";

/*
  Traffic Light Tracker — high-resolution tiled inference

  The model still receives 640×640 inputs, but instead of shrinking the entire
  camera frame into one input, the app scans overlapping crops from the upper
  part of the frame. This makes small, distant traffic lights occupy more pixels.
*/

const CONFIG = {
  modelUrl: "./models/traffic-light.onnx",
  inputSize: 640,
  trafficLightClassId: 9,

  // Generic COCO models often need a lower threshold for distant traffic lights.
  confidenceThreshold: 0.16,
  nmsThreshold: 0.42,
  maxDetections: 20,

  // A new tiled cycle starts no more often than this. Actual cycles may take longer.
  inferenceIntervalMs: 180,

  // Scan only the upper part of the road scene, where signals normally appear.
  upperRoiRatio: 0.72,

  // Tracking settings.
  trackIouThreshold: 0.16,
  trackMaxMisses: 4,
  trackSmoothing: 0.62
};

let camera;
let overlay;
let overlayContext;
let inputCanvas;
let inputContext;
let stage;
let emptyState;

let startButton;
let stopButton;
let scanMode;
let showTiles;
let zoomControl;
let zoomSlider;
let zoomValue;

let secureBadge;
let statusElement;
let runtimeElement;
let resolutionElement;
let latencyElement;
let tileCountElement;
let countElement;

let cameraStream = null;
let cameraTrack = null;
let inferenceSession = null;
let inputName = null;
let outputName = null;

let running = false;
let inferenceBusy = false;
let lastInferenceStart = 0;
let currentTiles = [];
let tracks = [];
let nextTrackId = 1;

window.addEventListener("DOMContentLoaded", initializeApplication);
window.addEventListener("error", handleGlobalError);
window.addEventListener("unhandledrejection", handleUnhandledRejection);

function initializeApplication() {
  camera = requiredElement("camera");
  overlay = requiredElement("overlay");
  inputCanvas = requiredElement("inputCanvas");
  stage = requiredElement("stage");
  emptyState = requiredElement("emptyState");

  startButton = requiredElement("startButton");
  stopButton = requiredElement("stopButton");
  scanMode = requiredElement("scanMode");
  showTiles = requiredElement("showTiles");
  zoomControl = requiredElement("zoomControl");
  zoomSlider = requiredElement("zoomSlider");
  zoomValue = requiredElement("zoomValue");

  secureBadge = requiredElement("secureBadge");
  statusElement = requiredElement("status");
  runtimeElement = requiredElement("runtime");
  resolutionElement = requiredElement("resolution");
  latencyElement = requiredElement("latency");
  tileCountElement = requiredElement("tileCount");
  countElement = requiredElement("count");

  overlayContext = overlay.getContext("2d");
  inputContext = inputCanvas.getContext("2d", { willReadFrequently: true });

  if (!overlayContext || !inputContext) {
    throw new Error("Unable to create a canvas context.");
  }

  startButton.addEventListener("click", startApplication);
  stopButton.addEventListener("click", stopApplication);
  scanMode.addEventListener("change", refreshTiles);
  showTiles.addEventListener("change", drawOverlay);
  zoomSlider.addEventListener("input", applyCameraZoom);

  window.addEventListener("resize", resizeOverlay);
  window.addEventListener("pagehide", stopApplication);

  configureSecureContextBadge();
  configureOnnxRuntime();
  statusElement.textContent = "Waiting to start";
}

function configureSecureContextBadge() {
  if (window.isSecureContext) {
    secureBadge.textContent = "HTTPS ready";
    secureBadge.classList.add("good");
  } else {
    secureBadge.textContent = "HTTPS required";
    secureBadge.classList.add("bad");
  }
}

function configureOnnxRuntime() {
  if (!window.ort) {
    runtimeElement.textContent = "CDN not loaded";
    return;
  }

  // One thread avoids cross-origin-isolation requirements and is reliable on iPhone.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.1/dist/";

  runtimeElement.textContent = "Ready to load";
}

async function startApplication() {
  startButton.disabled = true;

  try {
    validateBrowserSupport();

    statusElement.textContent = "Loading ONNX model";
    await loadModel();

    statusElement.textContent = "Requesting rear camera";
    await openCamera();

    running = true;
    stopButton.disabled = false;
    emptyState.hidden = true;
    statusElement.textContent = "Tracking";

    requestAnimationFrame(animationLoop);
  } catch (error) {
    console.error(error);
    statusElement.textContent = describeStartupError(error);
    startButton.disabled = false;
    stopButton.disabled = true;
    closeCamera();
  }
}

function stopApplication() {
  running = false;
  inferenceBusy = false;
  closeCamera();

  tracks = [];
  currentTiles = [];
  clearOverlay();

  if (emptyState) emptyState.hidden = false;
  if (statusElement) statusElement.textContent = "Stopped";
  if (resolutionElement) resolutionElement.textContent = "—";
  if (latencyElement) latencyElement.textContent = "—";
  if (tileCountElement) tileCountElement.textContent = "—";
  if (countElement) countElement.textContent = "0";
  if (startButton) startButton.disabled = false;
  if (stopButton) stopButton.disabled = true;
  if (zoomControl) zoomControl.hidden = true;
}

function validateBrowserSupport() {
  if (!window.ort) {
    throw new Error(
      "ONNX Runtime failed to load. Check the ort.min.js request and internet connection."
    );
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser does not support camera access.");
  }

  if (!window.isSecureContext) {
    throw new Error("Camera access requires HTTPS or localhost.");
  }
}

async function loadModel() {
  if (inferenceSession) return;

  runtimeElement.textContent = "Loading WASM";

  inferenceSession = await ort.InferenceSession.create(CONFIG.modelUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all"
  });

  inputName = inferenceSession.inputNames[0];
  outputName = inferenceSession.outputNames[0];

  if (!inputName || !outputName) {
    throw new Error("The ONNX model has no usable input or output.");
  }

  runtimeElement.textContent = "WASM";
  await warmUpModel();
}

async function warmUpModel() {
  const valueCount = 3 * CONFIG.inputSize * CONFIG.inputSize;
  const tensor = new ort.Tensor(
    "float32",
    new Float32Array(valueCount),
    [1, 3, CONFIG.inputSize, CONFIG.inputSize]
  );

  await inferenceSession.run({ [inputName]: tensor });
}

async function openCamera() {
  cameraStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      frameRate: { ideal: 30, max: 30 }
    }
  });

  cameraTrack = cameraStream.getVideoTracks()[0] || null;
  camera.srcObject = cameraStream;

  await camera.play();
  await waitForVideoMetadata();

  stage.style.aspectRatio = `${camera.videoWidth} / ${camera.videoHeight}`;
  resolutionElement.textContent = `${camera.videoWidth} × ${camera.videoHeight}`;

  resizeOverlay();
  refreshTiles();
  configureZoomControl();

  if (cameraTrack) {
    console.log("Camera settings:", cameraTrack.getSettings?.());
  }
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
  }

  cameraStream = null;
  cameraTrack = null;

  if (camera) camera.srcObject = null;
}

function waitForVideoMetadata() {
  if (camera.videoWidth && camera.videoHeight) return Promise.resolve();

  return new Promise((resolve) => {
    camera.addEventListener("loadedmetadata", resolve, { once: true });
  });
}

function configureZoomControl() {
  zoomControl.hidden = true;

  if (!cameraTrack?.getCapabilities) return;

  let capabilities;
  let settings;

  try {
    capabilities = cameraTrack.getCapabilities();
    settings = cameraTrack.getSettings?.() || {};
  } catch (error) {
    console.warn("Camera capabilities unavailable:", error);
    return;
  }

  const zoom = capabilities.zoom;

  if (!zoom || !Number.isFinite(zoom.min) || !Number.isFinite(zoom.max)) {
    return;
  }

  zoomSlider.min = String(zoom.min);
  zoomSlider.max = String(zoom.max);
  zoomSlider.step = String(zoom.step || 0.1);
  zoomSlider.value = String(settings.zoom ?? zoom.min);
  zoomValue.textContent = `${Number(zoomSlider.value).toFixed(1)}×`;
  zoomControl.hidden = false;
}

async function applyCameraZoom() {
  if (!cameraTrack) return;

  const requestedZoom = Number(zoomSlider.value);
  zoomValue.textContent = `${requestedZoom.toFixed(1)}×`;

  try {
    await cameraTrack.applyConstraints({
      advanced: [{ zoom: requestedZoom }]
    });
  } catch (error) {
    console.warn("Camera zoom was rejected:", error);
  }
}

async function animationLoop(timestamp) {
  if (!running) return;

  drawOverlay();

  const enoughTimeElapsed =
    timestamp - lastInferenceStart >= CONFIG.inferenceIntervalMs;

  const cameraReady =
    camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA;

  if (!inferenceBusy && enoughTimeElapsed && cameraReady) {
    lastInferenceStart = timestamp;
    inferenceBusy = true;

    runTiledDetection()
      .catch((error) => {
        console.error(error);
        statusElement.textContent = `Detection error: ${error.message}`;
      })
      .finally(() => {
        inferenceBusy = false;
      });
  }

  requestAnimationFrame(animationLoop);
}

async function runTiledDetection() {
  const cycleStarted = performance.now();
  currentTiles = createScanTiles(
    camera.videoWidth,
    camera.videoHeight,
    scanMode.value
  );

  tileCountElement.textContent = String(currentTiles.length);

  const allDetections = [];

  // Run tiles sequentially to keep memory use controlled on phones.
  for (const tile of currentTiles) {
    const preprocessing = preprocessTile(camera, tile);
    const outputMap = await inferenceSession.run({
      [inputName]: preprocessing.tensor
    });

    const output = outputMap[outputName];

    if (!output) {
      throw new Error(`Model output "${outputName}" was not returned.`);
    }

    const tileDetections = parseYoloOutput(output, preprocessing);
    allDetections.push(...tileDetections);
  }

  // Remove duplicates created where adjacent tiles overlap.
  const mergedDetections = nonMaximumSuppression(
    allDetections,
    CONFIG.nmsThreshold,
    CONFIG.maxDetections
  );

  tracks = updateTracks(tracks, mergedDetections);

  latencyElement.textContent = `${Math.round(performance.now() - cycleStarted)} ms`;
  countElement.textContent = String(
    tracks.filter((track) => track.misses === 0).length
  );
}

function createScanTiles(sourceWidth, sourceHeight, mode) {
  if (!sourceWidth || !sourceHeight) return [];

  if (mode === "full") {
    return [
      {
        id: 1,
        x: 0,
        y: 0,
        width: sourceWidth,
        height: sourceHeight
      }
    ];
  }

  const desiredCount = mode === "two" ? 2 : 3;
  const roiHeight = Math.max(
    1,
    Math.round(sourceHeight * CONFIG.upperRoiRatio)
  );

  // Square crops preserve more detail when resized to the square model input.
  const tileSize = Math.min(sourceWidth, roiHeight);

  if (sourceWidth <= tileSize || desiredCount === 1) {
    return [
      {
        id: 1,
        x: 0,
        y: 0,
        width: sourceWidth,
        height: roiHeight
      }
    ];
  }

  const maxX = sourceWidth - tileSize;
  const step = maxX / (desiredCount - 1);
  const tiles = [];

  for (let index = 0; index < desiredCount; index += 1) {
    tiles.push({
      id: index + 1,
      x: Math.round(step * index),
      y: 0,
      width: tileSize,
      height: tileSize
    });
  }

  return tiles;
}

function refreshTiles() {
  if (!camera?.videoWidth || !camera?.videoHeight) return;

  currentTiles = createScanTiles(
    camera.videoWidth,
    camera.videoHeight,
    scanMode.value
  );

  tileCountElement.textContent = String(currentTiles.length);
  drawOverlay();
}

function preprocessTile(videoElement, tile) {
  const size = CONFIG.inputSize;
  const scale = Math.min(size / tile.width, size / tile.height);
  const drawWidth = Math.round(tile.width * scale);
  const drawHeight = Math.round(tile.height * scale);
  const padX = Math.floor((size - drawWidth) / 2);
  const padY = Math.floor((size - drawHeight) / 2);

  inputContext.fillStyle = "#000000";
  inputContext.fillRect(0, 0, size, size);

  inputContext.drawImage(
    videoElement,
    tile.x,
    tile.y,
    tile.width,
    tile.height,
    padX,
    padY,
    drawWidth,
    drawHeight
  );

  const rgba = inputContext.getImageData(0, 0, size, size).data;
  const pixelCount = size * size;
  const chw = new Float32Array(pixelCount * 3);

  for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex += 1) {
    const rgbaIndex = pixelIndex * 4;
    chw[pixelIndex] = rgba[rgbaIndex] / 255;
    chw[pixelCount + pixelIndex] = rgba[rgbaIndex + 1] / 255;
    chw[pixelCount * 2 + pixelIndex] = rgba[rgbaIndex + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", chw, [1, 3, size, size]),
    scale,
    padX,
    padY,
    tile,
    sourceWidth: videoElement.videoWidth,
    sourceHeight: videoElement.videoHeight
  };
}

function parseYoloOutput(output, metadata) {
  const dimensions = output.dims;
  const data = output.data;

  if (!Array.isArray(dimensions) || dimensions.length !== 3 || dimensions[0] !== 1) {
    throw new Error(`Unexpected model output shape: ${JSON.stringify(dimensions)}`);
  }

  // End-to-end export: [1, N, 6] or [1, 6, N].
  if (dimensions[2] === 6 || dimensions[1] === 6) {
    return parseEndToEndOutput(data, dimensions, metadata);
  }

  // Standard Ultralytics output: [1, 84, N] or [1, N, 84].
  const channelFirst = dimensions[1] < dimensions[2];
  const channelCount = channelFirst ? dimensions[1] : dimensions[2];
  const candidateCount = channelFirst ? dimensions[2] : dimensions[1];

  const isYoloV5Style = channelCount >= 85;
  const classOffset = isYoloV5Style ? 5 : 4;
  const trafficLightChannel = classOffset + CONFIG.trafficLightClassId;

  if (trafficLightChannel >= channelCount) {
    throw new Error(
      `Traffic-light class ${CONFIG.trafficLightClassId} is not present in a ` +
      `${channelCount}-channel model output.`
    );
  }

  const detections = [];

  for (let candidateIndex = 0; candidateIndex < candidateCount; candidateIndex += 1) {
    const value = (channelIndex) =>
      channelFirst
        ? data[channelIndex * candidateCount + candidateIndex]
        : data[candidateIndex * channelCount + channelIndex];

    const objectness = isYoloV5Style ? value(4) : 1;
    const score = objectness * value(trafficLightChannel);

    if (score < CONFIG.confidenceThreshold) continue;

    const centerX = value(0);
    const centerY = value(1);
    const width = value(2);
    const height = value(3);

    const videoBox = modelBoxToVideoBox(
      {
        x: centerX - width / 2,
        y: centerY - height / 2,
        width,
        height
      },
      metadata
    );

    if (videoBox.width >= 2 && videoBox.height >= 2) {
      detections.push({ box: videoBox, score });
    }
  }

  return detections;
}

function parseEndToEndOutput(data, dimensions, metadata) {
  const rowMajor = dimensions[2] === 6;
  const rowCount = rowMajor ? dimensions[1] : dimensions[2];
  const detections = [];

  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    const value = (columnIndex) =>
      rowMajor
        ? data[rowIndex * 6 + columnIndex]
        : data[columnIndex * rowCount + rowIndex];

    const score = value(4);
    const classId = Math.round(value(5));

    if (
      classId !== CONFIG.trafficLightClassId ||
      score < CONFIG.confidenceThreshold
    ) {
      continue;
    }

    const videoBox = modelBoxToVideoBox(
      {
        x: value(0),
        y: value(1),
        width: value(2) - value(0),
        height: value(3) - value(1)
      },
      metadata
    );

    if (videoBox.width >= 2 && videoBox.height >= 2) {
      detections.push({ box: videoBox, score });
    }
  }

  return detections;
}

function modelBoxToVideoBox(modelBox, metadata) {
  const localX = (modelBox.x - metadata.padX) / metadata.scale;
  const localY = (modelBox.y - metadata.padY) / metadata.scale;
  const localWidth = modelBox.width / metadata.scale;
  const localHeight = modelBox.height / metadata.scale;

  const rawLeft = metadata.tile.x + localX;
  const rawTop = metadata.tile.y + localY;
  const rawRight = rawLeft + localWidth;
  const rawBottom = rawTop + localHeight;

  const left = clamp(rawLeft, 0, metadata.sourceWidth);
  const top = clamp(rawTop, 0, metadata.sourceHeight);
  const right = clamp(rawRight, 0, metadata.sourceWidth);
  const bottom = clamp(rawBottom, 0, metadata.sourceHeight);

  return {
    x: left,
    y: top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top)
  };
}

function nonMaximumSuppression(detections, iouThreshold, maximum) {
  const remaining = [...detections].sort((a, b) => b.score - a.score);
  const selected = [];

  while (remaining.length > 0 && selected.length < maximum) {
    const best = remaining.shift();
    selected.push(best);

    for (let index = remaining.length - 1; index >= 0; index -= 1) {
      if (intersectionOverUnion(best.box, remaining[index].box) > iouThreshold) {
        remaining.splice(index, 1);
      }
    }
  }

  return selected;
}

function updateTracks(previousTracks, detections) {
  const unmatchedTrackIndexes = new Set(
    previousTracks.map((_, index) => index)
  );
  const nextTracks = [];
  const maximumCenterDistance = Math.max(140, camera.videoWidth * 0.075);

  for (const detection of detections) {
    let bestTrackIndex = -1;
    let bestMatchScore = -Infinity;

    for (const trackIndex of unmatchedTrackIndexes) {
      const track = previousTracks[trackIndex];
      const overlap = intersectionOverUnion(track.box, detection.box);
      const distance = centerDistance(track.box, detection.box);

      if (
        overlap < CONFIG.trackIouThreshold &&
        distance > maximumCenterDistance
      ) {
        continue;
      }

      const matchScore = overlap * 2 - distance / 1000;

      if (matchScore > bestMatchScore) {
        bestMatchScore = matchScore;
        bestTrackIndex = trackIndex;
      }
    }

    if (bestTrackIndex >= 0) {
      const previous = previousTracks[bestTrackIndex];
      unmatchedTrackIndexes.delete(bestTrackIndex);

      nextTracks.push({
        id: previous.id,
        box: smoothBox(previous.box, detection.box, CONFIG.trackSmoothing),
        score: detection.score,
        misses: 0
      });
    } else {
      nextTracks.push({
        id: nextTrackId++,
        box: detection.box,
        score: detection.score,
        misses: 0
      });
    }
  }

  for (const trackIndex of unmatchedTrackIndexes) {
    const track = previousTracks[trackIndex];
    const misses = track.misses + 1;

    if (misses <= CONFIG.trackMaxMisses) {
      nextTracks.push({ ...track, misses });
    }
  }

  return nextTracks;
}

function drawOverlay() {
  resizeOverlay();
  clearOverlay();

  if (showTiles.checked) drawTileGuides();

  for (const track of tracks.filter((item) => item.misses === 0)) {
    drawTrack(track);
  }
}

function drawTileGuides() {
  overlayContext.save();
  overlayContext.setLineDash([14, 10]);
  overlayContext.lineWidth = Math.max(2, overlay.width / 600);
  overlayContext.strokeStyle = "rgba(250, 204, 21, 0.8)";
  overlayContext.font = `700 ${Math.max(15, overlay.width / 50)}px system-ui`;
  overlayContext.fillStyle = "rgba(250, 204, 21, 0.95)";

  for (const tile of currentTiles) {
    overlayContext.strokeRect(tile.x, tile.y, tile.width, tile.height);
    overlayContext.fillText(`Tile ${tile.id}`, tile.x + 8, tile.y + 22);
  }

  overlayContext.restore();
}

function drawTrack(track) {
  const { x, y, width, height } = track.box;
  const label = `Traffic light #${track.id} · ${Math.round(track.score * 100)}%`;

  overlayContext.lineWidth = Math.max(3, overlay.width / 300);
  overlayContext.strokeStyle = "#38bdf8";
  overlayContext.fillStyle = "rgba(14, 116, 144, 0.9)";
  overlayContext.strokeRect(x, y, width, height);

  overlayContext.font = `700 ${Math.max(17, overlay.width / 38)}px system-ui`;
  const textWidth = overlayContext.measureText(label).width;
  const labelHeight = Math.max(28, overlay.width / 30);
  const labelTop = Math.max(0, y - labelHeight);

  overlayContext.fillRect(x, labelTop, textWidth + 14, labelHeight);
  overlayContext.fillStyle = "#ffffff";
  overlayContext.textBaseline = "middle";
  overlayContext.fillText(label, x + 7, labelTop + labelHeight / 2);
}

function resizeOverlay() {
  if (!camera?.videoWidth || !camera?.videoHeight) return;

  if (
    overlay.width !== camera.videoWidth ||
    overlay.height !== camera.videoHeight
  ) {
    overlay.width = camera.videoWidth;
    overlay.height = camera.videoHeight;
  }
}

function clearOverlay() {
  if (!overlayContext || !overlay) return;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
}

function intersectionOverUnion(first, second) {
  const left = Math.max(first.x, second.x);
  const top = Math.max(first.y, second.y);
  const right = Math.min(first.x + first.width, second.x + second.width);
  const bottom = Math.min(first.y + first.height, second.y + second.height);

  const intersection =
    Math.max(0, right - left) * Math.max(0, bottom - top);

  const union =
    first.width * first.height +
    second.width * second.height -
    intersection;

  return union > 0 ? intersection / union : 0;
}

function centerDistance(first, second) {
  return Math.hypot(
    first.x + first.width / 2 - (second.x + second.width / 2),
    first.y + first.height / 2 - (second.y + second.height / 2)
  );
}

function smoothBox(oldBox, newBox, oldWeight) {
  const newWeight = 1 - oldWeight;

  return {
    x: oldBox.x * oldWeight + newBox.x * newWeight,
    y: oldBox.y * oldWeight + newBox.y * newWeight,
    width: oldBox.width * oldWeight + newBox.width * newWeight,
    height: oldBox.height * oldWeight + newBox.height * newWeight
  };
}

function describeStartupError(error) {
  const message = error?.message || String(error) || "Unknown error";

  if (!window.isSecureContext) return "Camera requires HTTPS or localhost";
  if (error?.name === "NotAllowedError") return "Camera permission was denied";
  if (error?.name === "NotFoundError") return "No camera was found";
  if (error?.name === "NotReadableError") return "Camera is already in use";

  if (message.includes("traffic-light.onnx") || message.includes("404")) {
    return "Model file not found: models/traffic-light.onnx";
  }

  return `Unable to start: ${message}`;
}

function handleGlobalError(event) {
  console.error("Global JavaScript error:", event.error || event.message);
  if (statusElement) {
    statusElement.textContent = `JavaScript error: ${event.message || "unknown"}`;
  }
}

function handleUnhandledRejection(event) {
  console.error("Unhandled promise rejection:", event.reason);
  if (statusElement) {
    statusElement.textContent =
      `Error: ${event.reason?.message || event.reason || "unknown"}`;
  }
}

function requiredElement(id) {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing HTML element: #${id}`);
  return element;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}
