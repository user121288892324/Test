/*
  Traffic Light Tracker
  ---------------------
  Static, browser-only proof of concept.

  Requirements:
    models/traffic-light.onnx

  Model:
    Ultralytics YOLO detection model exported to ONNX.
    For a standard COCO model, traffic-light class ID = 9.

  Runtime:
    ONNX Runtime Web using the WASM execution provider.

  Tracking:
    Lightweight box matching based on IoU and center distance.
*/

"use strict";


/* =========================================================
   CONFIGURATION
   ========================================================= */

const CONFIG = {
  modelUrl: "./models/traffic-light.onnx",

  // Must match the model export image size.
  inputSize: 640,

  // COCO class ID for "traffic light".
  trafficLightClassId: 9,

  // Lower values detect more objects but may create false positives.
  confidenceThreshold: 0.22,

  // Removes overlapping duplicate boxes.
  nmsThreshold: 0.45,

  // Approximately four inferences per second.
  inferenceIntervalMs: 250,

  maxDetections: 15,

  // Simple tracking settings.
  trackIouThreshold: 0.20,
  trackCenterDistance: 140,
  trackMaxMisses: 5,
  trackSmoothing: 0.65
};


/* =========================================================
   APPLICATION STATE
   ========================================================= */

let camera;
let overlay;
let overlayContext;
let inputCanvas;
let inputContext;

let startButton;
let stopButton;
let secureBadge;
let emptyState;

let statusElement;
let runtimeElement;
let latencyElement;
let countElement;
let stage;

let cameraStream = null;
let inferenceSession = null;

let inputName = null;
let outputName = null;

let running = false;
let inferenceBusy = false;
let lastInferenceTime = 0;

let tracks = [];
let nextTrackId = 1;


/* =========================================================
   STARTUP
   ========================================================= */

window.addEventListener("DOMContentLoaded", initializeApplication);

window.addEventListener("error", (event) => {
  console.error("Global JavaScript error:", event.error || event.message);

  if (statusElement) {
    statusElement.textContent =
      `JavaScript error: ${event.message || "unknown error"}`;
  }
});

window.addEventListener("unhandledrejection", (event) => {
  console.error("Unhandled promise error:", event.reason);

  if (statusElement) {
    statusElement.textContent =
      `Error: ${event.reason?.message || event.reason || "unknown error"}`;
  }
});


function initializeApplication() {
  getPageElements();

  /*
    Register the buttons immediately.

    Even if ONNX Runtime fails to load, clicking Start will now display a
    visible error rather than appearing to do nothing.
  */
  startButton.addEventListener("click", startApplication);
  stopButton.addEventListener("click", stopApplication);

  window.addEventListener("pagehide", stopApplication);
  window.addEventListener("resize", resizeOverlay);

  configureSecureContextBadge();
  configureOnnxRuntime();

  statusElement.textContent = "Waiting to start";

  console.log("Traffic Light Tracker initialized.");
  console.log("ONNX Runtime loaded:", Boolean(window.ort));
}


function getPageElements() {
  camera = requiredElement("camera");
  overlay = requiredElement("overlay");
  inputCanvas = requiredElement("inputCanvas");

  startButton = requiredElement("startButton");
  stopButton = requiredElement("stopButton");

  secureBadge = requiredElement("secureBadge");
  emptyState = requiredElement("emptyState");
  stage = requiredElement("stage");

  statusElement = requiredElement("status");
  runtimeElement = requiredElement("runtime");
  latencyElement = requiredElement("latency");
  countElement = requiredElement("count");

  overlayContext = overlay.getContext("2d");

  inputContext = inputCanvas.getContext(
    "2d",
    { willReadFrequently: true }
  );

  if (!overlayContext || !inputContext) {
    throw new Error("Unable to create the required canvas context.");
  }
}


function requiredElement(id) {
  const element = document.getElementById(id);

  if (!element) {
    throw new Error(`Missing HTML element: #${id}`);
  }

  return element;
}


function configureSecureContextBadge() {
  if (window.isSecureContext) {
    secureBadge.textContent = "HTTPS ready";
    secureBadge.classList.add("good");
    secureBadge.classList.remove("bad");
  } else {
    secureBadge.textContent = "HTTPS required";
    secureBadge.classList.add("bad");
    secureBadge.classList.remove("good");
  }
}


function configureOnnxRuntime() {
  if (!window.ort) {
    runtimeElement.textContent = "CDN not loaded";
    console.error(
      "ONNX Runtime Web is missing. Check the ort.min.js request in Network."
    );
    return;
  }

  /*
    Use one WASM thread for maximum compatibility.

    A multi-threaded WASM configuration can require additional
    cross-origin-isolation headers.
  */
  ort.env.wasm.numThreads = 1;

  /*
    ONNX Runtime loads supporting .wasm/.mjs files from this directory.
    Keep this version exactly the same as the script in index.html.
  */
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";

  runtimeElement.textContent = "Ready to load";
}


/* =========================================================
   START AND STOP
   ========================================================= */

async function startApplication() {
  console.log("Start camera button clicked.");

  statusElement.textContent = "Start button clicked";
  startButton.disabled = true;

  try {
    validateBrowserSupport();

    statusElement.textContent = "Loading ONNX model";
    await loadModel();

    statusElement.textContent = "Requesting camera permission";
    await openCamera();

    statusElement.textContent = "Tracking";
    stopButton.disabled = false;
    running = true;

    requestAnimationFrame(animationLoop);
  } catch (error) {
    console.error("Unable to start:", error);

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

  if (overlayContext && overlay) {
    overlayContext.clearRect(
      0,
      0,
      overlay.width,
      overlay.height
    );
  }

  if (emptyState) {
    emptyState.hidden = false;
  }

  if (statusElement) {
    statusElement.textContent = "Stopped";
  }

  if (latencyElement) {
    latencyElement.textContent = "—";
  }

  if (countElement) {
    countElement.textContent = "0";
  }

  if (startButton) {
    startButton.disabled = false;
  }

  if (stopButton) {
    stopButton.disabled = true;
  }
}


function validateBrowserSupport() {
  if (!window.ort) {
    throw new Error(
      "ONNX Runtime failed to load. Check your internet connection, " +
      "browser extensions, and the ort.min.js request."
    );
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      "This browser does not support camera access through getUserMedia."
    );
  }

  if (!window.isSecureContext) {
    throw new Error(
      "Camera access requires HTTPS or localhost."
    );
  }
}


/* =========================================================
   MODEL LOADING
   ========================================================= */

async function loadModel() {
  if (inferenceSession) {
    return;
  }

  runtimeElement.textContent = "Loading WASM";

  console.log("Loading model:", CONFIG.modelUrl);

  inferenceSession = await ort.InferenceSession.create(
    CONFIG.modelUrl,
    {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    }
  );

  if (!inferenceSession.inputNames.length) {
    throw new Error("The ONNX model does not contain an input.");
  }

  if (!inferenceSession.outputNames.length) {
    throw new Error("The ONNX model does not contain an output.");
  }

  inputName = inferenceSession.inputNames[0];
  outputName = inferenceSession.outputNames[0];

  console.log("Model input:", inputName);
  console.log("Model output:", outputName);

  runtimeElement.textContent = "WASM";

  await warmUpModel();

  console.log("ONNX model loaded successfully.");
}


async function warmUpModel() {
  const valueCount =
    1 *
    3 *
    CONFIG.inputSize *
    CONFIG.inputSize;

  const warmupData = new Float32Array(valueCount);

  const warmupTensor = new ort.Tensor(
    "float32",
    warmupData,
    [
      1,
      3,
      CONFIG.inputSize,
      CONFIG.inputSize
    ]
  );

  await inferenceSession.run({
    [inputName]: warmupTensor
  });
}


/* =========================================================
   CAMERA
   ========================================================= */

async function openCamera() {
  cameraStream = await navigator.mediaDevices.getUserMedia({
    audio: false,

    video: {
      facingMode: {
        ideal: "environment"
      },

      width: {
        ideal: 1280
      },

      height: {
        ideal: 720
      }
    }
  });

  camera.srcObject = cameraStream;

  await camera.play();
  await waitForVideoMetadata();

  stage.style.aspectRatio =
    `${camera.videoWidth} / ${camera.videoHeight}`;

  resizeOverlay();

  emptyState.hidden = true;
}


function closeCamera() {
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) {
      track.stop();
    }

    cameraStream = null;
  }

  if (camera) {
    camera.srcObject = null;
  }
}


function waitForVideoMetadata() {
  if (camera.videoWidth && camera.videoHeight) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    camera.addEventListener(
      "loadedmetadata",
      resolve,
      { once: true }
    );
  });
}


/* =========================================================
   MAIN LOOP
   ========================================================= */

async function animationLoop(timestamp) {
  if (!running) {
    return;
  }

  drawTracks();

  const enoughTimeElapsed =
    timestamp - lastInferenceTime >=
    CONFIG.inferenceIntervalMs;

  const videoReady =
    camera.readyState >=
    HTMLMediaElement.HAVE_CURRENT_DATA;

  if (
    !inferenceBusy &&
    enoughTimeElapsed &&
    videoReady
  ) {
    lastInferenceTime = timestamp;
    inferenceBusy = true;

    runDetection()
      .catch((error) => {
        console.error("Detection failed:", error);
        statusElement.textContent =
          `Detection error: ${error.message}`;
      })
      .finally(() => {
        inferenceBusy = false;
      });
  }

  requestAnimationFrame(animationLoop);
}


/* =========================================================
   INFERENCE
   ========================================================= */

async function runDetection() {
  const started = performance.now();

  const preprocessing = preprocessFrame(camera);

  const results = await inferenceSession.run({
    [inputName]: preprocessing.tensor
  });

  const output = results[outputName];

  if (!output) {
    throw new Error(
      `Model output "${outputName}" was not returned.`
    );
  }

  const rawDetections =
    parseYoloOutput(output, preprocessing);

  const confidentDetections =
    rawDetections.filter(
      (detection) =>
        detection.score >=
        CONFIG.confidenceThreshold
    );

  const selectedDetections =
    nonMaximumSuppression(
      confidentDetections,
      CONFIG.nmsThreshold,
      CONFIG.maxDetections
    );

  tracks = updateTracks(
    tracks,
    selectedDetections
  );

  const elapsed =
    performance.now() - started;

  latencyElement.textContent =
    `${Math.round(elapsed)} ms`;

  countElement.textContent = String(
    tracks.filter(
      (track) => track.misses === 0
    ).length
  );
}


/* =========================================================
   IMAGE PREPROCESSING
   ========================================================= */

function preprocessFrame(videoElement) {
  const size = CONFIG.inputSize;

  const sourceWidth = videoElement.videoWidth;
  const sourceHeight = videoElement.videoHeight;

  if (!sourceWidth || !sourceHeight) {
    throw new Error("Camera frame dimensions are unavailable.");
  }

  /*
    Letterbox the camera frame into a square without stretching it.
  */
  const scale = Math.min(
    size / sourceWidth,
    size / sourceHeight
  );

  const drawWidth =
    Math.round(sourceWidth * scale);

  const drawHeight =
    Math.round(sourceHeight * scale);

  const padX =
    Math.floor((size - drawWidth) / 2);

  const padY =
    Math.floor((size - drawHeight) / 2);

  inputContext.fillStyle = "#000000";

  inputContext.fillRect(
    0,
    0,
    size,
    size
  );

  inputContext.drawImage(
    videoElement,
    padX,
    padY,
    drawWidth,
    drawHeight
  );

  const rgba = inputContext.getImageData(
    0,
    0,
    size,
    size
  ).data;

  const pixelCount = size * size;

  /*
    Convert browser RGBA pixels into YOLO's NCHW RGB float tensor:
      [1, 3, 640, 640]
  */
  const chw = new Float32Array(
    pixelCount * 3
  );

  for (
    let pixelIndex = 0;
    pixelIndex < pixelCount;
    pixelIndex += 1
  ) {
    const rgbaIndex = pixelIndex * 4;

    chw[pixelIndex] =
      rgba[rgbaIndex] / 255;

    chw[pixelCount + pixelIndex] =
      rgba[rgbaIndex + 1] / 255;

    chw[
      pixelCount * 2 + pixelIndex
    ] =
      rgba[rgbaIndex + 2] / 255;
  }

  return {
    tensor: new ort.Tensor(
      "float32",
      chw,
      [
        1,
        3,
        size,
        size
      ]
    ),

    scale,
    padX,
    padY,
    sourceWidth,
    sourceHeight
  };
}


/* =========================================================
   YOLO OUTPUT PARSING
   ========================================================= */

function parseYoloOutput(output, metadata) {
  const dimensions = output.dims;
  const data = output.data;

  if (
    !Array.isArray(dimensions) ||
    dimensions.length !== 3 ||
    dimensions[0] !== 1
  ) {
    throw new Error(
      "Unexpected model output shape: " +
      JSON.stringify(dimensions)
    );
  }

  /*
    Support a model exported with end-to-end NMS:
      [1, N, 6]
      [1, 6, N]

    Each row:
      x1, y1, x2, y2, score, class_id
  */
  if (
    dimensions[2] === 6 ||
    dimensions[1] === 6
  ) {
    return parseEndToEndOutput(
      data,
      dimensions,
      metadata
    );
  }

  /*
    Standard Ultralytics raw detection output:

      YOLOv8/YOLO11:
        [1, 84, N] or [1, N, 84]

        4 box values
        80 class probabilities

      Some YOLOv5-style exports:
        [1, 85, N] or [1, N, 85]

        4 box values
        1 objectness value
        80 class probabilities
  */

  const channelFirst =
    dimensions[1] < dimensions[2];

  const channelCount =
    channelFirst
      ? dimensions[1]
      : dimensions[2];

  const candidateCount =
    channelFirst
      ? dimensions[2]
      : dimensions[1];

  const isYoloV5Style =
    channelCount >= 85;

  const classOffset =
    isYoloV5Style ? 5 : 4;

  const trafficLightChannel =
    classOffset +
    CONFIG.trafficLightClassId;

  if (
    trafficLightChannel >=
    channelCount
  ) {
    throw new Error(
      `Traffic-light class ${CONFIG.trafficLightClassId} ` +
      `is not present in a ${channelCount}-channel output.`
    );
  }

  const detections = [];

  for (
    let candidateIndex = 0;
    candidateIndex < candidateCount;
    candidateIndex += 1
  ) {
    const value = (channelIndex) => {
      if (channelFirst) {
        return data[
          channelIndex * candidateCount +
          candidateIndex
        ];
      }

      return data[
        candidateIndex * channelCount +
        channelIndex
      ];
    };

    const objectness =
      isYoloV5Style
        ? value(4)
        : 1;

    const classProbability =
      value(trafficLightChannel);

    const score =
      objectness *
      classProbability;

    if (
      score <
      CONFIG.confidenceThreshold
    ) {
      continue;
    }

    const centerX = value(0);
    const centerY = value(1);
    const width = value(2);
    const height = value(3);

    const modelBox = {
      x: centerX - width / 2,
      y: centerY - height / 2,
      width,
      height
    };

    const videoBox =
      modelBoxToVideoBox(
        modelBox,
        metadata
      );

    if (
      videoBox.width < 2 ||
      videoBox.height < 2
    ) {
      continue;
    }

    detections.push({
      box: videoBox,
      score
    });
  }

  return detections;
}


function parseEndToEndOutput(
  data,
  dimensions,
  metadata
) {
  const rowMajor =
    dimensions[2] === 6;

  const rowCount =
    rowMajor
      ? dimensions[1]
      : dimensions[2];

  const detections = [];

  for (
    let rowIndex = 0;
    rowIndex < rowCount;
    rowIndex += 1
  ) {
    const value = (columnIndex) => {
      if (rowMajor) {
        return data[
          rowIndex * 6 +
          columnIndex
        ];
      }

      return data[
        columnIndex * rowCount +
        rowIndex
      ];
    };

    const classId =
      Math.round(value(5));

    const score =
      value(4);

    if (
      classId !==
        CONFIG.trafficLightClassId ||
      score <
        CONFIG.confidenceThreshold
    ) {
      continue;
    }

    const x1 = value(0);
    const y1 = value(1);
    const x2 = value(2);
    const y2 = value(3);

    const videoBox =
      modelBoxToVideoBox(
        {
          x: x1,
          y: y1,
          width: x2 - x1,
          height: y2 - y1
        },
        metadata
      );

    if (
      videoBox.width >= 2 &&
      videoBox.height >= 2
    ) {
      detections.push({
        box: videoBox,
        score
      });
    }
  }

  return detections;
}


function modelBoxToVideoBox(
  modelBox,
  metadata
) {
  const x =
    (modelBox.x - metadata.padX) /
    metadata.scale;

  const y =
    (modelBox.y - metadata.padY) /
    metadata.scale;

  const width =
    modelBox.width /
    metadata.scale;

  const height =
    modelBox.height /
    metadata.scale;

  const left = clamp(
    x,
    0,
    metadata.sourceWidth
  );

  const top = clamp(
    y,
    0,
    metadata.sourceHeight
  );

  const right = clamp(
    x + width,
    0,
    metadata.sourceWidth
  );

  const bottom = clamp(
    y + height,
    0,
    metadata.sourceHeight
  );

  return {
    x: left,
    y: top,
    width: Math.max(
      0,
      right - left
    ),
    height: Math.max(
      0,
      bottom - top
    )
  };
}


/* =========================================================
   NON-MAXIMUM SUPPRESSION
   ========================================================= */

function nonMaximumSuppression(
  detections,
  iouThreshold,
  maxDetections
) {
  const remaining = [...detections].sort(
    (a, b) => b.score - a.score
  );

  const selected = [];

  while (
    remaining.length > 0 &&
    selected.length < maxDetections
  ) {
    const best = remaining.shift();

    selected.push(best);

    for (
      let index = remaining.length - 1;
      index >= 0;
      index -= 1
    ) {
      const overlap =
        intersectionOverUnion(
          best.box,
          remaining[index].box
        );

      if (overlap > iouThreshold) {
        remaining.splice(index, 1);
      }
    }
  }

  return selected;
}


/* =========================================================
   SIMPLE TRACKING
   ========================================================= */

function updateTracks(
  previousTracks,
  detections
) {
  const unmatchedTrackIndexes =
    new Set(
      previousTracks.map(
        (_, index) => index
      )
    );

  const nextTracks = [];

  for (const detection of detections) {
    let bestTrackIndex = -1;
    let bestMatchScore = -Infinity;

    for (
      const trackIndex of
      unmatchedTrackIndexes
    ) {
      const track =
        previousTracks[trackIndex];

      const overlap =
        intersectionOverUnion(
          track.box,
          detection.box
        );

      const distance =
        centerDistance(
          track.box,
          detection.box
        );

      const qualifies =
        overlap >=
          CONFIG.trackIouThreshold ||
        distance <=
          CONFIG.trackCenterDistance;

      if (!qualifies) {
        continue;
      }

      const matchScore =
        overlap * 2 -
        distance / 1000;

      if (
        matchScore >
        bestMatchScore
      ) {
        bestMatchScore =
          matchScore;

        bestTrackIndex =
          trackIndex;
      }
    }

    if (bestTrackIndex >= 0) {
      const previous =
        previousTracks[
          bestTrackIndex
        ];

      unmatchedTrackIndexes.delete(
        bestTrackIndex
      );

      nextTracks.push({
        id: previous.id,

        box: smoothBox(
          previous.box,
          detection.box,
          CONFIG.trackSmoothing
        ),

        score: detection.score,
        misses: 0
      });
    } else {
      nextTracks.push({
        id: nextTrackId,
        box: detection.box,
        score: detection.score,
        misses: 0
      });

      nextTrackId += 1;
    }
  }

  /*
    Preserve unmatched tracks briefly so a momentary missed detection does
    not immediately create a new tracking ID.
  */
  for (
    const trackIndex of
    unmatchedTrackIndexes
  ) {
    const track =
      previousTracks[trackIndex];

    const nextMissCount =
      track.misses + 1;

    if (
      nextMissCount <=
      CONFIG.trackMaxMisses
    ) {
      nextTracks.push({
        ...track,
        misses: nextMissCount
      });
    }
  }

  return nextTracks;
}


/* =========================================================
   DRAWING
   ========================================================= */

function drawTracks() {
  resizeOverlay();

  overlayContext.clearRect(
    0,
    0,
    overlay.width,
    overlay.height
  );

  const visibleTracks =
    tracks.filter(
      (track) => track.misses === 0
    );

  for (const track of visibleTracks) {
    const {
      x,
      y,
      width,
      height
    } = track.box;

    const confidence =
      Math.round(
        track.score * 100
      );

    const label =
      `Traffic light #${track.id} · ${confidence}%`;

    overlayContext.lineWidth =
      Math.max(
        3,
        overlay.width / 300
      );

    overlayContext.strokeStyle =
      "#38bdf8";

    overlayContext.fillStyle =
      "rgba(14, 116, 144, 0.88)";

    overlayContext.strokeRect(
      x,
      y,
      width,
      height
    );

    const fontSize =
      Math.max(
        17,
        overlay.width / 38
      );

    overlayContext.font =
      `700 ${fontSize}px system-ui`;

    const textWidth =
      overlayContext.measureText(
        label
      ).width;

    const labelHeight =
      Math.max(
        28,
        overlay.width / 30
      );

    const labelTop =
      Math.max(
        0,
        y - labelHeight
      );

    overlayContext.fillRect(
      x,
      labelTop,
      textWidth + 14,
      labelHeight
    );

    overlayContext.fillStyle =
      "#ffffff";

    overlayContext.textBaseline =
      "middle";

    overlayContext.fillText(
      label,
      x + 7,
      labelTop +
        labelHeight / 2
    );
  }
}


function resizeOverlay() {
  if (
    !camera ||
    !camera.videoWidth ||
    !camera.videoHeight
  ) {
    return;
  }

  if (
    overlay.width !==
      camera.videoWidth ||
    overlay.height !==
      camera.videoHeight
  ) {
    overlay.width =
      camera.videoWidth;

    overlay.height =
      camera.videoHeight;
  }
}


/* =========================================================
   BOX UTILITIES
   ========================================================= */

function intersectionOverUnion(
  first,
  second
) {
  const left =
    Math.max(
      first.x,
      second.x
    );

  const top =
    Math.max(
      first.y,
      second.y
    );

  const right =
    Math.min(
      first.x + first.width,
      second.x + second.width
    );

  const bottom =
    Math.min(
      first.y + first.height,
      second.y + second.height
    );

  const intersection =
    Math.max(
      0,
      right - left
    ) *
    Math.max(
      0,
      bottom - top
    );

  const union =
    first.width *
      first.height +
    second.width *
      second.height -
    intersection;

  if (union <= 0) {
    return 0;
  }

  return intersection / union;
}


function centerDistance(
  first,
  second
) {
  const firstCenterX =
    first.x +
    first.width / 2;

  const firstCenterY =
    first.y +
    first.height / 2;

  const secondCenterX =
    second.x +
    second.width / 2;

  const secondCenterY =
    second.y +
    second.height / 2;

  return Math.hypot(
    firstCenterX -
      secondCenterX,
    firstCenterY -
      secondCenterY
  );
}


function smoothBox(
  oldBox,
  newBox,
  oldWeight
) {
  const newWeight =
    1 - oldWeight;

  return {
    x:
      oldBox.x * oldWeight +
      newBox.x * newWeight,

    y:
      oldBox.y * oldWeight +
      newBox.y * newWeight,

    width:
      oldBox.width * oldWeight +
      newBox.width * newWeight,

    height:
      oldBox.height * oldWeight +
      newBox.height * newWeight
  };
}


function clamp(
  value,
  minimum,
  maximum
) {
  return Math.min(
    maximum,
    Math.max(
      minimum,
      value
    )
  );
}


/* =========================================================
   ERROR MESSAGES
   ========================================================= */

function describeStartupError(error) {
  const message =
    error?.message ||
    String(error) ||
    "Unknown error";

  if (!window.isSecureContext) {
    return "Camera requires HTTPS or localhost";
  }

  if (
    error?.name ===
    "NotAllowedError"
  ) {
    return "Camera permission was denied";
  }

  if (
    error?.name ===
    "NotFoundError"
  ) {
    return "No camera was found";
  }

  if (
    error?.name ===
    "NotReadableError"
  ) {
    return "Camera is already in use";
  }

  if (
    message.includes(
      "traffic-light.onnx"
    ) ||
    message.includes("404")
  ) {
    return "Model file not found: models/traffic-light.onnx";
  }

  if (
    message.includes(
      "ONNX Runtime"
    )
  ) {
    return message;
  }

  return `Unable to start: ${message}`;
}
