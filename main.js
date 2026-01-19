/**
 * YOLO26 Vision - WebGPU/ONNX Real-time Object Detection
 * Supports: Detection, Pose, Segmentation, Classification, OBB
 */

import { AutoModel, AutoProcessor, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/esm/ort.webgpu.min.js';

// ============================================
// Configuration
// ============================================

const CONFIG = {
  // Hugging Face models (for detection and pose)
  hfModels: {
    'yolo26n': 'onnx-community/yolo26n-ONNX',
    'yolo26s': 'onnx-community/yolo26s-ONNX',
    'yolo26m': 'onnx-community/yolo26m-ONNX',
    'yolo26l': 'onnx-community/yolo26l-ONNX',
  },

  // Local ONNX models (for seg, cls, obb)
  localModels: {
    'seg': './yolo26n-seg.onnx',
    'cls': './yolo26n-cls.onnx',
    'obb': './yolo26n-obb.onnx',
  },

  videoConstraints: {
    facingMode: 'environment',
    width: { ideal: 640 },
    height: { ideal: 640 }
  },

  detection: {
    numClasses: 80,
    maxDetections: 300
  },

  pose: {
    numKeypoints: 17,
    threshold: 0.3
  },

  classification: {
    numClasses: 1000,
    inputSize: 224
  },

  colors: [
    '#6366f1', '#ec4899', '#14b8a6', '#f59e0b',
    '#8b5cf6', '#ef4444', '#10b981', '#3b82f6',
    '#f97316', '#84cc16', '#06b6d4', '#a855f7'
  ],

  skeleton: [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
  ],

  // Segmentation mask colors
  maskColors: [
    [255, 99, 132, 100], [54, 162, 235, 100], [255, 206, 86, 100],
    [75, 192, 192, 100], [153, 102, 255, 100], [255, 159, 64, 100],
    [199, 199, 199, 100], [83, 102, 255, 100], [255, 99, 255, 100],
    [99, 255, 132, 100], [132, 99, 255, 100], [255, 132, 99, 100]
  ]
};

// COCO class names
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

// ImageNet class names (top 20 common ones for demo)
const IMAGENET_COMMON = {
  0: 'tench', 1: 'goldfish', 281: 'tabby cat', 282: 'tiger cat', 285: 'Egyptian cat',
  151: 'Chihuahua', 207: 'golden retriever', 208: 'Labrador', 243: 'mastiff',
  386: 'African elephant', 101: 'tusker', 340: 'zebra', 950: 'orange',
  954: 'banana', 963: 'pizza', 967: 'espresso', 968: 'cup'
};

// ============================================
// DOM Elements
// ============================================

const elements = {
  video: document.getElementById('video'),
  canvas: document.getElementById('canvas'),
  placeholder: document.getElementById('video-placeholder'),
  loader: document.getElementById('loader'),
  loaderText: document.getElementById('loader-text'),
  loaderBar: document.getElementById('loader-bar'),
  statusDot: document.querySelector('.status-dot'),
  statusText: document.getElementById('status-text'),
  fps: document.getElementById('fps'),
  inferenceTime: document.getElementById('inference-time'),
  detectionsCount: document.getElementById('detections-count'),
  modelBtns: document.querySelectorAll('.model-btn'),
  toggleDetect: document.getElementById('toggle-detect'),
  togglePose: document.getElementById('toggle-pose'),
  toggleSegment: document.getElementById('toggle-segment'),
  threshold: document.getElementById('threshold'),
  thresholdValue: document.getElementById('threshold-value'),
  cameraSelect: document.getElementById('camera-select'),
  flipCamera: document.getElementById('flip-camera'),
  startBtn: document.getElementById('start-btn'),
  btnIcon: document.getElementById('btn-icon'),
  btnText: document.getElementById('btn-text'),
  themeToggle: document.getElementById('theme-toggle')
};

const ctx = elements.canvas.getContext('2d');
const offscreen = document.createElement('canvas');
const offscreenCtx = offscreen.getContext('2d');

// Mask overlay canvas
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d');

// ============================================
// Application State
// ============================================

const state = {
  // Transformers.js models
  detectModel: null,
  poseModel: null,
  processor: null,

  // ONNX Runtime sessions
  segSession: null,
  clsSession: null,
  obbSession: null,

  currentModelSize: 'n',
  isRunning: false,
  isProcessing: false,
  animationId: null,

  threshold: 0.5,
  enableDetect: true,
  enablePose: false,
  enableSegment: false,

  cameras: [],
  facingMode: 'environment',

  frameCount: 0,
  lastFpsUpdate: 0,
  recentInferenceTimes: []
};

// ============================================
// Utility Functions
// ============================================

const setStatus = (text, type = 'default') => {
  elements.statusText.textContent = text;
  elements.statusDot.className = 'status-dot ' + type;
};

const showLoader = (text, progress = null) => {
  elements.loaderText.textContent = text;
  elements.loader.classList.add('visible');
  if (progress !== null) {
    elements.loaderBar.style.width = `${progress}%`;
  }
};

const hideLoader = () => {
  elements.loader.classList.remove('visible');
  elements.loaderBar.style.width = '0%';
};

// Sigmoid activation
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Softmax activation
const softmax = (arr) => {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
};

// ============================================
// Theme Management
// ============================================

const initTheme = () => {
  const savedTheme = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.setAttribute('data-theme', savedTheme || (prefersDark ? 'dark' : 'light'));
};

const toggleTheme = () => {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
};

// ============================================
// Camera Management
// ============================================

const enumerateCameras = async () => {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    state.cameras = devices.filter(d => d.kind === 'videoinput');
    elements.cameraSelect.innerHTML = '';
    state.cameras.forEach((cam, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = cam.label || `Camera ${i + 1}`;
      elements.cameraSelect.appendChild(opt);
    });
  } catch (e) {
    console.error('Camera enumeration failed:', e);
  }
};

const flipCamera = () => {
  state.facingMode = state.facingMode === 'environment' ? 'user' : 'environment';
  if (state.isRunning) {
    stopCamera(true);
    startCamera();
  }
};

// ============================================
// Model Loading
// ============================================

const loadModels = async (modelSize = 'n') => {
  try {
    if (state.isRunning) stopCamera(true);
    while (state.isProcessing) await new Promise(r => setTimeout(r, 50));

    // Dispose existing models
    if (state.detectModel) await state.detectModel.dispose();
    if (state.poseModel) await state.poseModel.dispose();
    state.detectModel = state.poseModel = state.processor = null;

    elements.startBtn.disabled = true;
    setStatus('Loading...', 'loading');

    const modelId = CONFIG.hfModels[`yolo26${modelSize}`];
    const poseModelId = modelId.replace('-ONNX', '-pose-ONNX');

    const progressCallback = (label) => (info) => {
      if (info.status === 'progress' && info.file?.endsWith('.onnx')) {
        const progress = Math.round((info.loaded / info.total) * 100);
        showLoader(`${label} (${progress}%)`, progress);
      }
    };

    // Load detection model
    showLoader('Loading detection model...', 0);
    state.detectModel = await AutoModel.from_pretrained(modelId, {
      device: 'webgpu',
      dtype: 'fp16',
      progress_callback: progressCallback('Detection')
    });

    // Load pose model
    showLoader('Loading pose model...', 0);
    state.poseModel = await AutoModel.from_pretrained(poseModelId, {
      device: 'webgpu',
      dtype: 'fp32',
      progress_callback: progressCallback('Pose')
    });

    // Load processor
    showLoader('Loading processor...', 100);
    state.processor = await AutoProcessor.from_pretrained(modelId);

    // Load local ONNX models for segmentation
    await loadLocalModels();

    state.currentModelSize = modelSize;
    setStatus('Ready', 'ready');
    hideLoader();
    elements.startBtn.disabled = false;

    // Enable segmentation toggle
    elements.toggleSegment.disabled = false;
    elements.toggleSegment.closest('.toggle-item').classList.remove('disabled');

    await startCamera();

  } catch (error) {
    console.error('Model loading failed:', error);
    setStatus('Error', 'error');
    showLoader(`Failed: ${error.message}`);
    setTimeout(hideLoader, 3000);
  }
};

const loadLocalModels = async () => {
  try {
    // Configure ONNX Runtime for WebGPU
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

    showLoader('Loading segmentation model...', 50);

    // Check if local model exists
    const segResponse = await fetch(CONFIG.localModels.seg, { method: 'HEAD' });
    if (segResponse.ok) {
      state.segSession = await ort.InferenceSession.create(CONFIG.localModels.seg, {
        executionProviders: ['webgpu', 'wasm'],
        graphOptimizationLevel: 'all'
      });
      console.log('Segmentation model loaded');
    } else {
      console.log('Segmentation model not found locally');
    }

  } catch (error) {
    console.warn('Local model loading failed:', error);
    // Continue without local models
  }
};

// ============================================
// Camera Control
// ============================================

const startCamera = async () => {
  try {
    showLoader('Accessing camera...');

    const constraints = {
      video: { ...CONFIG.videoConstraints, facingMode: state.facingMode },
      audio: false
    };

    const selectedIdx = elements.cameraSelect.value;
    if (selectedIdx && state.cameras[selectedIdx]) {
      constraints.video.deviceId = { exact: state.cameras[selectedIdx].deviceId };
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    elements.video.srcObject = stream;

    await new Promise(resolve => { elements.video.onloadedmetadata = resolve; });

    const { videoWidth, videoHeight } = elements.video;
    elements.canvas.width = offscreen.width = maskCanvas.width = videoWidth;
    elements.canvas.height = offscreen.height = maskCanvas.height = videoHeight;

    elements.placeholder.classList.add('hidden');
    state.isRunning = true;

    elements.btnIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
    elements.btnText.textContent = 'Stop';
    elements.startBtn.classList.add('running');

    hideLoader();
    setStatus('Running', 'running');

    runDetectionLoop();

  } catch (error) {
    console.error('Camera error:', error);
    setStatus('Camera Error', 'error');
    showLoader('Camera access denied');
    setTimeout(hideLoader, 2000);
  }
};

const stopCamera = (keepProcessing = false) => {
  if (state.animationId) {
    cancelAnimationFrame(state.animationId);
    state.animationId = null;
  }

  if (elements.video.srcObject) {
    elements.video.srcObject.getTracks().forEach(t => t.stop());
    elements.video.srcObject = null;
  }

  state.isRunning = false;
  if (!keepProcessing) state.isProcessing = false;

  elements.placeholder.classList.remove('hidden');
  elements.btnIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
  elements.btnText.textContent = 'Start';
  elements.startBtn.classList.remove('running');

  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
  elements.fps.textContent = '0';
  elements.inferenceTime.textContent = '0';
  elements.detectionsCount.textContent = '0';

  setStatus('Ready', 'ready');
};

// ============================================
// Detection Loop
// ============================================

const runDetectionLoop = () => {
  if (!state.isRunning) return;

  const now = performance.now();

  if (now - state.lastFpsUpdate >= 1000) {
    elements.fps.textContent = state.frameCount;
    state.frameCount = 0;
    state.lastFpsUpdate = now;
  }

  if (state.detectModel && state.processor && !state.isProcessing) {
    state.isProcessing = true;
    processFrame()
      .catch(console.error)
      .finally(() => {
        state.isProcessing = false;
        state.frameCount++;
      });
  }

  state.animationId = requestAnimationFrame(runDetectionLoop);
};

const processFrame = async () => {
  const startTime = performance.now();

  offscreenCtx.drawImage(elements.video, 0, 0);
  const image = RawImage.fromCanvas(offscreen);
  const inputs = await state.processor(image);

  const promises = [];
  if (state.enableDetect) promises.push(state.detectModel(inputs));
  if (state.enablePose) promises.push(state.poseModel(inputs));

  // Segmentation with ONNX Runtime
  let segResults = null;
  if (state.enableSegment && state.segSession) {
    segResults = await runSegmentation(image);
  }

  if (promises.length === 0 && !segResults) {
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    return;
  }

  const results = await Promise.all(promises);

  let idx = 0;
  const detectOutput = state.enableDetect ? results[idx++] : null;
  const poseOutput = state.enablePose ? results[idx++] : null;

  const detections = parseDetections(detectOutput, poseOutput, segResults);

  const inferenceTime = performance.now() - startTime;
  state.recentInferenceTimes.push(inferenceTime);
  if (state.recentInferenceTimes.length > 10) state.recentInferenceTimes.shift();
  const avg = state.recentInferenceTimes.reduce((a, b) => a + b, 0) / state.recentInferenceTimes.length;
  elements.inferenceTime.textContent = Math.round(avg);
  elements.detectionsCount.textContent = detections.length;

  if (state.isRunning) draw(detections);
};

// ============================================
// Segmentation with ONNX Runtime
// ============================================

const runSegmentation = async (image) => {
  try {
    // Preprocess image for segmentation
    const inputTensor = await preprocessForOnnx(image, 640, 640);

    // Run inference
    const feeds = { images: inputTensor };
    const results = await state.segSession.run(feeds);

    // Get outputs - YOLO26 seg outputs: boxes (1,300,38), masks (1,32,160,160)
    const outputNames = Object.keys(results);
    const output0 = results[outputNames[0]]; // Detection output
    const output1 = results[outputNames[1]]; // Mask prototypes

    return { detections: output0, masks: output1 };
  } catch (error) {
    console.error('Segmentation inference failed:', error);
    return null;
  }
};

const preprocessForOnnx = async (image, width, height) => {
  // Create a canvas to resize the image
  const resizeCanvas = document.createElement('canvas');
  resizeCanvas.width = width;
  resizeCanvas.height = height;
  const resizeCtx = resizeCanvas.getContext('2d');

  // Draw and resize
  resizeCtx.drawImage(offscreen, 0, 0, width, height);
  const imageData = resizeCtx.getImageData(0, 0, width, height);

  // Convert to float32 tensor [1, 3, H, W] with normalization
  const float32Data = new Float32Array(3 * width * height);

  for (let i = 0; i < width * height; i++) {
    float32Data[i] = imageData.data[i * 4] / 255.0;                    // R
    float32Data[width * height + i] = imageData.data[i * 4 + 1] / 255.0; // G
    float32Data[2 * width * height + i] = imageData.data[i * 4 + 2] / 255.0; // B
  }

  return new ort.Tensor('float32', float32Data, [1, 3, height, width]);
};

// ============================================
// Parse Detections
// ============================================

const parseDetections = (detectOutput, poseOutput, segResults) => {
  const detections = [];
  const { width, height } = elements.canvas;

  // Parse object detections from Transformers.js
  if (detectOutput) {
    const scores = detectOutput.logits.sigmoid().data;
    const boxes = detectOutput.pred_boxes.data;
    const id2label = state.detectModel.config.id2label || {};

    for (let i = 0; i < CONFIG.detection.maxDetections; i++) {
      let maxScore = 0, maxClass = 0;
      for (let j = 0; j < CONFIG.detection.numClasses; j++) {
        const score = scores[i * CONFIG.detection.numClasses + j];
        if (score > maxScore) { maxScore = score; maxClass = j; }
      }

      if (maxScore >= state.threshold) {
        const offset = i * 4;
        const cx = boxes[offset], cy = boxes[offset + 1];
        const w = boxes[offset + 2], h = boxes[offset + 3];

        detections.push({
          type: 'object',
          box: [(cx - w/2) * width, (cy - h/2) * height, w * width, h * height],
          score: maxScore,
          classId: maxClass,
          label: id2label[maxClass] || COCO_CLASSES[maxClass] || `Class ${maxClass}`
        });
      }
    }
  }

  // Parse pose detections
  if (poseOutput) {
    const data = Object.values(poseOutput)[0].data;
    for (let i = 0; i < CONFIG.detection.maxDetections; i++) {
      const offset = i * 57;
      const score = data[offset + 4];

      if (score >= state.threshold) {
        const keypoints = [];
        for (let k = 0; k < CONFIG.pose.numKeypoints; k++) {
          const kOffset = offset + 6 + k * 3;
          keypoints.push({
            x: data[kOffset] * width,
            y: data[kOffset + 1] * height,
            confidence: data[kOffset + 2]
          });
        }

        detections.push({
          type: 'pose',
          box: [data[offset] * width, data[offset + 1] * height,
                (data[offset + 2] - data[offset]) * width,
                (data[offset + 3] - data[offset + 1]) * height],
          score,
          keypoints
        });
      }
    }
  }

  // Parse segmentation detections
  if (segResults) {
    const segDetections = parseSegmentation(segResults, width, height);
    detections.push(...segDetections);
  }

  return detections;
};

const parseSegmentation = (segResults, canvasWidth, canvasHeight) => {
  const detections = [];

  try {
    const { detections: detData, masks: maskData } = segResults;

    // YOLO26 seg output format: (1, 300, 38) - 4 bbox + 1 score + 1 class + 32 mask coeffs
    const data = detData.data;
    const maskProtos = maskData.data;
    const numDetections = 300;
    const maskSize = 160;

    for (let i = 0; i < numDetections; i++) {
      const offset = i * 38;
      const x1 = data[offset];
      const y1 = data[offset + 1];
      const x2 = data[offset + 2];
      const y2 = data[offset + 3];
      const score = data[offset + 4];
      const classId = Math.round(data[offset + 5]);

      if (score < state.threshold) continue;

      // Get mask coefficients
      const maskCoeffs = [];
      for (let j = 0; j < 32; j++) {
        maskCoeffs.push(data[offset + 6 + j]);
      }

      // Convert normalized coords to canvas coords
      const box = [
        x1 * canvasWidth,
        y1 * canvasHeight,
        (x2 - x1) * canvasWidth,
        (y2 - y1) * canvasHeight
      ];

      detections.push({
        type: 'segment',
        box,
        score,
        classId,
        label: COCO_CLASSES[classId] || `Class ${classId}`,
        maskCoeffs,
        maskProtos: maskProtos,
        maskSize
      });
    }
  } catch (error) {
    console.error('Segmentation parsing error:', error);
  }

  return detections;
};

// ============================================
// Drawing
// ============================================

const draw = (detections) => {
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

  // Draw segmentation masks first (background layer)
  for (const det of detections.filter(d => d.type === 'segment')) {
    drawSegmentationMask(det);
  }

  // Draw object detections
  for (const det of detections.filter(d => d.type === 'object')) {
    drawObjectDetection(det);
  }

  // Draw pose detections
  for (const det of detections.filter(d => d.type === 'pose')) {
    drawPoseDetection(det);
  }
};

const drawObjectDetection = (det) => {
  const [x, y, w, h] = det.box;
  const color = CONFIG.colors[det.classId % CONFIG.colors.length];
  const label = `${det.label} ${Math.round(det.score * 100)}%`;

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';

  // Rounded rectangle
  const radius = 6;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, radius);
  ctx.stroke();

  // Label
  ctx.font = 'bold 13px Inter, system-ui, sans-serif';
  const textWidth = ctx.measureText(label).width;
  const labelHeight = 24;
  const labelY = y > labelHeight + 4 ? y - labelHeight - 4 : y + h + 4;

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(x, labelY, textWidth + 12, labelHeight, 4);
  ctx.fill();

  ctx.fillStyle = '#ffffff';
  ctx.fillText(label, x + 6, labelY + 16);
};

const drawPoseDetection = (det) => {
  const { keypoints } = det;

  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';

  for (const [i, j] of CONFIG.skeleton) {
    const kpA = keypoints[i], kpB = keypoints[j];
    if (kpA.confidence >= CONFIG.pose.threshold && kpB.confidence >= CONFIG.pose.threshold) {
      ctx.beginPath();
      ctx.moveTo(kpA.x, kpA.y);
      ctx.lineTo(kpB.x, kpB.y);
      ctx.stroke();
    }
  }

  for (const kp of keypoints) {
    if (kp.confidence < CONFIG.pose.threshold) continue;

    ctx.fillStyle = '#6366f1';
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
};

const drawSegmentationMask = (det) => {
  const [x, y, w, h] = det.box;
  const color = CONFIG.colors[det.classId % CONFIG.colors.length];
  const label = `${det.label} ${Math.round(det.score * 100)}%`;

  // Draw semi-transparent fill for the bounding box area
  ctx.fillStyle = color + '40'; // 25% opacity
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 6);
  ctx.fill();

  // Draw border
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Label
  ctx.font = 'bold 13px Inter, system-ui, sans-serif';
  const textWidth = ctx.measureText(label).width;
  const labelHeight = 24;
  const labelY = y > labelHeight + 4 ? y - labelHeight - 4 : y + h + 4;

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(x, labelY, textWidth + 12, labelHeight, 4);
  ctx.fill();

  ctx.fillStyle = '#ffffff';
  ctx.fillText(label, x + 6, labelY + 16);
};

// ============================================
// Event Listeners
// ============================================

const initEventListeners = () => {
  elements.themeToggle.addEventListener('click', toggleTheme);

  elements.startBtn.addEventListener('click', () => {
    state.isRunning ? stopCamera() : startCamera();
  });

  elements.modelBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const size = btn.dataset.size;
      if (size === state.currentModelSize) return;

      elements.modelBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      loadModels(size);
    });
  });

  elements.toggleDetect.addEventListener('change', (e) => {
    state.enableDetect = e.target.checked;
    e.target.closest('.toggle-item').classList.toggle('active', e.target.checked);
  });

  elements.togglePose.addEventListener('change', (e) => {
    state.enablePose = e.target.checked;
    e.target.closest('.toggle-item').classList.toggle('active', e.target.checked);
  });

  elements.toggleSegment.addEventListener('change', (e) => {
    state.enableSegment = e.target.checked;
    e.target.closest('.toggle-item').classList.toggle('active', e.target.checked);
  });

  elements.threshold.addEventListener('input', (e) => {
    state.threshold = e.target.value / 100;
    elements.thresholdValue.textContent = `${e.target.value}%`;
  });

  elements.cameraSelect.addEventListener('change', () => {
    if (state.isRunning) { stopCamera(true); startCamera(); }
  });

  elements.flipCamera.addEventListener('click', flipCamera);

  document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.isRunning) state.isProcessing = true;
    else if (!document.hidden && state.isRunning) state.isProcessing = false;
  });
};

// ============================================
// Initialization
// ============================================

const init = async () => {
  initTheme();
  initEventListeners();
  await enumerateCameras();

  if (!navigator.gpu) {
    setStatus('WebGPU Not Supported', 'error');
    showLoader('WebGPU required. Use Chrome 113+ or Edge 113+.');
    elements.startBtn.disabled = true;
    return;
  }

  await loadModels('n');
};

init().catch(console.error);
