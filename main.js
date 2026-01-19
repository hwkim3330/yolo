/**
 * YOLO26 Vision - WebGPU Real-time Object Detection
 * Apple-inspired interface for YOLO26 model inference
 */

import { AutoModel, AutoProcessor, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

// ============================================
// Configuration
// ============================================

const CONFIG = {
  defaultModel: 'onnx-community/yolo26n-ONNX',
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
    threshold: 0.0001
  },
  colors: [
    '#6366f1', '#ec4899', '#14b8a6', '#f59e0b',
    '#8b5cf6', '#ef4444', '#10b981', '#3b82f6',
    '#f97316', '#84cc16', '#06b6d4', '#a855f7'
  ],
  skeleton: [
    [0, 1], [0, 2], [1, 3], [2, 4],           // Head
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  // Arms
    [5, 11], [6, 12], [11, 12],               // Torso
    [11, 13], [13, 15], [12, 14], [14, 16]    // Legs
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

// ============================================
// DOM Elements
// ============================================

const elements = {
  // Video
  video: document.getElementById('video'),
  canvas: document.getElementById('canvas'),
  placeholder: document.getElementById('video-placeholder'),

  // Loader
  loader: document.getElementById('loader'),
  loaderText: document.getElementById('loader-text'),
  loaderBar: document.getElementById('loader-bar'),

  // Status
  statusIndicator: document.getElementById('status-indicator'),
  statusDot: document.querySelector('.status-dot'),
  statusText: document.getElementById('status-text'),

  // Stats
  fps: document.getElementById('fps'),
  inferenceTime: document.getElementById('inference-time'),
  detectionsCount: document.getElementById('detections-count'),

  // Controls
  modelBtns: document.querySelectorAll('.model-btn'),
  toggleDetect: document.getElementById('toggle-detect'),
  togglePose: document.getElementById('toggle-pose'),
  toggleSegment: document.getElementById('toggle-segment'),
  threshold: document.getElementById('threshold'),
  thresholdValue: document.getElementById('threshold-value'),
  cameraSelect: document.getElementById('camera-select'),
  flipCamera: document.getElementById('flip-camera'),

  // Buttons
  startBtn: document.getElementById('start-btn'),
  btnIcon: document.getElementById('btn-icon'),
  btnText: document.getElementById('btn-text'),
  themeToggle: document.getElementById('theme-toggle')
};

// Get canvas context
const ctx = elements.canvas.getContext('2d');

// Offscreen canvas for frame capture
const offscreen = document.createElement('canvas');
const offscreenCtx = offscreen.getContext('2d');

// ============================================
// Application State
// ============================================

const state = {
  detectModel: null,
  poseModel: null,
  processor: null,
  currentModelId: null,

  isRunning: false,
  isProcessing: false,
  animationId: null,

  threshold: 0.5,
  enableDetect: true,
  enablePose: false,

  cameras: [],
  currentCameraIndex: 0,
  facingMode: 'environment',

  // Performance tracking
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

const formatNumber = (num) => {
  return num.toLocaleString('en-US', { maximumFractionDigits: 1 });
};

// ============================================
// Theme Management
// ============================================

const initTheme = () => {
  const savedTheme = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = savedTheme || (prefersDark ? 'dark' : 'light');

  document.documentElement.setAttribute('data-theme', theme);
};

const toggleTheme = () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
};

// ============================================
// Camera Management
// ============================================

const enumerateCameras = async () => {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    state.cameras = devices.filter(d => d.kind === 'videoinput');

    elements.cameraSelect.innerHTML = '';
    state.cameras.forEach((camera, index) => {
      const option = document.createElement('option');
      option.value = index;
      option.textContent = camera.label || `Camera ${index + 1}`;
      elements.cameraSelect.appendChild(option);
    });
  } catch (error) {
    console.error('Failed to enumerate cameras:', error);
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

const loadModels = async (modelId) => {
  try {
    // Stop camera if running
    if (state.isRunning) {
      stopCamera(true);
    }

    // Wait for any processing to complete
    while (state.isProcessing) {
      await new Promise(r => setTimeout(r, 50));
    }

    // Dispose existing models
    if (state.detectModel) {
      await state.detectModel.dispose();
      state.detectModel = null;
    }
    if (state.poseModel) {
      await state.poseModel.dispose();
      state.poseModel = null;
    }
    state.processor = null;

    elements.startBtn.disabled = true;
    setStatus('Loading...', 'loading');

    const poseModelId = modelId.replace('-ONNX', '-pose-ONNX');

    // Progress callback
    const createProgressCallback = (label) => (info) => {
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
      progress_callback: createProgressCallback('Detection model')
    });

    // Load pose model
    showLoader('Loading pose model...', 0);
    state.poseModel = await AutoModel.from_pretrained(poseModelId, {
      device: 'webgpu',
      dtype: 'fp32',
      progress_callback: createProgressCallback('Pose model')
    });

    // Load processor
    showLoader('Loading processor...', 100);
    state.processor = await AutoProcessor.from_pretrained(modelId);

    state.currentModelId = modelId;

    setStatus('Ready', 'ready');
    hideLoader();
    elements.startBtn.disabled = false;

    // Auto-start camera
    await startCamera();

  } catch (error) {
    console.error('Model loading failed:', error);
    setStatus('Error', 'error');
    showLoader(`Failed: ${error.message}`);

    setTimeout(hideLoader, 3000);
  }
};

// ============================================
// Camera Control
// ============================================

const startCamera = async () => {
  try {
    showLoader('Accessing camera...');

    const constraints = {
      video: {
        ...CONFIG.videoConstraints,
        facingMode: state.facingMode
      },
      audio: false
    };

    // Use specific camera if selected
    const selectedIndex = elements.cameraSelect.value;
    if (selectedIndex && state.cameras[selectedIndex]) {
      constraints.video.deviceId = { exact: state.cameras[selectedIndex].deviceId };
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    elements.video.srcObject = stream;

    await new Promise((resolve) => {
      elements.video.onloadedmetadata = resolve;
    });

    // Set canvas dimensions
    const { videoWidth, videoHeight } = elements.video;
    elements.canvas.width = offscreen.width = videoWidth;
    elements.canvas.height = offscreen.height = videoHeight;

    // Update UI
    elements.placeholder.classList.add('hidden');
    state.isRunning = true;

    elements.btnIcon.innerHTML = `
      <svg viewBox="0 0 24 24" fill="currentColor">
        <rect x="6" y="6" width="12" height="12" rx="2"/>
      </svg>
    `;
    elements.btnText.textContent = 'Stop';
    elements.startBtn.classList.add('running');

    hideLoader();
    setStatus('Running', 'running');

    // Start detection loop
    runDetectionLoop();

  } catch (error) {
    console.error('Camera error:', error);
    setStatus('Camera Error', 'error');
    showLoader('Camera access denied');
    setTimeout(hideLoader, 2000);
  }
};

const stopCamera = (keepProcessing = false) => {
  // Cancel animation frame
  if (state.animationId) {
    cancelAnimationFrame(state.animationId);
    state.animationId = null;
  }

  // Stop video stream
  if (elements.video.srcObject) {
    elements.video.srcObject.getTracks().forEach(track => track.stop());
    elements.video.srcObject = null;
  }

  state.isRunning = false;
  if (!keepProcessing) {
    state.isProcessing = false;
  }

  // Update UI
  elements.placeholder.classList.remove('hidden');
  elements.btnIcon.innerHTML = `
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5v14l11-7z"/>
    </svg>
  `;
  elements.btnText.textContent = 'Start';
  elements.startBtn.classList.remove('running');

  // Clear canvas
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

  // Reset stats
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

  // Update FPS every second
  if (now - state.lastFpsUpdate >= 1000) {
    elements.fps.textContent = state.frameCount;
    state.frameCount = 0;
    state.lastFpsUpdate = now;
  }

  // Process frame if models ready and not already processing
  if (state.detectModel && state.poseModel && state.processor && !state.isProcessing) {
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

  // Capture frame
  offscreenCtx.drawImage(elements.video, 0, 0);
  const image = RawImage.fromCanvas(offscreen);
  const inputs = await state.processor(image);

  // Run inference
  const promises = [];
  if (state.enableDetect) promises.push(state.detectModel(inputs));
  if (state.enablePose) promises.push(state.poseModel(inputs));

  if (promises.length === 0) {
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    return;
  }

  const results = await Promise.all(promises);

  // Parse results
  let idx = 0;
  const detectOutput = state.enableDetect ? results[idx++] : null;
  const poseOutput = state.enablePose ? results[idx++] : null;

  const detections = parseDetections(detectOutput, poseOutput);

  // Update inference time
  const inferenceTime = performance.now() - startTime;
  state.recentInferenceTimes.push(inferenceTime);
  if (state.recentInferenceTimes.length > 10) {
    state.recentInferenceTimes.shift();
  }
  const avgInference = state.recentInferenceTimes.reduce((a, b) => a + b, 0) / state.recentInferenceTimes.length;
  elements.inferenceTime.textContent = Math.round(avgInference);
  elements.detectionsCount.textContent = detections.length;

  // Draw results
  if (state.isRunning) {
    draw(detections);
  }
};

const parseDetections = (detectOutput, poseOutput) => {
  const detections = [];
  const { width, height } = elements.canvas;

  // Parse object detections
  if (detectOutput) {
    const scores = detectOutput.logits.sigmoid().data;
    const boxes = detectOutput.pred_boxes.data;
    const id2label = state.detectModel.config.id2label || {};

    for (let i = 0; i < CONFIG.detection.maxDetections; i++) {
      let maxScore = 0;
      let maxClass = 0;

      for (let j = 0; j < CONFIG.detection.numClasses; j++) {
        const score = scores[i * CONFIG.detection.numClasses + j];
        if (score > maxScore) {
          maxScore = score;
          maxClass = j;
        }
      }

      if (maxScore >= state.threshold) {
        const offset = i * 4;
        const cx = boxes[offset];
        const cy = boxes[offset + 1];
        const w = boxes[offset + 2];
        const h = boxes[offset + 3];

        detections.push({
          type: 'object',
          box: [
            (cx - w / 2) * width,
            (cy - h / 2) * height,
            w * width,
            h * height
          ],
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
      const offset = i * 57; // 4 bbox + 1 score + 1 class + 17*3 keypoints
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
          box: [
            data[offset] * width,
            data[offset + 1] * height,
            (data[offset + 2] - data[offset]) * width,
            (data[offset + 3] - data[offset + 1]) * height
          ],
          score,
          keypoints
        });
      }
    }
  }

  return detections;
};

// ============================================
// Drawing
// ============================================

const draw = (detections) => {
  ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);

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

  // Draw box with rounded corners
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';

  const radius = 6;
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
  ctx.stroke();

  // Draw label background
  ctx.font = 'bold 13px Inter, system-ui, sans-serif';
  const textWidth = ctx.measureText(label).width;
  const labelHeight = 24;
  const labelY = y > labelHeight + 4 ? y - labelHeight - 4 : y + h + 4;

  // Rounded label background
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(x, labelY, textWidth + 12, labelHeight, 4);
  ctx.fill();

  // Label text
  ctx.fillStyle = '#ffffff';
  ctx.fillText(label, x + 6, labelY + 16);
};

const drawPoseDetection = (det) => {
  const { keypoints } = det;

  // Draw skeleton
  ctx.strokeStyle = '#22d3ee';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';

  for (const [i, j] of CONFIG.skeleton) {
    const kpA = keypoints[i];
    const kpB = keypoints[j];

    if (kpA.confidence >= CONFIG.pose.threshold && kpB.confidence >= CONFIG.pose.threshold) {
      ctx.beginPath();
      ctx.moveTo(kpA.x, kpA.y);
      ctx.lineTo(kpB.x, kpB.y);
      ctx.stroke();
    }
  }

  // Draw keypoints
  for (const kp of keypoints) {
    if (kp.confidence < CONFIG.pose.threshold) continue;

    // Outer circle
    ctx.fillStyle = '#6366f1';
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 6, 0, Math.PI * 2);
    ctx.fill();

    // Inner circle (white)
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
};

// ============================================
// Event Listeners
// ============================================

const initEventListeners = () => {
  // Theme toggle
  elements.themeToggle.addEventListener('click', toggleTheme);

  // Start/Stop button
  elements.startBtn.addEventListener('click', () => {
    if (state.isRunning) {
      stopCamera();
    } else {
      startCamera();
    }
  });

  // Model selection
  elements.modelBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const modelId = btn.dataset.model;
      if (modelId === state.currentModelId) return;

      // Update active state
      elements.modelBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Load new model
      loadModels(modelId);
    });
  });

  // Task toggles
  elements.toggleDetect.addEventListener('change', (e) => {
    state.enableDetect = e.target.checked;
    e.target.closest('.toggle-item').classList.toggle('active', e.target.checked);
  });

  elements.togglePose.addEventListener('change', (e) => {
    state.enablePose = e.target.checked;
    e.target.closest('.toggle-item').classList.toggle('active', e.target.checked);
  });

  // Threshold slider
  elements.threshold.addEventListener('input', (e) => {
    state.threshold = e.target.value / 100;
    elements.thresholdValue.textContent = `${e.target.value}%`;
  });

  // Camera controls
  elements.cameraSelect.addEventListener('change', () => {
    if (state.isRunning) {
      stopCamera(true);
      startCamera();
    }
  });

  elements.flipCamera.addEventListener('click', flipCamera);

  // Handle visibility change (pause when tab hidden)
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.isRunning) {
      // Pause processing when hidden
      state.isProcessing = true;
    } else if (!document.hidden && state.isRunning) {
      state.isProcessing = false;
    }
  });
};

// ============================================
// Initialization
// ============================================

const init = async () => {
  // Initialize theme
  initTheme();

  // Initialize event listeners
  initEventListeners();

  // Enumerate cameras
  await enumerateCameras();

  // Check WebGPU support
  if (!navigator.gpu) {
    setStatus('WebGPU Not Supported', 'error');
    showLoader('WebGPU is required. Please use Chrome 113+ or Edge 113+.');
    elements.startBtn.disabled = true;
    return;
  }

  // Load default model
  await loadModels(CONFIG.defaultModel);
};

// Start app
init().catch(console.error);
