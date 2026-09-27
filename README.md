# YOLO26 Vision — WebGPU

**Demo:** https://hwkim3330.github.io/yolo/ (Chrome/Edge with WebGPU, webcam)

Real-time object detection and pose estimation from the webcam, running entirely in the browser with Ultralytics YOLO26 ONNX models on WebGPU. No server.

## Features

- Model sizes: Nano, Small, Medium, Large (`onnx-community/yolo26{n,s,m,l}-ONNX`).
- Tasks: detection and pose (`yolo26*-pose-ONNX`), toggled independently.
- Confidence threshold slider, camera selection, FPS / latency / object count.

## Run locally

```bash
python3 -m http.server 8000   # open http://localhost:8000
```

## Files

`index.html` (UI), `main.js` (model loading and inference loop with Transformers.js), `styles.css`.

**Tech:** Transformers.js 3.8, ONNX Runtime Web (WebGPU), vanilla JS.

**Status:** working demo.
