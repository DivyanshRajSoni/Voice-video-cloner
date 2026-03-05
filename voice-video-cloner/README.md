# Voice & Video Cloner

A non-real-time AI-powered tool that transforms a 30-second video of you talking into a target persona — swapping both **face** and **voice**.

Upload your video + a target persona's face/voice, and the AI generates a fully cloned output video file.

---

## Features

- **Face Swap** — Replaces your face with the target persona using InsightFace (ONNX-based)
- **Voice Clone** — Re-synthesizes your speech in the target's voice using Coqui XTTS-v2
- **Speech Transcription** — Automatically transcribes your speech via OpenAI Whisper
- **Web UI** — Beautiful drag-and-drop interface with real-time progress tracking
- **GPU Accelerated** — Supports CUDA for faster processing
- **Multi-language** — Supports 16+ languages for voice cloning

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Web Browser (UI)                   │
│  Upload: Source Video + Target Face + Target Voice   │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP POST /api/clone
                       ▼
┌─────────────────────────────────────────────────────┐
│                 Flask Backend (app.py)                │
│            Job Queue + Progress Tracking             │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌──────────┐ ┌──────────────┐
   │ Extract    │ │ Voice    │ │ Face Swap    │
   │ Audio      │ │ Cloner   │ │ (per frame)  │
   │ (moviepy)  │ │ (XTTS)   │ │ (InsightFace)│
   └────────────┘ └──────────┘ └──────────────┘
          │            │            │
          └────────────┼────────────┘
                       ▼
              ┌────────────────┐
              │ Combine Video  │
              │ + Cloned Audio │
              │ (moviepy/ffmpeg)│
              └───────┬────────┘
                      ▼
              Output MP4 File
```

---

## Prerequisites

- **Python 3.9+**
- **FFmpeg** (must be in PATH)
- **~6GB disk space** (for AI models)
- **GPU recommended** (NVIDIA with CUDA) — CPU works but is slower

### Install FFmpeg

```bash
# Windows (via Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

---

## Quick Start

### 1. Clone / Navigate to project

```bash
cd voice-video-cloner
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU Users:** Also install `onnxruntime-gpu`:
> ```bash
> pip install onnxruntime-gpu
> ```

### 4. Download the Face Swap model

Download `inswapper_128.onnx` and place it in the `models/` folder:

- **Download:** https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx
- **Place at:** `models/inswapper_128.onnx`

### 5. Run setup check

```bash
python setup.py
```

### 6. Start the application

```bash
python app.py
```

### 7. Open your browser

Navigate to **http://localhost:5000**

---

## Usage

1. **Upload Source Video** — Your ~30 second video of yourself talking
2. **Upload Target Face** — A clear photo or video of the persona you want to become
3. **Upload Target Voice** — An audio clip (10+ seconds) of the target persona's voice
4. **Select Language** — Choose the language spoken in the video
5. **Click "Start Cloning"** — Wait for processing (progress is shown in real-time)
6. **Download** — Get your cloned video!

---

## Project Structure

```
voice-video-cloner/
├── app.py                    # Flask web server & API routes
├── setup.py                  # Setup verification script
├── requirements.txt          # Python dependencies
├── core/
│   ├── __init__.py
│   ├── face_swapper.py       # InsightFace face detection & swap
│   ├── voice_cloner.py       # Coqui XTTS-v2 voice cloning
│   └── video_processor.py    # Pipeline orchestrator
├── templates/
│   └── index.html            # Web UI template
├── static/
│   ├── css/
│   │   └── style.css         # UI styles
│   └── js/
│       └── app.js            # Frontend logic
├── models/                   # AI model files (downloaded)
│   └── inswapper_128.onnx    # Face swap model (download manually)
├── uploads/                  # Temporary uploaded files
└── outputs/                  # Generated output videos
```

---

## API Endpoints

| Method | Endpoint               | Description                        |
|--------|------------------------|------------------------------------|
| GET    | `/`                    | Web UI                             |
| POST   | `/api/clone`           | Start a cloning job                |
| GET    | `/api/status/<job_id>` | Check job progress                 |
| GET    | `/api/download/<job_id>` | Download completed output        |
| GET    | `/api/health`          | Health check + GPU status          |

---

## Processing Pipeline

| Stage           | Tool                  | Description                                    |
|-----------------|-----------------------|------------------------------------------------|
| Extract Audio   | MoviePy               | Separates audio track from source video        |
| Transcribe      | OpenAI Whisper        | Converts speech to text                        |
| Voice Clone     | Coqui XTTS-v2        | Re-synthesizes text in target persona's voice  |
| Face Swap       | InsightFace + ONNX    | Replaces face in every video frame             |
| Combine         | MoviePy + FFmpeg      | Merges face-swapped video with cloned audio    |

---

## Performance

| Hardware         | 30s Video (720p) | 30s Video (1080p) |
|------------------|-------------------|--------------------|
| RTX 3060 (GPU)   | ~3-5 minutes      | ~5-8 minutes       |
| CPU only (i7)    | ~15-25 minutes    | ~30-45 minutes     |

---

## Troubleshooting

### "Face swap model not found"
Download `inswapper_128.onnx` and place it in the `models/` directory.

### "No face detected"
- Ensure the target face image has a clear, front-facing face
- Good lighting and no obstructions

### "CUDA out of memory"
- Reduce video resolution before uploading
- Close other GPU-intensive applications

### Voice cloning sounds off
- Provide a longer target voice sample (15-30 seconds ideal)
- Use a clean audio recording with minimal background noise

---

## Disclaimer

This tool is built for **educational and research purposes only**. Always:
- Obtain consent before cloning someone's face or voice
- Do not use for deception, fraud, or harassment
- Comply with local laws regarding synthetic media
- Label AI-generated content appropriately

---

## Tech Stack

- **Backend:** Flask (Python)
- **Face Swap:** InsightFace + ONNX Runtime
- **Voice Clone:** Coqui TTS (XTTS-v2)
- **Transcription:** OpenAI Whisper
- **Video Processing:** OpenCV + MoviePy + FFmpeg
- **Frontend:** Vanilla HTML/CSS/JS
