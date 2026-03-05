---
title: Voice & Video Cloner
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: true
license: mit
short_description: AI-powered face swap + voice cloning tool
tags:
  - face-swap
  - voice-cloning
  - deepfake
  - insightface
  - edge-tts
  - whisper
---

# 🎭 Voice & Video Cloner

**Created by Divyansh Raj Soni** | BTech AIML, GGITS

Upload a ~30 second video of yourself talking, provide a target persona's face and voice, and the AI will generate a fully cloned output video with:

- **Face Swap** — Your face replaced with the target persona (InsightFace)
- **Voice Clone** — Your speech re-synthesized in a new voice (Edge-TTS + Whisper)

## How to Use

1. Upload your **source video** (~30 seconds of you talking)
2. Upload a **target face** (clear photo of the persona)
3. Upload a **target voice** (audio sample of the persona's voice)
4. Select language and voice style
5. Click **Start Cloning** and wait for processing
6. Download your cloned video!

## Tech Stack

- **Face Detection & Swap:** InsightFace + ONNX Runtime
- **Speech Transcription:** faster-whisper
- **Voice Synthesis:** Edge-TTS (Microsoft Neural Voices)
- **Video Processing:** OpenCV + MoviePy

## Disclaimer

For educational and research purposes only. Always obtain consent before cloning someone's face or voice.
