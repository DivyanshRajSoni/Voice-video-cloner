"""
🎭 Voice & Video Cloner — Hugging Face Space
Created by Divyansh Raj Soni | BTech AIML, GGITS
"""

import os
import uuid
import time
import tempfile
import shutil
import gradio as gr

# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx"
MODEL_PATH = os.path.join(MODELS_DIR, "inswapper_128.onnx")

VOICE_CHOICES = {
    "Auto (default)": "",
    "English - Guy (US Male)": "en-US-GuyNeural",
    "English - Davis (US Male)": "en-US-DavisNeural",
    "English - Ryan (UK Male)": "en-GB-RyanNeural",
    "English - Jenny (US Female)": "en-US-JennyNeural",
    "English - Aria (US Female)": "en-US-AriaNeural",
    "English - Sonia (UK Female)": "en-GB-SoniaNeural",
    "Hindi - Madhur (Male)": "hi-IN-MadhurNeural",
    "Hindi - Swara (Female)": "hi-IN-SwaraNeural",
    "Spanish - Alvaro (Male)": "es-ES-AlvaroNeural",
    "Spanish - Elvira (Female)": "es-ES-ElviraNeural",
    "French - Henri (Male)": "fr-FR-HenriNeural",
    "French - Denise (Female)": "fr-FR-DeniseNeural",
    "German - Conrad (Male)": "de-DE-ConradNeural",
    "German - Katja (Female)": "de-DE-KatjaNeural",
    "Japanese - Keita (Male)": "ja-JP-KeitaNeural",
    "Japanese - Nanami (Female)": "ja-JP-NanamiNeural",
    "Chinese - Yunxi (Male)": "zh-CN-YunxiNeural",
    "Chinese - Xiaoxiao (Female)": "zh-CN-XiaoxiaoNeural",
    "Korean - InJoon (Male)": "ko-KR-InJoonNeural",
    "Korean - SunHi (Female)": "ko-KR-SunHiNeural",
    "Italian - Diego (Male)": "it-IT-DiegoNeural",
    "Italian - Elsa (Female)": "it-IT-ElsaNeural",
    "Portuguese - Antonio (Male)": "pt-BR-AntonioNeural",
    "Portuguese - Francisca (Female)": "pt-BR-FranciscaNeural",
    "Arabic - Hamed (Male)": "ar-SA-HamedNeural",
    "Arabic - Zariyah (Female)": "ar-SA-ZariyahNeural",
    "Russian - Dmitry (Male)": "ru-RU-DmitryNeural",
    "Russian - Svetlana (Female)": "ru-RU-SvetlanaNeural",
    "Turkish - Ahmet (Male)": "tr-TR-AhmetNeural",
    "Turkish - Emel (Female)": "tr-TR-EmelNeural",
}

DEFAULT_VOICES = {
    "en": "en-US-GuyNeural", "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural", "de": "de-DE-ConradNeural",
    "it": "it-IT-DiegoNeural", "pt": "pt-BR-AntonioNeural",
    "hi": "hi-IN-MadhurNeural", "ja": "ja-JP-KeitaNeural",
    "zh": "zh-CN-YunxiNeural", "ko": "ko-KR-InJoonNeural",
    "ar": "ar-SA-HamedNeural", "ru": "ru-RU-DmitryNeural",
    "tr": "tr-TR-AhmetNeural",
}

LANGUAGE_CHOICES = ["en", "hi", "es", "fr", "de", "it", "pt", "ja", "zh", "ko", "ar", "ru", "tr"]


# ═══════════════════════════════════════════════════════════════
#  Lazy-loaded globals (nothing heavy imported at module level)
# ═══════════════════════════════════════════════════════════════
_face_analyser = None
_swapper = None
_whisper_model = None


def _ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000_000:
        return
    print("[Model] Downloading inswapper_128.onnx (~530 MB)...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[Model] Done: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.0f} MB")


def _get_face_tools():
    global _face_analyser, _swapper
    if _face_analyser is not None:
        return _face_analyser, _swapper

    import insightface
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    _face_analyser = FaceAnalysis(name="buffalo_l", root=MODELS_DIR, providers=providers)
    _face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    _ensure_model()
    _swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=providers)
    print("[FaceSwapper] Ready!")
    return _face_analyser, _swapper


def _get_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    from faster_whisper import WhisperModel
    _whisper_model = WhisperModel("base", compute_type="int8")
    print("[Whisper] Ready!")
    return _whisper_model


def _best_face(analyser, frame):
    faces = analyser.get(frame)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _transcribe(audio_path, language="en"):
    whisper = _get_whisper()
    try:
        segments, info = whisper.transcribe(audio_path, language=language.split("-")[0], beam_size=5)
        return " ".join(s.text for s in segments).strip()
    except Exception as e:
        print(f"[Whisper Error] {e}")
        return ""


def _synthesize(text, voice, output_path):
    import asyncio
    import edge_tts

    async def _run():
        c = edge_tts.Communicate(text, voice)
        await c.save(output_path)

    asyncio.run(_run())


def _extract_audio(video_path, output_path):
    from moviepy import VideoFileClip
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        raise ValueError("Video has no audio track.")
    clip.audio.write_audiofile(output_path, logger=None)
    clip.close()


# ═══════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════
def process_video(source_video, target_face_image, target_voice_audio, language, voice_choice, progress=gr.Progress()):
    import cv2

    if source_video is None:
        raise gr.Error("Please upload a source video.")
    if target_face_image is None:
        raise gr.Error("Please upload a target face image.")
    if target_voice_audio is None:
        raise gr.Error("Please upload a target voice audio.")

    job_id = uuid.uuid4().hex[:8]
    job_dir = tempfile.mkdtemp(prefix=f"clone_{job_id}_")

    try:
        # 1 — Extract audio
        progress(0.05, desc="🎵 Extracting audio...")
        src_audio = os.path.join(job_dir, "src.wav")
        _extract_audio(source_video, src_audio)

        # 2 — Transcribe
        progress(0.15, desc="📝 Transcribing speech...")
        text = _transcribe(src_audio, language)
        if not text:
            text = "Hello, this is a voice cloning demonstration."

        # 3 — Synthesize
        progress(0.25, desc="🗣️ Synthesizing voice...")
        voice = VOICE_CHOICES.get(voice_choice, "") or DEFAULT_VOICES.get(language, "en-US-GuyNeural")
        cloned = os.path.join(job_dir, "cloned.mp3")
        _synthesize(text, voice, cloned)

        # 4 — Face swap
        progress(0.35, desc="🎭 Loading face models (first time may take minutes)...")
        analyser, swapper = _get_face_tools()

        img = cv2.imread(target_face_image)
        if img is None:
            raise gr.Error("Cannot read target face image.")
        ref = _best_face(analyser, img)
        if ref is None:
            raise gr.Error("No face found in target image. Use a clear front-facing photo.")

        progress(0.40, desc="🎭 Swapping faces...")
        swapped = os.path.join(job_dir, "swapped.mp4")
        cap = cv2.VideoCapture(source_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        writer = cv2.VideoWriter(swapped, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        n, t0 = 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            src = _best_face(analyser, frame)
            if src is not None:
                frame = swapper.get(frame, src, ref, paste_back=True)
            writer.write(frame)
            n += 1
            if n % 10 == 0:
                elapsed = time.time() - t0
                spd = n / max(elapsed, 0.01)
                left = (total - n) / max(spd, 0.01)
                progress(0.40 + 0.45 * n / total, desc=f"🎭 Frame {n}/{total} ({spd:.1f}fps, ~{left:.0f}s)")
        cap.release()
        writer.release()

        # 5 — Combine
        progress(0.90, desc="🎬 Combining video + audio...")
        from moviepy import VideoFileClip, AudioFileClip
        vc = VideoFileClip(swapped)
        ac = AudioFileClip(cloned)
        if ac.duration > vc.duration:
            ac = ac.with_subclip(0, vc.duration)
        out = os.path.join(job_dir, f"cloned_{job_id}.mp4")
        vc.with_audio(ac).write_videofile(out, codec="libx264", audio_codec="aac", logger=None)
        vc.close()
        ac.close()

        progress(1.0, desc="✅ Done!")
        return out

    except gr.Error:
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise gr.Error(f"Processing failed: {e}")


# ═══════════════════════════════════════════════════════════════
#  Gradio UI — the `demo` variable is auto-discovered by HF
# ═══════════════════════════════════════════════════════════════
CSS = """
.hd{text-align:center;margin-bottom:8px}
.hd h1{background:linear-gradient(135deg,#6366f1,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.2em;font-weight:800}
.hd p{color:#9090a8;font-size:.95em}
.cb{text-align:center;padding:6px 16px;font-size:.85em;color:#a0a0b8}
.cb strong{color:#a855f7}
.ft{text-align:center;font-size:.75em;color:#707088;margin-top:8px;padding:8px;border-top:1px solid #2a2a3a}
"""

with gr.Blocks(
    title="Voice & Video Cloner — Divyansh Raj Soni",
    theme=gr.themes.Base(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate"),
    css=CSS,
) as demo:

    gr.HTML('<div class="hd"><h1>🎭 Voice & Video Cloner</h1>'
            '<p>Upload a 30-sec video → Get a cloned output with new face & voice</p></div>'
            '<div class="cb">Created by <strong>Divyansh Raj Soni</strong> | BTech AIML, GGITS</div>')

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Inputs")
            inp_video = gr.Video(label="① Source Video (~30 sec)")
            inp_face = gr.Image(label="② Target Face (clear photo)", type="filepath")
            inp_voice = gr.Audio(label="③ Target Voice Sample", type="filepath")
            with gr.Row():
                inp_lang = gr.Dropdown(LANGUAGE_CHOICES, value="en", label="Language")
                inp_style = gr.Dropdown(list(VOICE_CHOICES.keys()), value="Auto (default)", label="Voice Style")
            btn = gr.Button("⚡ Start Cloning", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### 🎬 Output")
            out_video = gr.Video(label="Cloned Video", interactive=False)

    gr.HTML('<div class="ft">⚠️ Educational & research use only. Obtain consent before cloning.<br>'
            'InsightFace • Edge-TTS • faster-whisper • OpenCV • MoviePy</div>')

    btn.click(fn=process_video, inputs=[inp_video, inp_face, inp_voice, inp_lang, inp_style], outputs=[out_video])

demo.queue()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
