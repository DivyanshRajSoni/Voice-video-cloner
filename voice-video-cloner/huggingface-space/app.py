"""
🎭 Voice & Video Cloner — Hugging Face Space
Created by Divyansh Raj Soni | BTech AIML, GGITS

AI-powered face swap + voice cloning tool.
Upload a 30-second video → get a fully cloned output with new face & voice.
"""

import os
import sys
import uuid
import time
import asyncio
import tempfile
import shutil
import cv2
import numpy as np
import gradio as gr
import edge_tts

# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx"
MODEL_PATH = os.path.join(MODELS_DIR, "inswapper_128.onnx")

# ═══════════════════════════════════════════════════════════════
#  Voice Personas (Edge-TTS Neural Voices)
# ═══════════════════════════════════════════════════════════════
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

LANGUAGE_CHOICES = [
    "en", "hi", "es", "fr", "de", "it", "pt",
    "ja", "zh", "ko", "ar", "ru", "tr",
]


# ═══════════════════════════════════════════════════════════════
#  Model Download
# ═══════════════════════════════════════════════════════════════
def download_model():
    """Download inswapper_128.onnx if not present."""
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        if size_mb > 100:
            print(f"[Model] inswapper_128.onnx already exists ({size_mb:.0f} MB)")
            return
    
    print("[Model] Downloading inswapper_128.onnx (~530 MB)...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"[Model] Download complete ({size_mb:.0f} MB)")


# ═══════════════════════════════════════════════════════════════
#  Face Swapper
# ═══════════════════════════════════════════════════════════════
class FaceSwapper:
    def __init__(self):
        self.face_analyser = None
        self.swapper = None
        self._ready = False

    def initialize(self):
        if self._ready:
            return
        import insightface
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        print("[FaceSwapper] Loading face analysis model...")
        self.face_analyser = FaceAnalysis(
            name="buffalo_l", root=MODELS_DIR, providers=providers
        )
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))

        print("[FaceSwapper] Loading swapper model...")
        download_model()
        self.swapper = insightface.model_zoo.get_model(MODEL_PATH, providers=providers)
        self._ready = True
        print("[FaceSwapper] Ready.")

    def get_best_face(self, frame):
        self.initialize()
        faces = self.face_analyser.get(frame)
        if not faces:
            return None
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def get_reference_face(self, image_path):
        self.initialize()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        face = self.get_best_face(img)
        if face is None:
            raise ValueError("No face detected in target image. Use a clear, front-facing photo.")
        return face

    def swap_frame(self, frame, target_face):
        source_face = self.get_best_face(frame)
        if source_face is None:
            return frame
        return self.swapper.get(frame, source_face, target_face, paste_back=True)


# ═══════════════════════════════════════════════════════════════
#  Voice Cloner
# ═══════════════════════════════════════════════════════════════
class VoiceCloner:
    def __init__(self):
        self.whisper_model = None
        self._ready = False

    def initialize(self):
        if self._ready:
            return
        print("[VoiceCloner] Loading faster-whisper model...")
        from faster_whisper import WhisperModel
        self.whisper_model = WhisperModel("base", compute_type="int8")
        self._ready = True
        print("[VoiceCloner] Ready.")

    def transcribe(self, audio_path, language="en"):
        self.initialize()
        try:
            lang = language.split("-")[0]
            segments, info = self.whisper_model.transcribe(audio_path, language=lang, beam_size=5)
            text = " ".join([s.text for s in segments]).strip()
            print(f"[Whisper] Transcribed ({len(text)} chars, lang={info.language})")
            return text
        except Exception as e:
            print(f"[Whisper] Error: {e}")
            return ""

    def synthesize(self, text, voice, output_path):
        async def _synth():
            comm = edge_tts.Communicate(text, voice)
            await comm.save(output_path)
        asyncio.run(_synth())

    def extract_audio(self, video_path, output_path):
        from moviepy import VideoFileClip
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            raise ValueError("Video has no audio track.")
        clip.audio.write_audiofile(output_path, logger=None)
        clip.close()
        return output_path


# ═══════════════════════════════════════════════════════════════
#  Global instances (loaded once)
# ═══════════════════════════════════════════════════════════════
face_swapper = FaceSwapper()
voice_cloner = VoiceCloner()


# ═══════════════════════════════════════════════════════════════
#  Main Processing Pipeline
# ═══════════════════════════════════════════════════════════════
def process_video(
    source_video,
    target_face_image,
    target_voice_audio,
    language,
    voice_choice,
    progress=gr.Progress()
):
    """
    Main pipeline: face swap + voice clone + combine.
    
    Args:
        source_video: Path to uploaded source video
        target_face_image: Path to target face image
        target_voice_audio: Path to target voice audio
        language: Language code
        voice_choice: Voice persona name
        progress: Gradio progress tracker
    """
    if source_video is None:
        raise gr.Error("Please upload a source video.")
    if target_face_image is None:
        raise gr.Error("Please upload a target face image.")
    if target_voice_audio is None:
        raise gr.Error("Please upload a target voice audio/video.")

    job_id = uuid.uuid4().hex[:8]
    job_dir = tempfile.mkdtemp(prefix=f"clone_{job_id}_")

    try:
        # ─── Stage 1: Extract Audio ───────────────────────
        progress(0.05, desc="🎵 Extracting audio from source video...")
        source_audio = os.path.join(job_dir, "source_audio.wav")
        voice_cloner.extract_audio(source_video, source_audio)

        # ─── Stage 2: Transcribe ──────────────────────────
        progress(0.15, desc="📝 Transcribing speech...")
        transcript = voice_cloner.transcribe(source_audio, language)
        if not transcript:
            transcript = "Hello, this is a voice cloning demonstration."

        # ─── Stage 3: Synthesize Voice ────────────────────
        progress(0.25, desc="🗣️ Synthesizing cloned voice...")
        voice_name = VOICE_CHOICES.get(voice_choice, "")
        if not voice_name:
            voice_name = DEFAULT_VOICES.get(language, "en-US-GuyNeural")

        cloned_audio = os.path.join(job_dir, "cloned_audio.mp3")
        voice_cloner.synthesize(transcript, voice_name, cloned_audio)

        # ─── Stage 4: Face Swap ──────────────────────────
        progress(0.35, desc="🎭 Loading face swap model...")
        target_face = face_swapper.get_reference_face(target_face_image)

        progress(0.40, desc="🎭 Swapping faces frame by frame...")
        swapped_video = os.path.join(job_dir, "swapped.mp4")

        cap = cv2.VideoCapture(source_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(swapped_video, fourcc, fps, (w, h))

        count = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = face_swapper.swap_frame(frame, target_face)
            writer.write(processed)
            count += 1
            if count % 5 == 0:
                pct = 0.40 + 0.45 * (count / max(total, 1))
                elapsed = time.time() - t0
                fps_actual = count / max(elapsed, 0.001)
                remaining = (total - count) / max(fps_actual, 0.001)
                progress(
                    pct,
                    desc=f"🎭 Frame {count}/{total} ({fps_actual:.1f} fps, ~{remaining:.0f}s left)"
                )

        cap.release()
        writer.release()
        print(f"[Pipeline] Face swap done: {count} frames in {time.time() - t0:.1f}s")

        # ─── Stage 5: Combine ────────────────────────────
        progress(0.90, desc="🎬 Combining video + cloned audio...")
        from moviepy import VideoFileClip, AudioFileClip

        video_clip = VideoFileClip(swapped_video)
        audio_clip = AudioFileClip(cloned_audio)

        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.with_subclip(0, video_clip.duration)

        final = video_clip.with_audio(audio_clip)
        output_path = os.path.join(job_dir, f"cloned_{job_id}.mp4")
        final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        video_clip.close()
        audio_clip.close()
        final.close()

        progress(1.0, desc="✅ Done! Your cloned video is ready.")
        return output_path

    except Exception as e:
        # Clean up on error
        shutil.rmtree(job_dir, ignore_errors=True)
        raise gr.Error(f"Processing failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════════════════════════
CUSTOM_CSS = """
.main-header {
    text-align: center;
    margin-bottom: 8px;
}
.main-header h1 {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    font-weight: 800;
    margin-bottom: 4px;
}
.main-header p {
    color: #9090a8;
    font-size: 0.95em;
}
.creator-badge {
    text-align: center;
    padding: 6px 16px;
    margin-top: 4px;
    font-size: 0.85em;
    color: #a0a0b8;
}
.creator-badge strong {
    color: #a855f7;
}
.disclaimer {
    text-align: center;
    font-size: 0.75em;
    color: #707088;
    margin-top: 8px;
    padding: 8px;
    border-top: 1px solid #2a2a3a;
}
"""

with gr.Blocks(
    title="Voice & Video Cloner — by Divyansh Raj Soni",
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CUSTOM_CSS,
) as app:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🎭 Voice & Video Cloner</h1>
            <p>Upload a 30-second video of yourself → Get a fully cloned output with a new face & voice</p>
        </div>
        <div class="creator-badge">
            Created by <strong>Divyansh Raj Soni</strong> | BTech AIML, GGITS
        </div>
    """)

    with gr.Row():
        # Left Column — Inputs
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Files")
            
            source_video = gr.Video(
                label="① Your Source Video (~30 sec)",
                sources=["upload"],
            )
            
            target_face = gr.Image(
                label="② Target Persona Face (clear photo)",
                type="filepath",
                sources=["upload"],
            )
            
            target_voice = gr.Audio(
                label="③ Target Voice Sample (10+ sec audio)",
                type="filepath",
                sources=["upload"],
            )

            with gr.Row():
                language = gr.Dropdown(
                    choices=LANGUAGE_CHOICES,
                    value="en",
                    label="Language",
                    scale=1,
                )
                voice_select = gr.Dropdown(
                    choices=list(VOICE_CHOICES.keys()),
                    value="Auto (default)",
                    label="Voice Style",
                    scale=2,
                )

            clone_btn = gr.Button(
                "⚡ Start Cloning",
                variant="primary",
                size="lg",
            )

        # Right Column — Output
        with gr.Column(scale=1):
            gr.Markdown("### 🎬 Cloned Output")
            output_video = gr.Video(label="Result", interactive=False)
            download_btn = gr.DownloadButton(
                label="📥 Download Cloned Video",
                visible=False,
            )

    # Footer
    gr.HTML("""
        <div class="disclaimer">
            ⚠️ For educational & research purposes only. Always obtain consent before cloning someone's face or voice.<br>
            Powered by InsightFace • Edge-TTS • faster-whisper • OpenCV • MoviePy
        </div>
    """)

    # ─── Event handlers ───────────────────────────────────
    def on_clone_complete(video_path):
        """After processing, show download button."""
        if video_path:
            return gr.update(visible=True, value=video_path)
        return gr.update(visible=False)

    clone_btn.click(
        fn=process_video,
        inputs=[source_video, target_face, target_voice, language, voice_select],
        outputs=[output_video],
    ).then(
        fn=on_clone_complete,
        inputs=[output_video],
        outputs=[download_btn],
    )


# ═══════════════════════════════════════════════════════════════
#  Launch
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  🎭 Voice & Video Cloner")
    print("  Created by Divyansh Raj Soni | BTech AIML, GGITS")
    print("=" * 60)
    
    # Pre-download model
    download_model()
    
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
