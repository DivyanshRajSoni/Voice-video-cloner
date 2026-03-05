"""
Voice Cloner Module — Production Grade
Uses Edge-TTS (Microsoft neural voices) for speech synthesis,
faster-whisper for transcription, and deep-translator for translation.

Pipeline: Transcribe (auto-detect) → Translate → Synthesize (neural voice)

Features:
- Whisper medium model on GPU for high-accuracy transcription
- VAD-filtered transcription for noise robustness
- Automatic language detection with confidence scoring
- Google Translate with retry + chunking for long texts
- Edge-TTS with rate/pitch control for natural speech
- Thread-safe async synthesis
- Robust error handling and logging
"""

import os
import re
import time
import asyncio
import logging
import edge_tts
from deep_translator import GoogleTranslator

logger = logging.getLogger("VoiceCloner")

# ─── Available voice personas (expanded) ───────────────────────
VOICE_PERSONAS = {
    # English
    "en-male-1": "en-US-GuyNeural",
    "en-male-2": "en-US-DavisNeural",
    "en-male-3": "en-GB-RyanNeural",
    "en-male-4": "en-US-ChristopherNeural",
    "en-female-1": "en-US-JennyNeural",
    "en-female-2": "en-US-AriaNeural",
    "en-female-3": "en-GB-SoniaNeural",
    "en-female-4": "en-US-MichelleNeural",
    # Spanish
    "es-male-1": "es-ES-AlvaroNeural",
    "es-female-1": "es-ES-ElviraNeural",
    "es-male-2": "es-MX-JorgeNeural",
    "es-female-2": "es-MX-DaliaNeural",
    # French
    "fr-male-1": "fr-FR-HenriNeural",
    "fr-female-1": "fr-FR-DeniseNeural",
    # German
    "de-male-1": "de-DE-ConradNeural",
    "de-female-1": "de-DE-KatjaNeural",
    # Hindi
    "hi-male-1": "hi-IN-MadhurNeural",
    "hi-female-1": "hi-IN-SwaraNeural",
    # Japanese
    "ja-male-1": "ja-JP-KeitaNeural",
    "ja-female-1": "ja-JP-NanamiNeural",
    # Chinese
    "zh-male-1": "zh-CN-YunxiNeural",
    "zh-female-1": "zh-CN-XiaoxiaoNeural",
    # Korean
    "ko-male-1": "ko-KR-InJoonNeural",
    "ko-female-1": "ko-KR-SunHiNeural",
    # Italian
    "it-male-1": "it-IT-DiegoNeural",
    "it-female-1": "it-IT-ElsaNeural",
    # Portuguese
    "pt-male-1": "pt-BR-AntonioNeural",
    "pt-female-1": "pt-BR-FranciscaNeural",
    # Arabic
    "ar-male-1": "ar-SA-HamedNeural",
    "ar-female-1": "ar-SA-ZariyahNeural",
    # Russian
    "ru-male-1": "ru-RU-DmitryNeural",
    "ru-female-1": "ru-RU-SvetlanaNeural",
    # Turkish
    "tr-male-1": "tr-TR-AhmetNeural",
    "tr-female-1": "tr-TR-EmelNeural",
    # Dutch
    "nl-male-1": "nl-NL-MaartenNeural",
    "nl-female-1": "nl-NL-ColetteNeural",
    # Polish
    "pl-male-1": "pl-PL-MarekNeural",
    "pl-female-1": "pl-PL-ZofiaNeural",
    # Czech
    "cs-male-1": "cs-CZ-AntoninNeural",
    "cs-female-1": "cs-CZ-VlastaNeural",
}

DEFAULT_VOICES = {
    "en": "en-US-GuyNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "it": "it-IT-DiegoNeural",
    "pt": "pt-BR-AntonioNeural",
    "pl": "pl-PL-MarekNeural",
    "tr": "tr-TR-AhmetNeural",
    "ru": "ru-RU-DmitryNeural",
    "nl": "nl-NL-MaartenNeural",
    "cs": "cs-CZ-AntoninNeural",
    "ar": "ar-SA-HamedNeural",
    "zh-cn": "zh-CN-YunxiNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "hi": "hi-IN-MadhurNeural",
}

# Language names for logging
LANG_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "tr": "Turkish", "nl": "Dutch", "pl": "Polish", "cs": "Czech",
}


class VoiceCloner:
    """
    Production-grade voice transformation pipeline.
    Uses faster-whisper (medium, GPU) + Google Translate + Edge-TTS.
    """

    # ── Language code mapping for Google Translate ──
    TRANSLATE_LANG_MAP = {
        "en": "en", "hi": "hi", "es": "es", "fr": "fr", "de": "de",
        "it": "it", "pt": "pt", "pl": "pl", "tr": "tr", "ru": "ru",
        "nl": "nl", "cs": "cs", "ar": "ar", "zh-cn": "zh-CN", "zh": "zh-CN",
        "ja": "ja", "ko": "ko",
    }

    def __init__(self):
        self.whisper_model = None
        self._initialized = False

    def initialize(self):
        """Load the Whisper transcription model (medium on GPU for accuracy)."""
        if self._initialized:
            return

        t0 = time.time()
        logger.info("Loading faster-whisper model (medium, GPU)...")
        print("[VoiceCloner] Loading faster-whisper model (medium)...")
        from faster_whisper import WhisperModel

        # Use GPU (float16) if available, else CPU (int8)
        try:
            import torch
            if torch.cuda.is_available():
                self.whisper_model = WhisperModel(
                    "medium",
                    device="cuda",
                    compute_type="float16",
                )
                elapsed = time.time() - t0
                logger.info(f"Whisper medium loaded on GPU in {elapsed:.1f}s")
                print(f"[VoiceCloner] Whisper medium loaded on GPU in {elapsed:.1f}s")
            else:
                raise RuntimeError("No CUDA")
        except Exception:
            self.whisper_model = WhisperModel(
                "medium",
                device="cpu",
                compute_type="int8",
            )
            elapsed = time.time() - t0
            logger.info(f"Whisper medium loaded on CPU in {elapsed:.1f}s")
            print(f"[VoiceCloner] Whisper medium loaded on CPU in {elapsed:.1f}s")

        self._initialized = True

    # ─────────────────────────────────────────────────────────
    # Main Pipeline
    # ─────────────────────────────────────────────────────────

    def clone_voice_from_audio(
        self,
        source_audio_path: str,
        target_speaker_wav: str,
        output_path: str,
        language: str = "en",
        voice_name: str = None
    ) -> str:
        """
        Full pipeline: transcribe → translate → synthesize.

        Args:
            source_audio_path: Audio from the source video.
            target_speaker_wav: Reference audio (voice selection hint).
            output_path: Path to save output audio.
            language: Target language code.
            voice_name: Specific persona key or Edge-TTS voice name.

        Returns:
            Path to the output audio file.
        """
        self.initialize()

        # Step 1: Transcribe with auto-detection
        print("[VoiceCloner] Step 1/4: Transcribing source audio (auto-detect)...")
        transcript, detected_lang, confidence = self._transcribe_audio_with_detection(
            source_audio_path
        )

        if not transcript or transcript.strip() == "":
            print("[VoiceCloner] Warning: Empty transcript — using fallback text.")
            transcript = "Hello, this is a voice cloning demonstration."
            detected_lang = "en"
            confidence = 0.0

        # Clean up the transcript
        transcript = self._clean_transcript(transcript)

        lang_name = LANG_NAMES.get(detected_lang, detected_lang)
        print(
            f"[VoiceCloner] Transcribed: {len(transcript)} chars, "
            f"detected={lang_name} ({detected_lang}), "
            f"confidence={confidence:.0%}"
        )
        print(f"[VoiceCloner] Transcript preview: {transcript[:150]}...")

        # Step 2: Translate if needed
        target_lang_code = self.TRANSLATE_LANG_MAP.get(language, language)
        detected_base = detected_lang.split("-")[0] if detected_lang else "en"
        target_base = language.split("-")[0] if language else "en"

        if detected_base != target_base:
            src_name = LANG_NAMES.get(detected_base, detected_base)
            tgt_name = LANG_NAMES.get(target_base, target_base)
            print(f"[VoiceCloner] Step 2/4: Translating {src_name} → {tgt_name}...")

            translated = self._translate_text(transcript, detected_lang, target_lang_code)
            if translated and translated.strip():
                print(f"[VoiceCloner] Translated: {len(translated)} chars")
                print(f"[VoiceCloner] Translated preview: {translated[:150]}...")
                transcript = translated
            else:
                print("[VoiceCloner] Translation returned empty — using original transcript.")
        else:
            print(f"[VoiceCloner] Step 2/4: No translation needed (source = target = {detected_base})")

        # Step 3: Select voice
        print("[VoiceCloner] Step 3/4: Selecting voice...")
        selected_voice = self._select_voice(voice_name, language)
        print(f"[VoiceCloner] Using voice: {selected_voice}")

        # Step 4: Synthesize
        print("[VoiceCloner] Step 4/4: Synthesizing speech with Edge-TTS...")
        self._synthesize_edge_tts(transcript, selected_voice, output_path)

        # Validate output
        if os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"[VoiceCloner] Output saved: {output_path} ({size_kb:.0f} KB)")
        else:
            raise RuntimeError(f"Synthesis failed — output file not created: {output_path}")

        return output_path

    # ─────────────────────────────────────────────────────────
    # Transcription
    # ─────────────────────────────────────────────────────────

    def _transcribe_audio_with_detection(self, audio_path: str) -> tuple:
        """
        Transcribe audio with automatic language detection.
        Uses VAD filter to skip silence/noise for better accuracy.

        Returns:
            (transcript_text, detected_language_code, confidence)
        """
        try:
            segments, info = self.whisper_model.transcribe(
                audio_path,
                beam_size=5,
                best_of=5,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                condition_on_previous_text=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=300,
                    threshold=0.35,
                ),
            )

            # Collect segments with timestamps
            all_segments = []
            for seg in segments:
                text = seg.text.strip()
                if text:
                    all_segments.append({
                        "text": text,
                        "start": seg.start,
                        "end": seg.end,
                        "avg_logprob": seg.avg_logprob,
                        "no_speech_prob": seg.no_speech_prob,
                    })

            # Filter out low-confidence / hallucinated segments
            good_segments = []
            for s in all_segments:
                if s["no_speech_prob"] > 0.6:
                    continue
                if s["avg_logprob"] < -1.5:
                    continue
                good_segments.append(s)

            text = " ".join(s["text"] for s in good_segments)
            detected = info.language or "en"
            confidence = info.language_probability or 0.0

            print(
                f"[VoiceCloner] Transcription: {len(all_segments)} segments, "
                f"{len(good_segments)} good, "
                f"lang={detected}, conf={confidence:.2f}"
            )

            return text.strip(), detected, confidence

        except Exception as e:
            print(f"[VoiceCloner] Transcription error: {e}")
            logger.error(f"Transcription error: {e}", exc_info=True)
            return "", "en", 0.0

    # ─────────────────────────────────────────────────────────
    # Translation
    # ─────────────────────────────────────────────────────────

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using Google Translate with retry logic
        and smart chunking for long texts.
        """
        src = source_lang.split("-")[0] if source_lang else "auto"
        if src in ("", "unknown"):
            src = "auto"

        max_chunk = 4500
        max_retries = 3

        try:
            if len(text) <= max_chunk:
                return self._translate_with_retry(text, src, target_lang, max_retries)

            # Split into sentence-based chunks
            chunks = self._smart_chunk_text(text, max_chunk)
            print(f"[VoiceCloner] Long text split into {len(chunks)} chunks for translation")

            translated_parts = []
            for i, chunk in enumerate(chunks):
                part = self._translate_with_retry(chunk, src, target_lang, max_retries)
                if part:
                    translated_parts.append(part)
                else:
                    print(f"[VoiceCloner] Chunk {i+1}/{len(chunks)} translation failed, keeping original")
                    translated_parts.append(chunk)

            return " ".join(translated_parts)

        except Exception as e:
            print(f"[VoiceCloner] Translation failed: {e}")
            logger.error(f"Translation failed: {e}", exc_info=True)
            return ""

    def _translate_with_retry(self, text: str, src: str, target: str, retries: int = 3) -> str:
        """Translate with exponential backoff retry."""
        for attempt in range(retries):
            try:
                result = GoogleTranslator(source=src, target=target).translate(text)
                if result and result.strip():
                    return result
            except Exception as e:
                wait = 2 ** attempt
                print(f"[VoiceCloner] Translation attempt {attempt+1}/{retries} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        return ""

    @staticmethod
    def _smart_chunk_text(text: str, max_chunk: int = 4500) -> list:
        """Split text at sentence boundaries, respecting max_chunk size."""
        sentences = re.split(r'(?<=[.!?।。])\s+', text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chunk:
                if current:
                    chunks.append(current.strip())
                if len(sentence) > max_chunk:
                    for i in range(0, len(sentence), max_chunk):
                        chunks.append(sentence[i:i + max_chunk])
                    current = ""
                else:
                    current = sentence
            else:
                current = f"{current} {sentence}" if current else sentence

        if current.strip():
            chunks.append(current.strip())
        return chunks

    # ─────────────────────────────────────────────────────────
    # Voice Selection
    # ─────────────────────────────────────────────────────────

    def _select_voice(self, voice_name: str, language: str) -> str:
        """Select the best voice for the given language and preference."""
        if voice_name and voice_name in VOICE_PERSONAS:
            return VOICE_PERSONAS[voice_name]
        if voice_name:
            return voice_name
        return DEFAULT_VOICES.get(language, "en-US-GuyNeural")

    # ─────────────────────────────────────────────────────────
    # Speech Synthesis
    # ─────────────────────────────────────────────────────────

    def _synthesize_edge_tts(self, text: str, voice: str, output_path: str):
        """
        Synthesize speech using Edge-TTS with natural rate/pitch.
        Handles async event loop correctly in both standalone and threaded contexts.
        """
        async def _do_synthesis():
            communicate = edge_tts.Communicate(
                text,
                voice,
                rate="-5%",
                pitch="+0Hz",
            )
            await communicate.save(output_path)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(lambda: asyncio.run(_do_synthesis())).result(timeout=120)
            else:
                loop.run_until_complete(_do_synthesis())
        except RuntimeError:
            asyncio.run(_do_synthesis())

    # ─────────────────────────────────────────────────────────
    # Text Cleaning
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean_transcript(text: str) -> str:
        """
        Clean Whisper transcript — remove artifacts, hallucinations, filler words.
        """
        if not text:
            return text

        # Remove repeated words (Whisper hallucination)
        text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1', text)

        # Remove filler sounds
        text = re.sub(r'\b(uh|um|hmm|ah|uhh|umm|erm)\b', '', text, flags=re.IGNORECASE)

        # Remove [music], (music), etc.
        text = re.sub(r'[\[\(].*?[\]\)]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text

    # ─────────────────────────────────────────────────────────
    # Audio Extraction
    # ─────────────────────────────────────────────────────────

    def extract_audio_from_video(self, video_path: str, output_audio_path: str) -> str:
        """Extract audio track from a video file at 16kHz mono for Whisper."""
        from moviepy import VideoFileClip

        print(f"[VoiceCloner] Extracting audio from {os.path.basename(video_path)}...")
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            raise ValueError("Video has no audio track.")

        clip.audio.write_audiofile(
            output_audio_path,
            fps=16000,
            nbytes=2,
            codec='pcm_s16le' if output_audio_path.endswith('.wav') else None,
            logger=None,
        )
        clip.close()

        size_kb = os.path.getsize(output_audio_path) / 1024
        print(f"[VoiceCloner] Audio extracted: {output_audio_path} ({size_kb:.0f} KB)")
        return output_audio_path

    # ─────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────

    def cleanup(self):
        """Free whisper model from memory."""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
            self._initialized = False

        import gc
        import sys
        mods_to_remove = [k for k in sys.modules if 'faster_whisper' in k or 'ctranslate2' in k]
        for mod in mods_to_remove:
            del sys.modules[mod]
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        print("[VoiceCloner] Whisper model freed from memory.")

    # ─────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def get_available_voices():
        """Return all available voice personas."""
        return VOICE_PERSONAS

    @staticmethod
    def get_voices_for_language(language: str):
        """Get available voices for a specific language."""
        prefix = language.split("-")[0] if "-" in language else language
        return {k: v for k, v in VOICE_PERSONAS.items() if k.startswith(prefix)}
