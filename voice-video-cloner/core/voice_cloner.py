"""
Voice Cloner Module
Uses Edge-TTS (Microsoft neural voices) for speech synthesis,
faster-whisper for transcription, and deep-translator for translation.
Transcribes source speech → translates to target language → synthesizes in neural voice.
"""

import os
import asyncio
import edge_tts
from deep_translator import GoogleTranslator


# ─── Available voice personas ──────────────────────────────────
VOICE_PERSONAS = {
    # English
    "en-male-1": "en-US-GuyNeural",
    "en-male-2": "en-US-DavisNeural",
    "en-male-3": "en-GB-RyanNeural",
    "en-female-1": "en-US-JennyNeural",
    "en-female-2": "en-US-AriaNeural",
    "en-female-3": "en-GB-SoniaNeural",
    # Spanish
    "es-male-1": "es-ES-AlvaroNeural",
    "es-female-1": "es-ES-ElviraNeural",
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


class VoiceCloner:
    """
    Voice transformation using Edge-TTS neural voices
    and faster-whisper for speech transcription.
    """

    def __init__(self):
        self.whisper_model = None
        self._initialized = False

    def initialize(self):
        """Load the Whisper transcription model."""
        if self._initialized:
            return

        print("[VoiceCloner] Loading faster-whisper model...")
        from faster_whisper import WhisperModel
        self.whisper_model = WhisperModel("base", compute_type="int8")
        self._initialized = True
        print("[VoiceCloner] Whisper model loaded.")

    # ── Language code mapping for Google Translate ──
    TRANSLATE_LANG_MAP = {
        "en": "en", "hi": "hi", "es": "es", "fr": "fr", "de": "de",
        "it": "it", "pt": "pt", "pl": "pl", "tr": "tr", "ru": "ru",
        "nl": "nl", "cs": "cs", "ar": "ar", "zh-cn": "zh-CN", "zh": "zh-CN",
        "ja": "ja", "ko": "ko",
    }

    def clone_voice_from_audio(
        self,
        source_audio_path: str,
        target_speaker_wav: str,
        output_path: str,
        language: str = "en",
        voice_name: str = None
    ) -> str:
        """
        Transform voice: transcribe source audio, translate to target
        language, then re-synthesize using a neural voice.
        """
        self.initialize()

        # Step 1: Transcribe source audio (auto-detect language)
        print("[VoiceCloner] Transcribing source audio (auto-detect language)...")
        transcript, detected_lang = self._transcribe_audio_with_detection(source_audio_path)

        if not transcript or transcript.strip() == "":
            print("[VoiceCloner] Warning: Empty transcript. Using fallback text.")
            transcript = "Hello, this is a voice cloning demonstration."
            detected_lang = "en"

        print(f"[VoiceCloner] Transcript ({len(transcript)} chars, detected={detected_lang}): {transcript[:120]}...")

        # Step 2: Translate to target language (if different from detected)
        target_lang_code = self.TRANSLATE_LANG_MAP.get(language, language)
        detected_base = detected_lang.split("-")[0] if detected_lang else "en"
        target_base = language.split("-")[0] if language else "en"

        if detected_base != target_base:
            print(f"[VoiceCloner] Translating {detected_lang} → {language}...")
            translated = self._translate_text(transcript, detected_lang, target_lang_code)
            if translated:
                print(f"[VoiceCloner] Translated ({len(translated)} chars): {translated[:120]}...")
                transcript = translated
            else:
                print("[VoiceCloner] Translation failed, using original transcript.")
        else:
            print(f"[VoiceCloner] Source and target language match ({detected_lang}), no translation needed.")

        # Step 3: Select voice
        if voice_name and voice_name in VOICE_PERSONAS:
            selected_voice = VOICE_PERSONAS[voice_name]
        elif voice_name:
            selected_voice = voice_name
        else:
            selected_voice = DEFAULT_VOICES.get(language, "en-US-GuyNeural")

        print(f"[VoiceCloner] Using voice: {selected_voice}")

        # Step 4: Synthesize with Edge-TTS
        print("[VoiceCloner] Synthesizing speech...")
        self._synthesize_edge_tts(transcript, selected_voice, output_path)

        print(f"[VoiceCloner] Output saved to {output_path}")
        return output_path

    def _transcribe_audio_with_detection(self, audio_path: str) -> tuple:
        """
        Transcribe audio using faster-whisper with automatic language detection.
        Returns (transcript_text, detected_language_code).
        """
        try:
            segments, info = self.whisper_model.transcribe(
                audio_path,
                beam_size=5
            )
            text = " ".join([seg.text for seg in segments])
            detected = info.language or "en"
            print(f"[VoiceCloner] Detected language: {detected} (prob: {info.language_probability:.2f})")
            return text.strip(), detected
        except Exception as e:
            print(f"[VoiceCloner] Transcription error: {e}")
            return "", "en"

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using Google Translate (via deep-translator).
        Handles long texts by chunking at ~4500 chars.
        """
        try:
            src = source_lang.split("-")[0] if source_lang else "auto"
            # Use 'auto' for detection if source is uncertain
            if src in ("", "unknown"):
                src = "auto"

            max_chunk = 4500
            if len(text) <= max_chunk:
                result = GoogleTranslator(source=src, target=target_lang).translate(text)
                return result or text

            # Chunk long text at sentence boundaries
            chunks = []
            current = ""
            for sentence in text.replace(". ", ".|").split("|"):
                if len(current) + len(sentence) > max_chunk:
                    if current:
                        chunks.append(current)
                    current = sentence
                else:
                    current += sentence
            if current:
                chunks.append(current)

            translated_parts = []
            for chunk in chunks:
                part = GoogleTranslator(source=src, target=target_lang).translate(chunk)
                translated_parts.append(part or chunk)

            return " ".join(translated_parts)
        except Exception as e:
            print(f"[VoiceCloner] Translation error: {e}")
            return ""

    def _synthesize_edge_tts(self, text: str, voice: str, output_path: str):
        """Synthesize speech using Edge-TTS."""
        async def _do_synthesis():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(lambda: asyncio.run(_do_synthesis())).result()
            else:
                loop.run_until_complete(_do_synthesis())
        except RuntimeError:
            asyncio.run(_do_synthesis())

    def extract_audio_from_video(self, video_path: str, output_audio_path: str) -> str:
        """
        Extract audio track from a video file using moviepy.
        
        Args:
            video_path: Path to the video file.
            output_audio_path: Path to save the extracted audio.
            
        Returns:
            Path to the extracted audio file.
        """
        from moviepy import VideoFileClip

        print(f"[VoiceCloner] Extracting audio from {video_path}...")
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            raise ValueError("Video has no audio track.")
        clip.audio.write_audiofile(output_audio_path, logger=None)
        clip.close()
        print(f"[VoiceCloner] Audio extracted to {output_audio_path}")
        return output_audio_path

    def cleanup(self):
        """Free whisper model from memory."""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
            self._initialized = False
        # Force garbage collection to reclaim RAM
        import gc
        import sys
        # Remove cached faster_whisper modules to free ctranslate2 memory
        mods_to_remove = [k for k in sys.modules if 'faster_whisper' in k or 'ctranslate2' in k]
        for mod in mods_to_remove:
            del sys.modules[mod]
        gc.collect()
        print("[VoiceCloner] Models freed from memory.")

    @staticmethod
    def get_available_voices():
        """Return all available voice personas."""
        return VOICE_PERSONAS

    @staticmethod
    def get_voices_for_language(language: str):
        """Get available voices for a specific language."""
        prefix = language.split("-")[0] if "-" in language else language
        return {k: v for k, v in VOICE_PERSONAS.items() if k.startswith(prefix)}
