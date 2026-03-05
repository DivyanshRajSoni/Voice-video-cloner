"""
Video Processor Module — Production Grade
Orchestrates the full pipeline: face swap + voice clone + recombine.

Production features:
- High-quality H.264 encoding (CRF 18, veryslow preset)
- Proper frame skipping and face caching for speed
- Memory management between pipeline stages
- Comprehensive progress reporting
- Automatic cleanup of intermediate files
- Robust error reporting with traceback
"""

import os
import uuid
import time
import logging
import cv2
import numpy as np
from typing import Optional, Callable

from core.face_swapper import FaceSwapper
from core.voice_cloner import VoiceCloner
from core.background_changer import BackgroundChanger

logger = logging.getLogger("VideoProcessor")


class VideoProcessor:
    """
    Production-grade video cloning pipeline:
    1. Extract audio from source video
    2. Clone voice (transcribe → translate → synthesize)
    3. Process video frames (face swap)
    4. AI background replacement (optional)
    5. Combine processed video + cloned audio (high-quality encoding)
    """

    def __init__(self, models_dir: str = "models", output_dir: str = "outputs"):
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.face_swapper = FaceSwapper(model_dir=models_dir)
        self.voice_cloner = VoiceCloner()
        self.background_changer = BackgroundChanger()
        os.makedirs(output_dir, exist_ok=True)

    def process(
        self,
        source_video_path: str,
        target_face_path: str,
        target_voice_path: str,
        language: str = "en",
        voice_name: str = "",
        bg_image_path: Optional[str] = None,
        bg_prompt: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        Run the full cloning pipeline.

        Args:
            source_video_path: Path to the source video.
            target_face_path: Path to target persona face image/video.
            target_voice_path: Path to target persona voice audio/video.
            language: Target language for voice synthesis.
            voice_name: Specific voice persona key.
            bg_image_path: Path to custom background image (optional).
            bg_prompt: Text prompt for AI background generation (optional).
            progress_callback: fn(stage, percent, message).

        Returns:
            Dict with output paths, metadata, and processing stats.
        """
        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join(self.output_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        pipeline_start = time.time()
        result = {
            "job_id": job_id,
            "status": "processing",
            "stages": {},
            "output_video": None,
            "error": None,
            "stats": {},
        }

        try:
            # ──────────────────────────────────────
            # Stage 1: Extract audio from source
            # ──────────────────────────────────────
            self._update_progress(progress_callback, "extract_audio", 0,
                                  "Extracting audio from source video...")
            t0 = time.time()
            source_audio_path = os.path.join(job_dir, "source_audio.wav")
            self.voice_cloner.extract_audio_from_video(source_video_path, source_audio_path)
            result["stages"]["extract_audio"] = f"complete ({time.time()-t0:.1f}s)"
            self._update_progress(progress_callback, "extract_audio", 100,
                                  "Audio extracted.")

            # ──────────────────────────────────────
            # Stage 2: Clone voice
            # ──────────────────────────────────────
            self._update_progress(progress_callback, "voice_clone", 0,
                                  "Transcribing & translating speech...")
            t0 = time.time()

            # Extract target voice reference
            target_voice_ref = target_voice_path
            if target_voice_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                target_voice_ref = os.path.join(job_dir, "target_voice_ref.wav")
                self.voice_cloner.extract_audio_from_video(target_voice_path, target_voice_ref)

            self._update_progress(progress_callback, "voice_clone", 30,
                                  "Cloning voice with neural synthesis...")

            cloned_audio_path = os.path.join(job_dir, "cloned_audio.mp3")
            self.voice_cloner.clone_voice_from_audio(
                source_audio_path=source_audio_path,
                target_speaker_wav=target_voice_ref,
                output_path=cloned_audio_path,
                language=language,
                voice_name=voice_name if voice_name else None
            )
            result["stages"]["voice_clone"] = f"complete ({time.time()-t0:.1f}s)"
            self._update_progress(progress_callback, "voice_clone", 100,
                                  "Voice cloned successfully.")

            # Free voice cloner models to release VRAM for face swap
            self._update_progress(progress_callback, "voice_clone", 100,
                                  "Freeing GPU memory for face swap...")
            self.voice_cloner.cleanup()
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            # ──────────────────────────────────────
            # Stage 3: Face swap video frames
            # ──────────────────────────────────────
            self._update_progress(progress_callback, "face_swap", 0,
                                  "Loading face detection models...")
            t0 = time.time()

            # Get reference face from target
            if target_face_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                target_face = self.face_swapper.extract_reference_face_from_video(target_face_path)
            else:
                target_face = self.face_swapper.extract_reference_face(target_face_path)

            self._update_progress(progress_callback, "face_swap", 5,
                                  "Face detected. Processing video frames...")

            swapped_video_path = os.path.join(job_dir, "swapped_video.mp4")
            self._process_video_frames(
                source_video_path, swapped_video_path, target_face, progress_callback
            )

            # Log face swap stats
            swap_stats = self.face_swapper.get_processing_stats()
            result["stats"]["face_swap"] = swap_stats
            print(f"[VideoProcessor] Face swap stats: {swap_stats}")

            result["stages"]["face_swap"] = f"complete ({time.time()-t0:.1f}s)"
            self._update_progress(progress_callback, "face_swap", 100,
                                  f"Face swap complete ({swap_stats['swap_rate']} frames swapped).")

            # ──────────────────────────────────────
            # Stage 4: Background Change (optional)
            # ──────────────────────────────────────
            video_for_combine = swapped_video_path

            if bg_image_path or bg_prompt:
                self._update_progress(progress_callback, "bg_change", 0,
                                      "Preparing AI background replacement...")
                t0 = time.time()

                # Free face swap GPU memory before bg processing
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                # Get or generate background image
                if bg_image_path and os.path.exists(bg_image_path):
                    bg_image = cv2.imread(bg_image_path)
                    if bg_image is None:
                        raise ValueError(f"Cannot read background image: {bg_image_path}")
                    logger.info(f"Using uploaded background: {bg_image_path}")
                elif bg_prompt:
                    self._update_progress(progress_callback, "bg_change", 5,
                                          f"Generating background: '{bg_prompt[:50]}...'")
                    bg_save_path = os.path.join(job_dir, "generated_bg.png")
                    bg_image = self.background_changer.generate_background(
                        prompt=bg_prompt,
                        width=1280,
                        height=720,
                        output_path=bg_save_path,
                    )
                    logger.info(f"AI background generated from prompt")
                else:
                    bg_image = None

                if bg_image is not None:
                    self._update_progress(progress_callback, "bg_change", 15,
                                          "Replacing background frame by frame...")
                    bg_video_path = os.path.join(job_dir, "bg_video.mp4")

                    def bg_progress(pct, msg):
                        # Scale: 15-95% of bg_change stage
                        scaled = 15 + int(pct * 0.8)
                        self._update_progress(progress_callback, "bg_change", scaled, msg)

                    self.background_changer.process_video_background(
                        input_path=swapped_video_path,
                        output_path=bg_video_path,
                        background=bg_image,
                        progress_callback=bg_progress,
                    )
                    video_for_combine = bg_video_path

                result["stages"]["bg_change"] = f"complete ({time.time()-t0:.1f}s)"
                self._update_progress(progress_callback, "bg_change", 100,
                                      "Background replaced successfully.")

                # Free segmentation model memory
                self.background_changer.cleanup()
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            # ──────────────────────────────────────
            # Stage 5: Combine video + cloned audio
            # ──────────────────────────────────────
            self._update_progress(progress_callback, "combine", 0,
                                  "Encoding final video with high-quality settings...")
            t0 = time.time()
            output_video_path = os.path.join(job_dir, f"output_{job_id}.mp4")
            self._combine_video_audio(video_for_combine, cloned_audio_path, output_video_path)
            result["stages"]["combine"] = f"complete ({time.time()-t0:.1f}s)"

            # Output file stats
            if os.path.exists(output_video_path):
                size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
                result["stats"]["output_size_mb"] = round(size_mb, 2)

            result["output_video"] = output_video_path
            result["status"] = "complete"
            total_time = time.time() - pipeline_start
            result["stats"]["total_time_seconds"] = round(total_time, 1)
            self._update_progress(progress_callback, "combine", 100,
                                  f"Done! Output video ready ({total_time:.0f}s total).")
            print(f"[VideoProcessor] Pipeline complete in {total_time:.1f}s")

            # Cleanup intermediate files to save disk space
            self._cleanup_intermediates(job_dir, output_video_path)

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"[VideoProcessor] Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Pipeline error: {e}", exc_info=True)

        return result

    def _process_video_frames(
        self,
        input_path: str,
        output_path: str,
        target_face,
        progress_callback: Optional[Callable] = None
    ):
        """
        Process all video frames with face swapping.
        Uses XVID for intermediate (fast write), final encoding is H.264 in combine step.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[VideoProcessor] Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame (face swap handles errors internally)
            processed_frame = self.face_swapper.process_frame(frame, target_face)
            writer.write(processed_frame)

            frame_count += 1
            if frame_count % 10 == 0 and progress_callback:
                pct = int((frame_count / max(total_frames, 1)) * 100)
                elapsed = time.time() - start_time
                fps_actual = frame_count / max(elapsed, 0.001)
                remaining = (total_frames - frame_count) / max(fps_actual, 0.001)
                self._update_progress(
                    progress_callback, "face_swap", min(pct, 99),
                    f"Frame {frame_count}/{total_frames} "
                    f"({fps_actual:.1f} fps, ~{remaining:.0f}s remaining)"
                )

        cap.release()
        writer.release()

        elapsed = time.time() - start_time
        avg_fps = frame_count / max(elapsed, 0.001)
        print(f"[VideoProcessor] Processed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} fps)")

    def _combine_video_audio(self, video_path: str, audio_path: str, output_path: str):
        """
        Combine processed video with cloned audio.
        Uses high-quality H.264 encoding for production output.
        """
        from moviepy import VideoFileClip, AudioFileClip

        print("[VideoProcessor] Combining video + audio (high-quality encoding)...")
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Match audio duration to video duration
        if audio.duration > video.duration:
            audio = audio.subclipped(0, video.duration)

        final = video.with_audio(audio)
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            bitrate="5000k",           # 5 Mbps for good quality
            audio_bitrate="192k",      # High quality audio
            preset="medium",           # Balance: quality vs speed
            ffmpeg_params=[
                "-crf", "20",          # Constant rate factor (lower = better, 18-23 recommended)
                "-pix_fmt", "yuv420p", # Ensure compatibility with all players
                "-movflags", "+faststart",  # Web streaming optimization
            ],
            logger=None
        )
        video.close()
        audio.close()
        final.close()

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[VideoProcessor] Output: {output_path} ({size_mb:.1f} MB)")

    @staticmethod
    def _cleanup_intermediates(job_dir: str, keep_path: str):
        """Remove intermediate files to save disk space."""
        try:
            for fname in os.listdir(job_dir):
                fpath = os.path.join(job_dir, fname)
                if fpath != keep_path and os.path.isfile(fpath):
                    os.remove(fpath)
        except Exception as e:
            print(f"[VideoProcessor] Cleanup warning: {e}")

    def _update_progress(self, callback, stage, percent, message):
        """Send progress update via callback."""
        if callback:
            callback(stage, percent, message)
        print(f"[{stage}] {percent}% - {message}")
