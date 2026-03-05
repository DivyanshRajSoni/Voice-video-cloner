"""
Video Processor Module
Orchestrates the full pipeline: face swap + voice clone + recombine.
"""

import os
import uuid
import time
import cv2
import numpy as np
from typing import Optional, Callable

from core.face_swapper import FaceSwapper
from core.voice_cloner import VoiceCloner


class VideoProcessor:
    """
    Orchestrates the complete video cloning pipeline:
    1. Extract audio from source video
    2. Process video frames (face swap)
    3. Clone voice
    4. Recombine processed video + cloned audio
    """

    def __init__(self, models_dir: str = "models", output_dir: str = "outputs"):
        """
        Initialize VideoProcessor.
        
        Args:
            models_dir: Directory containing AI models.
            output_dir: Directory to save output files.
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.face_swapper = FaceSwapper(model_dir=models_dir)
        self.voice_cloner = VoiceCloner()

        os.makedirs(output_dir, exist_ok=True)

    def process(
        self,
        source_video_path: str,
        target_face_path: str,
        target_voice_path: str,
        language: str = "en",
        voice_name: str = "",
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        Run the full cloning pipeline.
        
        Args:
            source_video_path: Path to the source video (your 30s video).
            target_face_path: Path to target persona face (image or video).
            target_voice_path: Path to target persona voice audio/video.
            language: Language for voice synthesis.
            voice_name: Specific voice persona key or Edge-TTS voice name.
            progress_callback: Optional callback fn(stage, percent, message).
            
        Returns:
            Dict with output file paths and metadata.
        """
        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join(self.output_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        result = {
            "job_id": job_id,
            "status": "processing",
            "stages": {},
            "output_video": None,
            "error": None,
        }

        try:
            # ──────────────────────────────────────
            # Stage 1: Extract audio from source
            # ──────────────────────────────────────
            self._update_progress(progress_callback, "extract_audio", 0, "Extracting audio from source video...")
            source_audio_path = os.path.join(job_dir, "source_audio.wav")
            self.voice_cloner.extract_audio_from_video(source_video_path, source_audio_path)
            result["stages"]["extract_audio"] = "complete"
            self._update_progress(progress_callback, "extract_audio", 100, "Audio extracted.")

            # ──────────────────────────────────────
            # Stage 2: Clone voice
            # ──────────────────────────────────────
            self._update_progress(progress_callback, "voice_clone", 0, "Cloning voice...")

            # Extract target voice reference
            target_voice_ref = target_voice_path
            if target_voice_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                target_voice_ref = os.path.join(job_dir, "target_voice_ref.wav")
                self.voice_cloner.extract_audio_from_video(target_voice_path, target_voice_ref)

            cloned_audio_path = os.path.join(job_dir, "cloned_audio.mp3")
            self.voice_cloner.clone_voice_from_audio(
                source_audio_path=source_audio_path,
                target_speaker_wav=target_voice_ref,
                output_path=cloned_audio_path,
                language=language,
                voice_name=voice_name if voice_name else None
            )
            result["stages"]["voice_clone"] = "complete"
            self._update_progress(progress_callback, "voice_clone", 100, "Voice cloned.")

            # Free voice cloner models to release RAM for face swap
            self._update_progress(progress_callback, "voice_clone", 100, "Freeing memory for face swap...")
            self.voice_cloner.cleanup()
            import gc
            gc.collect()

            # ──────────────────────────────────────
            # Stage 3: Face swap video frames
            # ──────────────────────────────────────
            self._update_progress(progress_callback, "face_swap", 0, "Starting face swap...")

            # Get reference face from target
            if target_face_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                target_face = self.face_swapper.extract_reference_face_from_video(target_face_path)
            else:
                target_face = self.face_swapper.extract_reference_face(target_face_path)

            swapped_video_path = os.path.join(job_dir, "swapped_video.mp4")
            self._process_video_frames(
                source_video_path, swapped_video_path, target_face, progress_callback
            )
            result["stages"]["face_swap"] = "complete"
            self._update_progress(progress_callback, "face_swap", 100, "Face swap complete.")

            # ──────────────────────────────────────
            # Stage 4: Combine video + cloned audio
            # ──────────────────────────────────────
            self._update_progress(progress_callback, "combine", 0, "Combining video and cloned audio...")
            output_video_path = os.path.join(job_dir, f"output_{job_id}.mp4")
            self._combine_video_audio(swapped_video_path, cloned_audio_path, output_video_path)
            result["stages"]["combine"] = "complete"
            result["output_video"] = output_video_path
            result["status"] = "complete"
            self._update_progress(progress_callback, "combine", 100, "Done! Output video ready.")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"[VideoProcessor] Error: {e}")
            import traceback
            traceback.print_exc()

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
        
        Args:
            input_path: Source video path.
            output_path: Output video path (no audio).
            target_face: Reference face for swapping.
            progress_callback: Progress callback function.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Swap face in frame
            processed_frame = self.face_swapper.process_frame(frame, target_face)
            writer.write(processed_frame)

            frame_count += 1
            if frame_count % 10 == 0 and progress_callback:
                pct = int((frame_count / max(total_frames, 1)) * 100)
                elapsed = time.time() - start_time
                fps_actual = frame_count / max(elapsed, 0.001)
                remaining = (total_frames - frame_count) / max(fps_actual, 0.001)
                self._update_progress(
                    progress_callback, "face_swap", pct,
                    f"Processing frame {frame_count}/{total_frames} "
                    f"({fps_actual:.1f} fps, ~{remaining:.0f}s remaining)"
                )

        cap.release()
        writer.release()
        print(f"[VideoProcessor] Processed {frame_count} frames in {time.time() - start_time:.1f}s")

    def _combine_video_audio(self, video_path: str, audio_path: str, output_path: str):
        """
        Combine processed video with cloned audio using moviepy.
        
        Args:
            video_path: Path to the face-swapped video (no audio).
            audio_path: Path to the cloned audio.
            output_path: Final output video path.
        """
        from moviepy import VideoFileClip, AudioFileClip

        print("[VideoProcessor] Combining video and audio...")
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
            logger=None
        )
        video.close()
        audio.close()
        final.close()
        print(f"[VideoProcessor] Output saved to {output_path}")

    def _update_progress(self, callback, stage, percent, message):
        """Send progress update via callback."""
        if callback:
            callback(stage, percent, message)
        print(f"[{stage}] {percent}% - {message}")
