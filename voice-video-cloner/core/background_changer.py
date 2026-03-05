"""
Background Changer Module — Production Grade
AI-powered background replacement for video frames.

Features:
- U2-Net based person segmentation (via rembg)
- AI background generation via HuggingFace Inference API (FLUX.1-schnell / SDXL)
- Custom background image upload support
- Edge feathering for natural compositing
- Temporal mask smoothing to reduce flickering
- Efficient frame-level caching
"""

import os
import logging
import time
import cv2
import numpy as np
from typing import Optional, Tuple, Callable
from PIL import Image

logger = logging.getLogger("BackgroundChanger")


class BackgroundChanger:
    """
    Production-grade video background replacement.
    Uses rembg (U2-Net) for person segmentation and
    HuggingFace Inference API for text-to-image generation.
    """

    def __init__(self):
        self._session = None
        self._prev_mask = None
        self._frame_count = 0
        self._bg_cache = None
        self._bg_cache_size = None

    # ──────────────────────────────────────────────────────
    # Segmentation
    # ──────────────────────────────────────────────────────

    def _get_session(self):
        """Lazy-load rembg session with u2net_human_seg model."""
        if self._session is None:
            try:
                from rembg import new_session
                self._session = new_session("u2net_human_seg")
                logger.info("Loaded U2-Net human segmentation model")
            except ImportError:
                logger.error("rembg not installed. Run: pip install rembg")
                raise
        return self._session

    def segment_person(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment person from frame, returning alpha mask (0-255).
        Uses U2-Net human segmentation via rembg.
        Applies temporal smoothing to reduce flickering.
        """
        session = self._get_session()
        from rembg import remove

        # Convert BGR → RGB for rembg
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Get mask only (faster than full RGBA output)
        result = remove(
            pil_image,
            session=session,
            only_mask=True,
            post_process_mask=True,
        )

        mask = np.array(result)

        # Ensure single-channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Temporal smoothing: blend 75% current + 25% previous to reduce flicker
        if self._prev_mask is not None and mask.shape == self._prev_mask.shape:
            mask = cv2.addWeighted(
                mask.astype(np.float32), 0.75,
                self._prev_mask.astype(np.float32), 0.25,
                0
            ).astype(np.uint8)

        self._prev_mask = mask.copy()
        self._frame_count += 1
        return mask

    # ──────────────────────────────────────────────────────
    # AI Background Generation (HuggingFace Inference API)
    # ──────────────────────────────────────────────────────

    def generate_background(
        self,
        prompt: str,
        width: int = 1280,
        height: int = 720,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate a background image from text prompt using HuggingFace API.

        Tries FLUX.1-schnell (fastest) → SDXL as fallback.
        Set HF_TOKEN env variable for higher rate limits.

        Returns:
            BGR numpy array of the generated background.
        """
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise RuntimeError("huggingface_hub not installed")

        token = os.environ.get("HF_TOKEN", None)
        client = InferenceClient(token=token)

        # Enhance the prompt for better background quality
        enhanced = (
            f"high quality professional photograph, {prompt}, "
            "detailed, sharp focus, 4k resolution, no people, no text, no watermark"
        )

        # Models to try in order (fast → quality)
        models = [
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ]

        image = None
        for model_id in models:
            try:
                logger.info(f"Generating background with {model_id}...")
                t0 = time.time()
                image = client.text_to_image(
                    prompt=enhanced,
                    model=model_id,
                    width=width,
                    height=height,
                )
                logger.info(f"Background generated in {time.time()-t0:.1f}s with {model_id}")
                break
            except Exception as e:
                logger.warning(f"Model {model_id} failed: {e}")
                continue

        if image is None:
            raise RuntimeError(
                "All background generation models failed. "
                "Set HF_TOKEN environment variable or check your internet connection."
            )

        # PIL Image → numpy BGR
        bg_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if output_path:
            cv2.imwrite(output_path, bg_bgr)
            logger.info(f"Background saved to {output_path}")

        return bg_bgr

    # ──────────────────────────────────────────────────────
    # Compositing
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _feather_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
        """Gaussian blur on mask edges for natural blending."""
        ksize = radius * 2 + 1
        return cv2.GaussianBlur(mask, (ksize, ksize), 0)

    def _resize_background(self, bg: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Resize + center-crop background to match frame dimensions. Result is cached."""
        key = (target_w, target_h)
        if self._bg_cache is not None and self._bg_cache_size == key:
            return self._bg_cache

        bg_h, bg_w = bg.shape[:2]
        target_ratio = target_w / target_h
        bg_ratio = bg_w / bg_h

        if bg_ratio > target_ratio:
            new_h = target_h
            new_w = int(bg_w * (target_h / bg_h))
        else:
            new_w = target_w
            new_h = int(bg_h * (target_w / bg_w))

        resized = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Center crop
        x0 = (new_w - target_w) // 2
        y0 = (new_h - target_h) // 2
        cropped = resized[y0:y0 + target_h, x0:x0 + target_w]

        self._bg_cache = cropped
        self._bg_cache_size = key
        return cropped

    def apply_background(
        self,
        frame: np.ndarray,
        background: np.ndarray,
        feather_radius: int = 5,
    ) -> np.ndarray:
        """
        Replace frame background with given background image.
        1. Segment person  2. Feather mask  3. Composite
        """
        h, w = frame.shape[:2]

        # 1. Segment person (mask: person = 255, bg = 0)
        mask = self.segment_person(frame)

        # 2. Feather edges
        mask = self._feather_mask(mask, feather_radius)

        # 3. Resize background to match frame
        bg = self._resize_background(background, w, h)

        # 4. Alpha composite
        alpha = mask.astype(np.float32) / 255.0
        alpha_3ch = np.stack([alpha] * 3, axis=-1)

        composite = (
            frame.astype(np.float32) * alpha_3ch +
            bg.astype(np.float32) * (1.0 - alpha_3ch)
        )
        return composite.astype(np.uint8)

    # ──────────────────────────────────────────────────────
    # Full Video Background Processing
    # ──────────────────────────────────────────────────────

    def process_video_background(
        self,
        input_path: str,
        output_path: str,
        background: np.ndarray,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Replace background in every frame of a video.

        Args:
            input_path: Source video with original background.
            output_path: Output video with replaced background.
            background: Background image (BGR numpy array).
            progress_callback: fn(percent, message)
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Background change: {width}x{height} @ {fps:.1f}fps, {total} frames")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        count = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                result = self.apply_background(frame, background)
            except Exception as e:
                logger.warning(f"BG frame {count} failed: {e}, using original")
                result = frame

            writer.write(result)
            count += 1

            if count % 5 == 0 and progress_callback:
                pct = int((count / max(total, 1)) * 100)
                elapsed = time.time() - t0
                fps_actual = count / max(elapsed, 0.001)
                remaining = (total - count) / max(fps_actual, 0.001)
                progress_callback(
                    min(pct, 99),
                    f"BG frame {count}/{total} ({fps_actual:.1f} fps, ~{remaining:.0f}s left)"
                )

        cap.release()
        writer.release()

        elapsed = time.time() - t0
        avg_fps = count / max(elapsed, 0.001)
        logger.info(f"Background done: {count} frames in {elapsed:.1f}s ({avg_fps:.1f} fps)")

    # ──────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────

    def cleanup(self):
        """Release all resources."""
        self._session = None
        self._prev_mask = None
        self._bg_cache = None
        self._bg_cache_size = None
        self._frame_count = 0
        logger.info("BackgroundChanger cleanup complete")
