"""
Face Swapper Module – Hybrid Approach
- Real / semi-realistic faces → InsightFace FaceAnalysis + inswapper_128 (high-quality face swap)
- Cartoon / stylised images   → automatic fallback to affine-warp + seamlessClone overlay

Uses FaceAnalysis (buffalo_l) with GPU acceleration (CUDA) when available.
"""

import os
import gc
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort


class FaceSwapper:
    """Face detection, swapping, and cartoon-overlay using InsightFace + OpenCV."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.face_app = None       # FaceAnalysis (det + rec + landmark + genderage)
        self.swap_model = None     # inswapper_128
        self._initialized = False

    # ── Initialization ─────────────────────────────────────────────

    def initialize(self):
        """Load FaceAnalysis pipeline + inswapper model."""
        if self._initialized:
            return

        gc.collect()
        providers = self._get_providers()

        # Full FaceAnalysis pipeline (buffalo_l: det_10g, w600k_r50, etc.)
        print("[FaceSwapper] Loading FaceAnalysis (buffalo_l) …")
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            root=self.model_dir,
            providers=providers,
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
        print("[FaceSwapper] FaceAnalysis loaded.")
        gc.collect()

        # Face swapper model (~530 MB)
        swap_path = os.path.join(self.model_dir, "inswapper_128.onnx")
        if not os.path.exists(swap_path):
            raise FileNotFoundError(
                f"Face swap model not found at {swap_path}. "
                f"Download from https://github.com/facefusion/facefusion-assets/releases"
            )
        print(f"[FaceSwapper] Loading swapper ({os.path.getsize(swap_path) // 1024 // 1024} MB) …")
        self.swap_model = insightface.model_zoo.get_model(swap_path, providers=providers)
        print("[FaceSwapper] Swapper loaded.")

        self._initialized = True
        gc.collect()
        print("[FaceSwapper] Initialization complete (FaceAnalysis + inswapper).")

    @staticmethod
    def _get_providers():
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    # ── Face Detection ─────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray, det_thresh=None):
        """Detect faces using FaceAnalysis. Returns list of Face objects."""
        self.initialize()
        if det_thresh is not None:
            old = self.face_app.det_model.det_thresh
            self.face_app.det_model.det_thresh = det_thresh

        faces = self.face_app.get(frame)

        if det_thresh is not None:
            self.face_app.det_model.det_thresh = old
        return faces

    def get_best_face(self, frame: np.ndarray, lenient=False):
        """Return the largest face detected in *frame*."""
        _largest = lambda ff: max(ff, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        faces = self.detect_faces(frame)
        if faces:
            return _largest(faces)

        if lenient:
            faces = self.detect_faces(frame, det_thresh=0.1)
            if faces:
                return _largest(faces)
            # Upscale tiny images
            h, w = frame.shape[:2]
            if max(h, w) < 400:
                scale = 640 / max(h, w)
                resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
                faces = self.detect_faces(resized, det_thresh=0.1)
                if faces:
                    return _largest(faces)
        return None

    # ── Reference Face Extraction ──────────────────────────────────

    def extract_reference_face(self, image_path: str):
        """
        Extract reference face from an image.

        Returns:
            insightface Face object  – for real / semi-realistic faces (inswapper path)
            dict with is_cartoon=True – for cartoon / stylised faces  (overlay path)
        """
        self.initialize()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Try InsightFace with progressively aggressive strategies
        face = self._try_insightface_detection(img)
        if face is not None:
            print(f"[FaceSwapper] Reference face detected via InsightFace: {image_path}")
            return face

        # Fallback → prepare for cartoon overlay
        print(f"[FaceSwapper] InsightFace could not detect face → using cartoon overlay: {image_path}")
        return self._prepare_cartoon_overlay(img)

    def _try_insightface_detection(self, img: np.ndarray):
        """Try several preprocessing strategies to detect a face with InsightFace."""
        _largest = lambda ff: max(ff, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        h, w = img.shape[:2]

        # 1. Normal with progressively lower thresholds
        for thresh in [0.5, 0.3, 0.1, 0.05]:
            faces = self.detect_faces(img, det_thresh=thresh)
            if faces:
                return _largest(faces)

        # 2. Upscale small images to 1024px
        if max(h, w) < 1024:
            scale = 1024 / max(h, w)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            for thresh in [0.3, 0.1, 0.05]:
                faces = self.detect_faces(resized, det_thresh=thresh)
                if faces:
                    return _largest(faces)

        # 3. Contrast enhancement + sharpen
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        for thresh in [0.3, 0.1, 0.05]:
            faces = self.detect_faces(enhanced, det_thresh=thresh)
            if faces:
                return _largest(faces)

        return None

    # ── Cartoon Overlay ────────────────────────────────────────────

    def _prepare_cartoon_overlay(self, img: np.ndarray) -> dict:
        """Prepare cartoon image for warp-and-blend overlay on video frames."""
        h, w = img.shape[:2]

        # Center-crop to ~65 % to isolate the face from background
        crop_r = 0.65
        ch, cw = int(h * crop_r), int(w * crop_r)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        face_crop = img[y0:y0 + ch, x0:x0 + cw].copy()

        # Soft elliptical mask
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cx, cy = cw // 2, ch // 2
        axes = (int(cw * 0.44), int(ch * 0.48))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 11)

        # Estimated 5-point keypoints (proportional to crop)
        src_kps = np.float32([
            [cw * 0.35, ch * 0.35],   # left eye
            [cw * 0.65, ch * 0.35],   # right eye
            [cw * 0.50, ch * 0.50],   # nose
            [cw * 0.38, ch * 0.65],   # left mouth corner
            [cw * 0.62, ch * 0.65],   # right mouth corner
        ])

        return {
            "is_cartoon": True,
            "face_crop": face_crop,
            "mask": mask,
            "src_kps": src_kps,
        }

    def _overlay_cartoon_on_frame(self, frame: np.ndarray, cartoon: dict) -> np.ndarray:
        """Warp the cartoon face onto a detected face in the video frame and blend."""
        source_face = self.get_best_face(frame)
        if source_face is None:
            return frame  # no face in this frame → return unchanged

        dst_kps = source_face.kps.astype(np.float32)  # (5, 2)

        # Affine transform using left-eye, right-eye, nose
        src_pts = cartoon["src_kps"][:3]
        dst_pts = dst_kps[:3]
        M = cv2.getAffineTransform(src_pts, dst_pts)

        fh, fw = frame.shape[:2]
        warped = cv2.warpAffine(
            cartoon["face_crop"], M, (fw, fh),
            borderMode=cv2.BORDER_REFLECT_101,
        )
        warped_mask = cv2.warpAffine(
            cartoon["mask"], M, (fw, fh),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        # Mask centre for seamlessClone
        ys, xs = np.where(warped_mask > 128)
        if len(ys) == 0:
            return frame
        center = (int(np.mean(xs)), int(np.mean(ys)))

        # Binary mask; must not touch frame border (seamlessClone requirement)
        binary_mask = (warped_mask > 128).astype(np.uint8) * 255
        binary_mask[0, :] = 0
        binary_mask[-1, :] = 0
        binary_mask[:, 0] = 0
        binary_mask[:, -1] = 0

        try:
            return cv2.seamlessClone(warped, frame, binary_mask, center, cv2.NORMAL_CLONE)
        except Exception as e:
            print(f"[FaceSwapper] seamlessClone error ({e}) – falling back to alpha blend")
            alpha = warped_mask.astype(np.float32) / 255.0
            alpha3 = np.stack([alpha] * 3, axis=-1)
            blended = warped.astype(np.float32) * alpha3 + frame.astype(np.float32) * (1.0 - alpha3)
            return blended.astype(np.uint8)

    # ── Reference from Video ───────────────────────────────────────

    def extract_reference_face_from_video(self, video_path: str, frame_index: int = 0):
        """Extract a Face object from a video frame."""
        self.initialize()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {frame_index} from video: {video_path}")

        face = self.get_best_face(frame, lenient=True)
        if face is None:
            raise ValueError(f"No face detected in video frame {frame_index}: {video_path}")
        return face

    # ── Face Swap / Overlay ────────────────────────────────────────

    def swap_face(self, frame: np.ndarray, source_face, target_face) -> np.ndarray:
        """Swap a detected face in the frame using inswapper."""
        self.initialize()
        return self.swap_model.get(frame, source_face, target_face, paste_back=True)

    def process_frame(self, frame: np.ndarray, target_face) -> np.ndarray:
        """
        Process a single video frame.
        - If target_face is an InsightFace Face → inswapper (high-quality face swap)
        - If target_face is a dict (cartoon)   → affine warp + seamless blend overlay
        """
        if isinstance(target_face, dict) and target_face.get("is_cartoon"):
            return self._overlay_cartoon_on_frame(frame, target_face)

        source_face = self.get_best_face(frame)
        if source_face is None:
            return frame
        return self.swap_face(frame, source_face, target_face)
