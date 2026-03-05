"""
Face Swapper Module
Uses InsightFace for face detection and swapping.
Replaces the face in each video frame with the target persona face.
Ultra-lightweight: loads only det_10g + inswapper_128 (skips landmark, age, gender, recognition)
"""

import os
import cv2
import numpy as np
import insightface
import onnxruntime as ort


class FaceSwapper:
    """Handles face detection and swapping using InsightFace models."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.det_model = None
        self.swap_model = None
        self._initialized = False

    def initialize(self):
        """Load only detection + swapper models directly (bypass FaceAnalysis)."""
        if self._initialized:
            return

        import gc
        gc.collect()

        providers = self._get_providers()

        # ── Session options for minimal memory ──
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.enable_mem_pattern = True
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        # ── Load ONLY detection model directly (~16MB) ──
        det_path = os.path.join(self.model_dir, "models", "buffalo_l", "det_10g.onnx")
        if not os.path.exists(det_path):
            raise FileNotFoundError(f"Detection model not found: {det_path}")

        print(f"[FaceSwapper] Loading detection model ({os.path.getsize(det_path) // 1024 // 1024}MB)...")
        self.det_model = insightface.model_zoo.get_model(det_path, providers=providers, session_options=sess_opts)
        self.det_model.prepare(ctx_id=0, input_size=(320, 320), det_thresh=0.5)
        print("[FaceSwapper] Detection model loaded.")

        gc.collect()

        # ── Load face swapper model (~530MB) ──
        swap_path = os.path.join(self.model_dir, "inswapper_128.onnx")
        if not os.path.exists(swap_path):
            raise FileNotFoundError(
                f"Face swap model not found at {swap_path}. "
                f"Download 'inswapper_128.onnx' from "
                f"https://github.com/facefusion/facefusion-assets/releases"
            )

        print(f"[FaceSwapper] Loading swapper model ({os.path.getsize(swap_path) // 1024 // 1024}MB)...")
        self.swap_model = insightface.model_zoo.get_model(swap_path, providers=providers, session_options=sess_opts)
        print("[FaceSwapper] Swapper model loaded.")

        self._initialized = True
        gc.collect()
        print("[FaceSwapper] Initialization complete (2 models only).")

    def _get_providers(self):
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def detect_faces(self, frame: np.ndarray, det_thresh=None):
        """Detect faces using det_10g model directly."""
        self.initialize()
        if det_thresh is not None:
            self.det_model.det_thresh = det_thresh
        bboxes, kpss = self.det_model.detect(frame, max_num=0, metric='default')
        # Build face objects compatible with inswapper
        faces = []
        for i in range(bboxes.shape[0]):
            face = insightface.app.common.Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4])
            faces.append(face)
        return faces

    def get_best_face(self, frame: np.ndarray, lenient=False):
        """Get the largest face in a frame."""
        faces = self.detect_faces(frame)
        if faces:
            return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        if lenient:
            # Try lower threshold
            faces = self.detect_faces(frame, det_thresh=0.1)
            if faces:
                return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            # Try upscaling small images
            h, w = frame.shape[:2]
            if max(h, w) < 400:
                scale = 640 / max(h, w)
                resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
                faces = self.detect_faces(resized, det_thresh=0.1)
                if faces:
                    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        return None

    def extract_reference_face(self, image_path: str):
        """Extract reference face from an image file."""
        self.initialize()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        face = self.get_best_face(img, lenient=True)
        if face is None:
            raise ValueError(
                f"No face detected in reference image: {image_path}. "
                f"For cartoon characters, use images with clear, front-facing, "
                f"human-like facial features (two eyes, nose, mouth visible)."
            )
        return face

    def extract_reference_face_from_video(self, video_path: str, frame_index: int = 0):
        """Extract reference face from a video file."""
        self.initialize()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {frame_index} from video: {video_path}")

        face = self.get_best_face(frame)
        if face is None:
            raise ValueError(f"No face detected in video frame {frame_index}: {video_path}")
        return face

    def swap_face(self, frame: np.ndarray, source_face, target_face) -> np.ndarray:
        """Swap a face in the frame."""
        self.initialize()
        return self.swap_model.get(frame, source_face, target_face, paste_back=True)

    def process_frame(self, frame: np.ndarray, target_face) -> np.ndarray:
        """Process a single frame: detect face and swap with target."""
        source_face = self.get_best_face(frame)
        if source_face is None:
            return frame
        return self.swap_face(frame, source_face, target_face)
