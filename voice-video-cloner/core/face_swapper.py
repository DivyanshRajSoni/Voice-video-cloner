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
        self.rec_model = None
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

        # ── Load recognition model (~270MB, needed for face embeddings) ──
        rec_path = os.path.join(self.model_dir, "models", "buffalo_l", "w600k_r50.onnx")
        if os.path.exists(rec_path):
            print(f"[FaceSwapper] Loading recognition model ({os.path.getsize(rec_path) // 1024 // 1024}MB)...")
            self.rec_model = insightface.model_zoo.get_model(rec_path, providers=providers, session_options=sess_opts)
            self.rec_model.prepare(ctx_id=0)
            print("[FaceSwapper] Recognition model loaded.")
        else:
            print(f"[FaceSwapper] WARNING: Recognition model not found at {rec_path}")

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
            # Compute face embedding (required by inswapper)
            if self.rec_model is not None:
                self.rec_model.get(frame, face)
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

    def _detect_with_strategies(self, img: np.ndarray):
        """Try multiple detection strategies for difficult images (cartoons etc.)."""
        _largest = lambda faces: max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        h, w = img.shape[:2]

        # Strategy 1: try with larger input size (640x640) and progressively lower thresholds
        self.det_model.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)
        for thresh in [0.5, 0.3, 0.1, 0.05, 0.02]:
            faces = self.detect_faces(img, det_thresh=thresh)
            if faces:
                print(f"[FaceSwapper] Detected face at 640px with threshold={thresh}")
                return _largest(faces)

        # Strategy 2: upscale to 1024px if small
        if max(h, w) < 1024:
            scale = 1024 / max(h, w)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            for thresh in [0.3, 0.1, 0.05, 0.02]:
                faces = self.detect_faces(resized, det_thresh=thresh)
                if faces:
                    print(f"[FaceSwapper] Detected face after upscale-1024 with threshold={thresh}")
                    return _largest(faces)

        # Strategy 3: increase contrast + sharpen (helps with flat cartoon colors)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        for thresh in [0.3, 0.1, 0.05, 0.02]:
            faces = self.detect_faces(enhanced, det_thresh=thresh)
            if faces:
                print(f"[FaceSwapper] Detected face after enhancement with threshold={thresh}")
                return _largest(faces)

        # Strategy 4: upscale the enhanced image  
        if max(h, w) < 1024:
            scale = 1024 / max(h, w)
            resized = cv2.resize(enhanced, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            resized = enhanced
        for thresh in [0.1, 0.05, 0.02]:
            faces = self.detect_faces(resized, det_thresh=thresh)
            if faces:
                print(f"[FaceSwapper] Detected face after upscale+enhance with threshold={thresh}")
                return _largest(faces)

        # Strategy 5: try center crop (face might be in center with lots of background)
        crop_h, crop_w = int(h * 0.7), int(w * 0.7)
        y1, x1 = (h - crop_h) // 2, (w - crop_w) // 2
        cropped = img[y1:y1+crop_h, x1:x1+crop_w]
        scale = 1024 / max(crop_h, crop_w)
        cropped = cv2.resize(cropped, (int(crop_w * scale), int(crop_h * scale)), interpolation=cv2.INTER_CUBIC)
        for thresh in [0.3, 0.1, 0.05, 0.02]:
            faces = self.detect_faces(cropped, det_thresh=thresh)
            if faces:
                print(f"[FaceSwapper] Detected face after center-crop with threshold={thresh}")
                return _largest(faces)

        # Reset input size to 320 for video frame processing (speed)
        self.det_model.prepare(ctx_id=0, input_size=(320, 320), det_thresh=0.5)
        return None

    def extract_reference_face(self, image_path: str):
        """Extract reference face from an image file. Uses aggressive detection for cartoons."""
        self.initialize()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Use aggressive multi-strategy detection for reference images
        face = self._detect_with_strategies(img)
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
