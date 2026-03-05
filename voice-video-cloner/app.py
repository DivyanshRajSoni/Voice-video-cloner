"""
Voice & Video Cloner — Production Flask Application

Features:
- File-based job tracking (multi-worker safe)
- Automatic cleanup of old jobs and uploads
- Input validation and security
- Structured logging
- Health check with system metrics
- GPU-accelerated AI pipeline
"""

import os
import json
import uuid
import time
import logging
import threading
import cv2
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify,
    send_file, url_for
)
from werkzeug.utils import secure_filename

# ─── Logging Setup ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("App")

# Lazy import - VideoProcessor needs heavy ML dependencies
VideoProcessor = None

def _get_video_processor():
    global VideoProcessor
    if VideoProcessor is None:
        from core.video_processor import VideoProcessor as VP
        VideoProcessor = VP
    return VideoProcessor

# ─── Configuration ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
ALL_ALLOWED = ALLOWED_VIDEO_EXT | ALLOWED_IMAGE_EXT | ALLOWED_AUDIO_EXT

MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB for production
MAX_VIDEO_SIZE = 300 * 1024 * 1024       # 300MB per video
MAX_AUDIO_SIZE = 100 * 1024 * 1024       # 100MB per audio
MAX_IMAGE_SIZE = 20 * 1024 * 1024        # 20MB per image
JOB_RETENTION_HOURS = 24                 # Auto-cleanup after 24h

# ─── App Setup ───────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config["JSON_SORT_KEYS"] = False

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── File-based Job Tracking (works across multiple gunicorn workers) ────
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)


def _job_path(job_id):
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def _read_job(job_id):
    path = _job_path(job_id)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_job(job_id, data):
    path = _job_path(job_id)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _update_job(job_id, **kwargs):
    job = _read_job(job_id)
    if job:
        job.update(kwargs)
        _write_job(job_id, job)


def allowed_file(filename):
    """Check if file extension is allowed."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALL_ALLOWED


def get_file_ext(filename):
    """Get file extension."""
    return os.path.splitext(filename)[1].lower()


def save_upload(file, prefix="file"):
    """Save uploaded file and return the path."""
    ext = get_file_ext(file.filename)
    safe_name = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(path)
    return path


# ─── Routes ──────────────────────────────────────────────────────

CARTOON_FACES_DIR = os.path.join(BASE_DIR, "static", "cartoon_faces")
HUMAN_FACES_DIR = os.path.join(BASE_DIR, "static", "human_faces")
os.makedirs(CARTOON_FACES_DIR, exist_ok=True)
os.makedirs(HUMAN_FACES_DIR, exist_ok=True)


@app.route("/")
def index():
    """Serve the main UI."""
    return render_template("index.html")


# ── Cartoon character name mapping ──
CARTOON_NAMES = {
    "character_1": "Animated Boy",
    "character_2": "Animated Girl",
    "character_3": "Toon Hero",
    "character_4": "Toon Heroine",
    "character_5": "Toon Character",
}

# ── Human face name mapping ──
HUMAN_NAMES = {
    "india_male": "Arjun (India)",
    "usa_female": "Emily (USA)",
    "uk_male": "James (UK)",
    "russia_female": "Anastasia (Russia)",
    "australia_male": "Liam (Australia)",
}


def _list_faces(directory, subfolder, name_map):
    """List face images in a directory with mapped display names."""
    faces = []
    if os.path.exists(directory):
        for fname in sorted(os.listdir(directory)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in ALLOWED_IMAGE_EXT:
                key = os.path.splitext(fname)[0]
                name = name_map.get(key, key.replace("_", " ").replace("-", " ").title())
                faces.append({
                    "filename": fname,
                    "name": name,
                    "url": url_for("static", filename=f"{subfolder}/{fname}")
                })
    return faces


@app.route("/api/cartoon-faces")
def list_cartoon_faces():
    """List available cartoon character faces."""
    return jsonify({"faces": _list_faces(CARTOON_FACES_DIR, "cartoon_faces", CARTOON_NAMES)})


@app.route("/api/human-faces")
def list_human_faces():
    """List available AI-generated human faces."""
    return jsonify({"faces": _list_faces(HUMAN_FACES_DIR, "human_faces", HUMAN_NAMES)})


@app.route("/api/clone", methods=["POST"])
def start_clone():
    """
    Start a cloning job.
    
    Expects multipart form data:
    - source_video: The 30-second video of yourself
    - target_face: Image or video of the target persona (for face)
    - target_voice: Audio or video of the target persona (for voice)
    - language: Language code (default: "en")
    """
    # Validate uploads
    if "source_video" not in request.files:
        return jsonify({"error": "Source video is required."}), 400

    # Check if cartoon face is selected OR target_face is uploaded
    cartoon_face = request.form.get("cartoon_face", "")
    has_target_face = "target_face" in request.files and request.files["target_face"].filename

    if not has_target_face and not cartoon_face:
        return jsonify({"error": "Target face image/video or cartoon character is required."}), 400

    if "target_voice" not in request.files:
        return jsonify({"error": "Target voice audio/video is required."}), 400

    source_video = request.files["source_video"]
    target_voice = request.files["target_voice"]
    language = request.form.get("language", "en")
    voice_name = request.form.get("voice_name", "")
    bg_prompt = request.form.get("bg_prompt", "").strip()

    # Validate file types
    if not source_video.filename or get_file_ext(source_video.filename) not in ALLOWED_VIDEO_EXT:
        return jsonify({"error": "Source must be a video file (.mp4, .avi, .mov, .mkv, .webm)."}), 400

    if has_target_face:
        target_face = request.files["target_face"]
        if get_file_ext(target_face.filename) not in (ALLOWED_VIDEO_EXT | ALLOWED_IMAGE_EXT):
            return jsonify({"error": "Target face must be an image or video file."}), 400

    if not target_voice.filename or get_file_ext(target_voice.filename) not in (ALLOWED_VIDEO_EXT | ALLOWED_AUDIO_EXT):
        return jsonify({"error": "Target voice must be an audio or video file."}), 400

    # Save uploads
    source_path = save_upload(source_video, "source")

    # Determine face path: preset face (cartoon/human) or uploaded file
    if cartoon_face:
        safe = secure_filename(cartoon_face)
        # Check both cartoon and human face directories
        face_path = os.path.join(CARTOON_FACES_DIR, safe)
        if not os.path.exists(face_path):
            face_path = os.path.join(HUMAN_FACES_DIR, safe)
        if not os.path.exists(face_path):
            return jsonify({"error": "Selected face not found."}), 400
    else:
        target_face = request.files["target_face"]
        face_path = save_upload(target_face, "face")

    voice_path = save_upload(target_voice, "voice")

    # Handle background image upload (optional)
    bg_image_path = None
    if "bg_image" in request.files and request.files["bg_image"].filename:
        bg_file = request.files["bg_image"]
        bg_ext = get_file_ext(bg_file.filename)
        if bg_ext in ALLOWED_IMAGE_EXT:
            bg_image_path = save_upload(bg_file, "bg")

    # Create job
    job_id = str(uuid.uuid4())[:8]
    _write_job(job_id, {
        "status": "queued",
        "progress": 0,
        "stage": "",
        "message": "Job queued...",
        "result": None,
        "created_at": datetime.now().isoformat(),
    })

    # Run processing in background thread
    thread = threading.Thread(
        target=_run_cloning_job,
        args=(job_id, source_path, face_path, voice_path, language, voice_name,
              bg_image_path, bg_prompt),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202


def _run_cloning_job(job_id, source_path, face_path, voice_path, language,
                     voice_name="", bg_image_path=None, bg_prompt=None):
    """Run the cloning pipeline in a background thread."""
    def progress_callback(stage, percent, message):
        _update_job(job_id, stage=stage, progress=percent, message=message, status="processing")

    try:
        VP = _get_video_processor()
        processor = VP(models_dir=MODELS_DIR, output_dir=OUTPUT_DIR)
        result = processor.process(
            source_video_path=source_path,
            target_face_path=face_path,
            target_voice_path=voice_path,
            language=language,
            voice_name=voice_name,
            bg_image_path=bg_image_path if bg_image_path else None,
            bg_prompt=bg_prompt if bg_prompt else None,
            progress_callback=progress_callback
        )

        if result["status"] == "complete":
            _update_job(job_id, result=result, status="complete", progress=100,
                        message="Cloning complete! Download your video.")
        else:
            _update_job(job_id, result=result, status="error",
                        message=result.get("error", "Unknown error"))

    except Exception as e:
        _update_job(job_id, status="error", message=str(e))
        import traceback
        traceback.print_exc()


@app.route("/api/status/<job_id>")
def job_status(job_id):
    """Get the status of a cloning job."""
    job = _read_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "stage": job.get("stage", ""),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "stats": job.get("result", {}).get("stats", {}) if job.get("result") else {},
    })


@app.route("/api/download/<job_id>")
def download_result(job_id):
    """Download the output video for a completed job."""
    job = _read_job(job_id)

    if not job:
        return jsonify({"error": "Job not found."}), 404

    if job["status"] != "complete":
        return jsonify({"error": "Job not yet complete."}), 400

    output_path = job["result"]["output_video"]
    if not output_path or not os.path.exists(output_path):
        return jsonify({"error": "Output file not found."}), 404

    return send_file(
        output_path,
        as_attachment=True,
        download_name=f"cloned_{job_id}.mp4",
        mimetype="video/mp4"
    )


@app.route("/api/health")
def health():
    """Health check endpoint with system metrics."""
    gpu_info = _get_gpu_info()
    import shutil
    disk = shutil.disk_usage(BASE_DIR)
    return jsonify({
        "status": "ok",
        "version": "2.1.0",
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_info,
        "disk": {
            "total_gb": round(disk.total / (1024**3), 1),
            "free_gb": round(disk.free / (1024**3), 1),
            "used_pct": round((disk.used / disk.total) * 100, 1),
        },
        "active_jobs": _count_active_jobs(),
    })


@app.route("/api/generate-bg", methods=["POST"])
def generate_bg_preview():
    """
    Generate a background image from a text prompt (for preview).
    Returns the generated image as JPEG.
    """
    data = request.get_json()
    prompt = data.get("prompt", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    try:
        from core.background_changer import BackgroundChanger
        changer = BackgroundChanger()
        bg = changer.generate_background(prompt=prompt, width=1280, height=720)

        # Encode to JPEG and return
        import io
        _, buf = cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return app.response_class(
            response=buf.tobytes(),
            status=200,
            mimetype="image/jpeg",
        )
    except Exception as e:
        logger.error(f"Background generation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/voices")
def list_voices():
    """List available voice personas."""
    from core.voice_cloner import VOICE_PERSONAS, DEFAULT_VOICES
    return jsonify({
        "voices": VOICE_PERSONAS,
        "defaults": DEFAULT_VOICES,
    })


def _get_gpu_info():
    """Get GPU information."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
                "memory_free_gb": round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3), 1),
            }
    except ImportError:
        pass
    return {"available": False}


def _count_active_jobs():
    """Count currently processing jobs."""
    count = 0
    try:
        for fname in os.listdir(JOBS_DIR):
            if fname.endswith('.json'):
                job = _read_job(fname.replace('.json', ''))
                if job and job.get('status') in ('queued', 'processing'):
                    count += 1
    except Exception:
        pass
    return count


# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Voice & Video Cloner")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
