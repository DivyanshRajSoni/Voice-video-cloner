"""
Setup script - Downloads required models and verifies dependencies.
Run this once before starting the application.
"""

import os
import sys
import subprocess


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def check_python_version():
    """Ensure Python 3.9+."""
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9 or higher is required.")
        sys.exit(1)
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        version_line = result.stdout.split('\n')[0] if result.stdout else "unknown"
        print(f"[OK] FFmpeg found: {version_line[:60]}")
    except FileNotFoundError:
        print("[WARNING] FFmpeg not found!")
        print("  Install FFmpeg:")
        print("  - Windows: choco install ffmpeg  OR  download from https://ffmpeg.org/download.html")
        print("  - macOS:   brew install ffmpeg")
        print("  - Linux:   sudo apt install ffmpeg")
        print()


def setup_models_dir():
    """Create models directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[OK] Models directory: {MODELS_DIR}")


def check_face_swap_model():
    """Check if the face swap model exists."""
    model_path = os.path.join(MODELS_DIR, "inswapper_128.onnx")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[OK] Face swap model found ({size_mb:.1f} MB)")
    else:
        print("[ACTION REQUIRED] Face swap model not found!")
        print(f"  Download 'inswapper_128.onnx' and place it in: {MODELS_DIR}")
        print(f"  Download from: https://github.com/facefusion/facefusion-assets/releases")
        print(f"  Direct link:   https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx")
        print()


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[OK] GPU available: {gpu_name}")
            print("  TIP: Install onnxruntime-gpu for faster face swapping:")
            print("       pip install onnxruntime-gpu")
        else:
            print("[INFO] No CUDA GPU detected. Processing will use CPU (slower).")
    except ImportError:
        print("[INFO] PyTorch not installed yet. GPU check skipped.")


def check_dependencies():
    """Check if key packages are importable."""
    packages = {
        "flask": "Flask",
        "cv2": "opencv-python",
        "numpy": "numpy",
        "moviepy": "moviepy",
        "insightface": "insightface",
        "onnxruntime": "onnxruntime",
        "TTS": "TTS (Coqui)",
        "whisper": "openai-whisper",
        "torch": "PyTorch",
        "torchaudio": "torchaudio",
    }

    missing = []
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            missing.append(name)

    return missing


def download_insightface_models():
    """Trigger InsightFace model download."""
    try:
        from insightface.app import FaceAnalysis
        print("[INFO] Downloading InsightFace buffalo_l model (first time only)...")
        app = FaceAnalysis(name="buffalo_l", root=MODELS_DIR)
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[OK] InsightFace models downloaded.")
    except Exception as e:
        print(f"[WARNING] InsightFace model download failed: {e}")
        print("  Models will be downloaded on first use.")


def main():
    """Run all setup checks."""
    print("=" * 60)
    print("  Voice & Video Cloner - Setup")
    print("=" * 60)
    print()

    # 1. Python version
    check_python_version()

    # 2. FFmpeg
    check_ffmpeg()

    # 3. Models directory
    setup_models_dir()

    # 4. Check dependencies
    print("\nChecking Python packages:")
    missing = check_dependencies()

    if missing:
        print(f"\n[WARNING] {len(missing)} packages are missing.")
        print("Install all dependencies with:")
        print("  pip install -r requirements.txt")
        print()
    else:
        print("\n[OK] All Python packages installed.")

    # 5. GPU
    print()
    check_gpu()

    # 6. Face swap model
    print()
    check_face_swap_model()

    # 7. Try downloading InsightFace models
    if "insightface" not in [m for m in [__import__.__module__ for _ in [None]]]:
        try:
            import insightface
            print()
            download_insightface_models()
        except ImportError:
            pass

    print()
    print("=" * 60)
    print("  Setup complete!")
    print("  Next steps:")
    print("  1. Install dependencies:   pip install -r requirements.txt")
    print("  2. Download face model:    See instructions above")
    print("  3. Run the app:            python app.py")
    print("  4. Open browser:           http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
