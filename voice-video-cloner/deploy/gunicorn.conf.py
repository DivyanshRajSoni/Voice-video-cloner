"""
Gunicorn Configuration for Voice & Video Cloner
Used in production on EC2 behind Nginx.
"""

import os
import multiprocessing

# ─── Server Socket ───────────────────────────────────────
bind = os.environ.get("BIND", "127.0.0.1:5000")

# ─── Workers ─────────────────────────────────────────────
# Keep low! Each worker loads AI models (~2-4 GB RAM each).
# For GPU instances: 2 workers is optimal
# For CPU instances: 1 worker (required for in-memory job tracking)
workers = int(os.environ.get("WORKERS", 1))
worker_class = "gthread"
threads = 4

# ─── Timeouts ────────────────────────────────────────────
# Video processing can take several minutes
timeout = 600          # 10 minutes — cloning jobs are long-running
graceful_timeout = 120
keepalive = 5

# ─── Logging ─────────────────────────────────────────────
accesslog = "-"
errorlog = "-"
loglevel = "info"

# ─── Process Naming ──────────────────────────────────────
proc_name = "voice-video-cloner"

# ─── Security ────────────────────────────────────────────
limit_request_line = 8190
limit_request_field_size = 8190

# ─── Server Hooks ────────────────────────────────────────
def on_starting(server):
    """Called just before the master process is initialized."""
    print("Voice & Video Cloner starting...")

def when_ready(server):
    """Called just after the server is started."""
    print(f"Voice & Video Cloner ready at {bind}")

def worker_abort(worker):
    """Called when a worker times out."""
    print(f"Worker {worker.pid} aborted (timeout)")
