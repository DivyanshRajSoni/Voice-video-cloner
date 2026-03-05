#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Voice & Video Cloner — Google Cloud (Compute Engine) Setup
# ═══════════════════════════════════════════════════════════
#
# Recommended VM Configuration:
#   ┌────────────────────┬──────────────────────────────────────────┐
#   │ GPU (Recommended)  │ n1-standard-4 + 1x NVIDIA T4            │
#   │                    │ 4 vCPU, 15 GB RAM, ~$0.45/hr            │
#   ├────────────────────┼──────────────────────────────────────────┤
#   │ CPU-only (Budget)  │ e2-standard-4                            │
#   │                    │ 4 vCPU, 16 GB RAM, ~$0.13/hr            │
#   ├────────────────────┼──────────────────────────────────────────┤
#   │ Boot Disk          │ Ubuntu 22.04 LTS, 60 GB SSD (pd-ssd)   │
#   ├────────────────────┼──────────────────────────────────────────┤
#   │ GPU Image          │ "Deep Learning VM with CUDA 12.1"       │
#   │                    │ (from GCP Marketplace — CUDA pre-done)  │
#   └────────────────────┴──────────────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════
# QUICK START (3 steps):
#
# 1. Create VM via gcloud CLI or Console (see below)
# 2. Upload project:
#      gcloud compute scp --recurse voice-video-cloner/ VM_NAME:~/  --zone=ZONE
# 3. SSH in & run:
#      gcloud compute ssh VM_NAME --zone=ZONE
#      cd ~/voice-video-cloner && chmod +x deploy/gcp-setup.sh && bash deploy/gcp-setup.sh
#
# ═══════════════════════════════════════════════════════════
# CREATE VM VIA gcloud CLI (copy-paste):
#
# --- GPU Instance (T4) ---
#   gcloud compute instances create voice-cloner-gpu \
#     --zone=us-central1-a \
#     --machine-type=n1-standard-4 \
#     --accelerator=type=nvidia-tesla-t4,count=1 \
#     --maintenance-policy=TERMINATE \
#     --image-family=common-cu121-ubuntu-2204 \
#     --image-project=deeplearning-platform-release \
#     --boot-disk-size=60GB \
#     --boot-disk-type=pd-ssd \
#     --tags=http-server,https-server \
#     --metadata=install-nvidia-driver=True
#
# --- CPU-only Instance ---
#   gcloud compute instances create voice-cloner-cpu \
#     --zone=us-central1-a \
#     --machine-type=e2-standard-4 \
#     --image-family=ubuntu-2204-lts \
#     --image-project=ubuntu-os-cloud \
#     --boot-disk-size=60GB \
#     --boot-disk-type=pd-ssd \
#     --tags=http-server,https-server
#
# ═══════════════════════════════════════════════════════════

set -e  # Exit on any error

echo "═══════════════════════════════════════════════════════"
echo "  Voice & Video Cloner — Google Cloud Deployment"
echo "═══════════════════════════════════════════════════════"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_USER="$(whoami)"

# ──────────────────────────────────────────────────────────
# 1. System Updates & Essential Packages
# ──────────────────────────────────────────────────────────
echo "[1/8] Installing system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3.10 python3.10-venv python3-pip \
    ffmpeg \
    nginx \
    git curl wget unzip \
    build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libsndfile1

echo "[OK] System packages installed."

# ──────────────────────────────────────────────────────────
# 2. Check / Install NVIDIA GPU Drivers
# ──────────────────────────────────────────────────────────
echo ""
echo "[2/8] Checking GPU..."
HAS_GPU=false

if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HAS_GPU=true
elif lspci | grep -i nvidia &> /dev/null; then
    echo "[INFO] NVIDIA hardware found but drivers not installed."
    echo "  If you used Deep Learning VM image, drivers install on first boot."
    echo "  Otherwise install manually:"
    echo "    sudo apt install -y nvidia-driver-535"
    echo "    sudo reboot"
    echo ""
    echo "  Continuing with CPU mode for now..."
else
    echo "[INFO] No NVIDIA GPU detected. Using CPU-only mode."
    echo "  For GPU: create VM with --accelerator=type=nvidia-tesla-t4,count=1"
fi

# ──────────────────────────────────────────────────────────
# 3. Python Virtual Environment & Dependencies
# ──────────────────────────────────────────────────────────
echo ""
echo "[3/8] Setting up Python virtual environment..."
cd "$PROJECT_DIR"

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

# Install PyTorch — GPU or CPU variant
if [ "$HAS_GPU" = true ]; then
    echo "  Installing PyTorch with CUDA support..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "  Installing PyTorch (CPU only)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install project requirements
pip install -r requirements.txt

# Production WSGI server
pip install gunicorn

# GPU-specific optimizations
if [ "$HAS_GPU" = true ]; then
    pip install onnxruntime-gpu
fi

echo "[OK] Python environment ready."

# ──────────────────────────────────────────────────────────
# 4. Create .env file
# ──────────────────────────────────────────────────────────
echo ""
echo "[4/8] Creating environment config..."

if [ ! -f "$PROJECT_DIR/.env" ]; then
    cat > "$PROJECT_DIR/.env" << 'ENVEOF'
# Voice & Video Cloner — Production Environment
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=CHANGE_ME_TO_A_RANDOM_STRING
MAX_CONTENT_LENGTH=209715200
WORKERS=2
BIND=127.0.0.1:5000
ENVEOF
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i "s/CHANGE_ME_TO_A_RANDOM_STRING/$SECRET/" "$PROJECT_DIR/.env"
    echo "[OK] .env created with random secret key."
else
    echo "[OK] .env already exists, skipping."
fi

# ──────────────────────────────────────────────────────────
# 5. Gunicorn systemd Service
# ──────────────────────────────────────────────────────────
echo ""
echo "[5/8] Setting up Gunicorn service..."

sudo tee /etc/systemd/system/voice-cloner.service > /dev/null << SERVICEEOF
[Unit]
Description=Voice & Video Cloner (Gunicorn)
After=network.target

[Service]
User=${APP_USER}
Group=www-data
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=${PROJECT_DIR}/.env
ExecStart=${PROJECT_DIR}/venv/bin/gunicorn \
    --config ${PROJECT_DIR}/deploy/gunicorn.conf.py \
    app:app
Restart=always
RestartSec=5
StandardOutput=append:/var/log/voice-cloner/app.log
StandardError=append:/var/log/voice-cloner/error.log

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo mkdir -p /var/log/voice-cloner
sudo chown ${APP_USER}:www-data /var/log/voice-cloner

sudo systemctl daemon-reload
sudo systemctl enable voice-cloner

echo "[OK] Gunicorn service configured."

# ──────────────────────────────────────────────────────────
# 6. Nginx Reverse Proxy
# ──────────────────────────────────────────────────────────
echo ""
echo "[6/8] Configuring Nginx..."

# GCP metadata API for external IP
GCP_IP=$(curl -s -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip \
    2>/dev/null || echo "localhost")

sudo tee /etc/nginx/sites-available/voice-cloner > /dev/null << NGINXEOF
server {
    listen 80;
    server_name ${GCP_IP} _;

    client_max_body_size 200M;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Static files — served directly by Nginx
    location /static/ {
        alias ${PROJECT_DIR}/static/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # Proxy all requests to Gunicorn
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
    }

    # Download — longer timeout for large video files
    location /api/download/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
NGINXEOF

sudo ln -sf /etc/nginx/sites-available/voice-cloner /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

echo "[OK] Nginx configured."

# ──────────────────────────────────────────────────────────
# 7. GCP Firewall Rule (allow HTTP)
# ──────────────────────────────────────────────────────────
echo ""
echo "[7/8] Checking GCP firewall..."

if command -v gcloud &> /dev/null; then
    # Check if rule already exists
    if gcloud compute firewall-rules describe allow-http --quiet 2>/dev/null; then
        echo "[OK] Firewall rule 'allow-http' already exists."
    else
        echo "  Creating firewall rule to allow HTTP (port 80)..."
        gcloud compute firewall-rules create allow-http \
            --direction=INGRESS \
            --action=ALLOW \
            --rules=tcp:80 \
            --source-ranges=0.0.0.0/0 \
            --target-tags=http-server \
            --description="Allow HTTP traffic for Voice Cloner" \
            --quiet 2>/dev/null || true
        echo "[OK] Firewall rule created."
    fi
else
    echo "[INFO] gcloud CLI not found on this VM."
    echo "  Make sure firewall allows HTTP (port 80) from GCP Console:"
    echo "  VPC Network → Firewall → Create rule → tcp:80 → 0.0.0.0/0"
fi

# ──────────────────────────────────────────────────────────
# 8. Run Model Setup & Start
# ──────────────────────────────────────────────────────────
echo ""
echo "[8/8] Running model setup..."
cd "$PROJECT_DIR"
source venv/bin/activate
python setup.py

# Start the service
sudo systemctl start voice-cloner

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  GCP DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  App URL:        http://${GCP_IP}"
echo "  Service:        sudo systemctl status voice-cloner"
echo "  App Logs:       sudo tail -f /var/log/voice-cloner/app.log"
echo "  Error Logs:     sudo tail -f /var/log/voice-cloner/error.log"
echo "  Nginx Logs:     sudo tail -f /var/log/nginx/access.log"
echo ""
echo "  Commands:"
echo "    Restart:      sudo systemctl restart voice-cloner"
echo "    Stop:         sudo systemctl stop voice-cloner"
echo "    Status:       sudo systemctl status voice-cloner"
echo ""
echo "  ──── IMPORTANT ────────────────────────────────────"
echo "  Make sure your VM has the 'http-server' network tag"
echo "  and a firewall rule allowing tcp:80 from 0.0.0.0/0."
echo ""
echo "  Add HTTPS later:"
echo "    sudo apt install certbot python3-certbot-nginx"
echo "    sudo certbot --nginx -d yourdomain.com"
echo "═══════════════════════════════════════════════════════"
