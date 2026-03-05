#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Voice & Video Cloner — Amazon EC2 Setup Script
# ═══════════════════════════════════════════════════════════
#
# Recommended EC2 Instance:
#   - GPU:  g4dn.xlarge  (T4 GPU, 4 vCPU, 16 GB RAM) ~$0.526/hr
#   - CPU:  t3.xlarge    (4 vCPU, 16 GB RAM)          ~$0.166/hr
#
# Recommended AMI:
#   - "Deep Learning AMI (Ubuntu 22.04)" — comes with CUDA pre-installed
#   - OR plain "Ubuntu 22.04 LTS" (this script installs everything)
#
# Usage:
#   1. Launch EC2 instance (Ubuntu 22.04, 50+ GB EBS storage)
#   2. SSH in:  ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
#   3. Upload project:  scp -i your-key.pem -r voice-video-cloner/ ubuntu@<EC2-PUBLIC-IP>:~/
#   4. Run:  cd ~/voice-video-cloner && bash deploy/ec2-setup.sh
#
# ═══════════════════════════════════════════════════════════

set -e  # Exit on any error

echo "═══════════════════════════════════════════════════════"
echo "  Voice & Video Cloner — EC2 Deployment"
echo "═══════════════════════════════════════════════════════"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_USER="ubuntu"

# ──────────────────────────────────────────────────────────
# 1. System Updates & Essential Packages
# ──────────────────────────────────────────────────────────
echo "[1/7] Installing system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3.10 python3.10-venv python3-pip \
    ffmpeg \
    nginx \
    git curl wget unzip \
    build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libsndfile1 \
    supervisor

echo "[OK] System packages installed."

# ──────────────────────────────────────────────────────────
# 2. Check for NVIDIA GPU (optional)
# ──────────────────────────────────────────────────────────
echo ""
echo "[2/7] Checking GPU..."
HAS_GPU=false

if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    HAS_GPU=true
else
    echo "[INFO] No NVIDIA GPU detected. Will use CPU-only mode."
    echo "  For GPU support, use a g4dn/g5/p3 instance with Deep Learning AMI."
fi

# ──────────────────────────────────────────────────────────
# 3. Python Virtual Environment
# ──────────────────────────────────────────────────────────
echo ""
echo "[3/7] Setting up Python virtual environment..."
cd "$PROJECT_DIR"

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

# Install PyTorch (GPU or CPU)
if [ "$HAS_GPU" = true ]; then
    echo "  Installing PyTorch with CUDA support..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "  Installing PyTorch (CPU only)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install project dependencies
pip install -r requirements.txt

# Install Gunicorn (production WSGI server)
pip install gunicorn

# Install onnxruntime-gpu if GPU available
if [ "$HAS_GPU" = true ]; then
    pip install onnxruntime-gpu
fi

echo "[OK] Python environment ready."

# ──────────────────────────────────────────────────────────
# 4. Create .env file
# ──────────────────────────────────────────────────────────
echo ""
echo "[4/7] Creating environment config..."

if [ ! -f "$PROJECT_DIR/.env" ]; then
    cat > "$PROJECT_DIR/.env" << 'ENVEOF'
# Voice & Video Cloner — Environment Configuration
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=CHANGE_ME_TO_A_RANDOM_STRING
MAX_CONTENT_LENGTH=209715200
WORKERS=2
BIND=127.0.0.1:5000
ENVEOF
    # Generate a random secret key
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i "s/CHANGE_ME_TO_A_RANDOM_STRING/$SECRET/" "$PROJECT_DIR/.env"
    echo "[OK] .env file created with random secret key."
else
    echo "[OK] .env file already exists, skipping."
fi

# ──────────────────────────────────────────────────────────
# 5. Setup Gunicorn systemd service
# ──────────────────────────────────────────────────────────
echo ""
echo "[5/7] Setting up Gunicorn service..."

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

# Create log directory
sudo mkdir -p /var/log/voice-cloner
sudo chown ${APP_USER}:www-data /var/log/voice-cloner

sudo systemctl daemon-reload
sudo systemctl enable voice-cloner

echo "[OK] Gunicorn service configured."

# ──────────────────────────────────────────────────────────
# 6. Setup Nginx reverse proxy
# ──────────────────────────────────────────────────────────
echo ""
echo "[6/7] Configuring Nginx..."

# Get the EC2 public IP (or use localhost)
EC2_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "localhost")

sudo tee /etc/nginx/sites-available/voice-cloner > /dev/null << NGINXEOF
server {
    listen 80;
    server_name ${EC2_IP} _;

    client_max_body_size 200M;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Static files (served directly by Nginx — faster)
    location /static/ {
        alias ${PROJECT_DIR}/static/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # Proxy to Gunicorn
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Long timeout for video processing status polling
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
    }

    # Download endpoint needs long timeout
    location /api/download/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
NGINXEOF

# Enable site, remove default
sudo ln -sf /etc/nginx/sites-available/voice-cloner /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test & reload
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

echo "[OK] Nginx configured. App will be available at http://${EC2_IP}"

# ──────────────────────────────────────────────────────────
# 7. Run model setup & start the app
# ──────────────────────────────────────────────────────────
echo ""
echo "[7/7] Running model setup..."
cd "$PROJECT_DIR"
source venv/bin/activate
python setup.py

# Start the service
sudo systemctl start voice-cloner

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  App URL:        http://${EC2_IP}"
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
echo "  IMPORTANT: Open port 80 (HTTP) in your EC2 Security Group!"
echo "  AWS Console → EC2 → Security Groups → Inbound Rules → Add:"
echo "    Type: HTTP | Port: 80 | Source: 0.0.0.0/0"
echo ""
echo "═══════════════════════════════════════════════════════"
