#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Voice & Video Cloner — GCP VM Create Helper
# ═══════════════════════════════════════════════════════════
#
# Run this script FROM YOUR LOCAL MACHINE (not on the VM).
# It creates the VM, uploads the project, and kicks off setup.
#
# Prerequisites:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Authenticate:  gcloud auth login
#   3. Set project:   gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   bash deploy/gcp-create-vm.sh [gpu|cpu]
#
# ═══════════════════════════════════════════════════════════

set -e

MODE="${1:-cpu}"   # default to cpu, pass "gpu" for GPU instance
VM_NAME="voice-cloner"
ZONE="us-central1-a"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "═══════════════════════════════════════════════════════"
echo "  Creating GCP VM: ${VM_NAME} (${MODE} mode)"
echo "═══════════════════════════════════════════════════════"

# ── Check gcloud ──
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not found."
    echo "Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# ── Verify project is set ──
GCP_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$GCP_PROJECT" ] || [ "$GCP_PROJECT" = "(unset)" ]; then
    echo "ERROR: No GCP project set."
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi
echo "GCP Project: ${GCP_PROJECT}"
echo "Zone:        ${ZONE}"
echo ""

# ── Create firewall rules ──
echo "[1/4] Setting up firewall rules..."
gcloud compute firewall-rules create allow-http \
    --direction=INGRESS --action=ALLOW \
    --rules=tcp:80 --source-ranges=0.0.0.0/0 \
    --target-tags=http-server \
    --description="Allow HTTP for Voice Cloner" \
    --quiet 2>/dev/null || echo "  (firewall rule already exists)"

gcloud compute firewall-rules create allow-https \
    --direction=INGRESS --action=ALLOW \
    --rules=tcp:443 --source-ranges=0.0.0.0/0 \
    --target-tags=https-server \
    --description="Allow HTTPS for Voice Cloner" \
    --quiet 2>/dev/null || echo "  (firewall rule already exists)"

echo "[OK] Firewall rules ready."

# ── Create VM ──
echo ""
echo "[2/4] Creating Compute Engine VM..."

if [ "$MODE" = "gpu" ]; then
    echo "  Type: n1-standard-4 + NVIDIA T4 GPU"
    gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type=n1-standard-4 \
        --accelerator=type=nvidia-tesla-t4,count=1 \
        --maintenance-policy=TERMINATE \
        --image-family=common-cu121-ubuntu-2204 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=60GB \
        --boot-disk-type=pd-ssd \
        --tags=http-server,https-server \
        --metadata=install-nvidia-driver=True
else
    echo "  Type: e2-standard-4 (CPU only)"
    gcloud compute instances create "$VM_NAME" \
        --zone="$ZONE" \
        --machine-type=e2-standard-4 \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=60GB \
        --boot-disk-type=pd-ssd \
        --tags=http-server,https-server
fi

echo "[OK] VM created."

# Wait for VM to be ready
echo "  Waiting 30s for VM to initialize..."
sleep 30

# ── Upload project ──
echo ""
echo "[3/4] Uploading project to VM..."
gcloud compute scp --recurse "$PROJECT_DIR" "${VM_NAME}:~/" --zone="$ZONE"
echo "[OK] Project uploaded."

# ── Run setup on VM ──
echo ""
echo "[4/4] Running deployment script on VM..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command \
    "cd ~/voice-video-cloner && chmod +x deploy/gcp-setup.sh && bash deploy/gcp-setup.sh"

# ── Get external IP ──
EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$ZONE" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ALL DONE!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  App URL:    http://${EXTERNAL_IP}"
echo "  SSH:        gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
echo "  Stop VM:    gcloud compute instances stop ${VM_NAME} --zone=${ZONE}"
echo "  Start VM:   gcloud compute instances start ${VM_NAME} --zone=${ZONE}"
echo "  Delete VM:  gcloud compute instances delete ${VM_NAME} --zone=${ZONE}"
echo ""
echo "  Cost-saving tip: Stop the VM when not in use!"
echo "═══════════════════════════════════════════════════════"
