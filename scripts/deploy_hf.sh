#!/usr/bin/env bash
set -e

# ===========================================================================
# Deploy Backend to Hugging Face Spaces (Bash)
# ===========================================================================

echo "======================================================="
echo "  NeuroScan - Hugging Face Spaces Deployment Helper"
echo "======================================================="

read -p "Enter your Hugging Face username: " HF_USERNAME
read -p "Enter your Space name (e.g. neuroscan-backend): " SPACE_NAME

if [ -z "$HF_USERNAME" ] || [ -z "$SPACE_NAME" ]; then
    echo "Error: Username and Space name are required."
    exit 1
fi

TARGET_DIR="deploy_hf_space"
REMOTE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

echo "[*] Cloning space repo into ./${TARGET_DIR}..."
rm -rf "$TARGET_DIR"
git clone "$REMOTE_URL" "$TARGET_DIR"

cd "$TARGET_DIR"
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.bin"
cd ..

echo "[*] Copying backend files..."
cp Dockerfile "$TARGET_DIR/"
cp requirements.txt "$TARGET_DIR/"
cp .dockerignore "$TARGET_DIR/"
cp .gitattributes "$TARGET_DIR/"
cp app.py "$TARGET_DIR/"
cp main.py "$TARGET_DIR/"

cat << 'EOF' > "$TARGET_DIR/README.md"
---
title: NeuroScan Brain Tumor Segmentation API
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---
EOF

cp -r src "$TARGET_DIR/"
cp -r configs "$TARGET_DIR/"
mkdir -p "$TARGET_DIR/checkpoints"
cp checkpoints/best_attention_unet.pt "$TARGET_DIR/checkpoints/"
[ -f checkpoints/best_attention_95.pt ] && cp checkpoints/best_attention_95.pt "$TARGET_DIR/checkpoints/"

cd "$TARGET_DIR"
git add .
git commit -m "Deploy NeuroScan Brain Tumor Segmentation API"
git push origin main
cd ..
rm -rf "$TARGET_DIR"

echo "======================================================="
echo " Successfully deployed to Hugging Face!"
echo " API URL: https://${HF_USERNAME}-${SPACE_NAME}.hf.space"
echo "======================================================="
