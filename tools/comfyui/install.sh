#!/bin/bash
set -e

COMFYUI_DIR="$HOME/ComfyUI"
MODELS_DIR="$COMFYUI_DIR/models"
VENV_DIR="$COMFYUI_DIR/venv"

echo "=== Installing ComfyUI ==="

# Clone
if [ ! -d "$COMFYUI_DIR" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi

cd "$COMFYUI_DIR"

# Python venv
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install torch with CUDA 12.8 (forward-compatible with CUDA 13.0 driver on DGX Spark)
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Installing PyTorch with CUDA 12.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, available={torch.cuda.is_available()}')"

# ComfyUI requirements
pip install --upgrade pip
pip install -r requirements.txt

# ComfyUI-Manager (for easy custom node install)
if [ ! -d "$COMFYUI_DIR/custom_nodes/ComfyUI-Manager" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git \
        "$COMFYUI_DIR/custom_nodes/ComfyUI-Manager"
fi

# Custom nodes: Pixelate + IPAdapter + Image Resize
for repo in \
    "https://github.com/filipemeneses/comfy_pixelization" \
    "https://github.com/cubiq/ComfyUI_IPAdapter_plus" \
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes" \
    "https://github.com/rgthree/rgthree-comfy"; do
    name=$(basename "$repo" .git)
    if [ ! -d "$COMFYUI_DIR/custom_nodes/$name" ]; then
        git clone "$repo" "$COMFYUI_DIR/custom_nodes/$name"
    fi
done

# Initialize submodules in custom nodes (comfy_pixelization needs this)
for d in "$COMFYUI_DIR/custom_nodes"/*/; do
    if [ -f "$d/.gitmodules" ]; then
        (cd "$d" && git submodule update --init --recursive)
    fi
done

# Install custom node dependencies
for d in "$COMFYUI_DIR/custom_nodes"/*/; do
    if [ -f "$d/requirements.txt" ]; then
        pip install -r "$d/requirements.txt" || true
    fi
done

# Download comfy_pixelization model checkpoints (from Google Drive)
PIXELIZATION_CKPT="$COMFYUI_DIR/custom_nodes/comfy_pixelization/checkpoints"
if [ -d "$PIXELIZATION_CKPT" ] && [ ! -f "$PIXELIZATION_CKPT/pixelart_vgg19.pth" ]; then
    echo "Downloading pixelization model checkpoints..."
    pip install -q gdown
    python -c "import gdown; gdown.download(id='1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM', output='$PIXELIZATION_CKPT/pixelart_vgg19.pth', quiet=False)"
    python -c "import gdown; gdown.download(id='1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az', output='$PIXELIZATION_CKPT/160_net_G_A.pth', quiet=False)"
    python -c "import gdown; gdown.download(id='17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_', output='$PIXELIZATION_CKPT/alias_net.pth', quiet=False)"
fi

echo ""
echo "=== Downloading Models ==="
mkdir -p "$MODELS_DIR/checkpoints" "$MODELS_DIR/loras" "$MODELS_DIR/ipadapter" "$MODELS_DIR/clip_vision"

# SDXL base (~6.9GB)
CKPT="$MODELS_DIR/checkpoints/sd_xl_base_1.0.safetensors"
if [ ! -f "$CKPT" ]; then
    echo "Downloading SDXL base..."
    wget -q --show-progress \
        https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors \
        -O "$CKPT"
fi

# Pixel Art XL LoRA (~130MB)
LORA="$MODELS_DIR/loras/pixel-art-xl.safetensors"
if [ ! -f "$LORA" ]; then
    echo "Downloading Pixel Art XL LoRA..."
    wget -q --show-progress \
        https://huggingface.co/nerijs/pixel-art-xl/resolve/main/pixel-art-xl.safetensors \
        -O "$LORA"
fi

# IPAdapter SDXL (for character consistency)
IPA="$MODELS_DIR/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors"
if [ ! -f "$IPA" ]; then
    echo "Downloading IPAdapter SDXL Plus..."
    wget -q --show-progress \
        https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors \
        -O "$IPA"
fi

# CLIP Vision for IPAdapter
CLIP_VISION="$MODELS_DIR/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
if [ ! -f "$CLIP_VISION" ]; then
    echo "Downloading CLIP Vision H..."
    wget -q --show-progress \
        https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors \
        -O "$CLIP_VISION"
fi

echo ""
echo "=== Installation Complete ==="
echo "Start server: bash tools/comfyui/serve.sh"
echo "Access at: http://<dgx-ip>:8188"
