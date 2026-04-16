#!/bin/bash
# Start ComfyUI server accessible from local network
COMFYUI_DIR="$HOME/ComfyUI"
source "$COMFYUI_DIR/venv/bin/activate"
cd "$COMFYUI_DIR"
python main.py --listen 0.0.0.0 --port 8188 --preview-method auto --force-fp16 "$@"
