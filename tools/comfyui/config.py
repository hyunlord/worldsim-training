"""Central configuration for WorldSim ComfyUI pipeline.

Environment variable overrides supported for all settings. Defaults
match the local DGX setup (DGX Spark, port 8188).

Environment variables:
    COMFYUI_HOST        - Server hostname (default: 127.0.0.1)
    COMFYUI_PORT        - Server port (default: 8188)
    COMFYUI_URL         - Full URL (overrides HOST+PORT if set)

    COMFYUI_TIMEOUT_GET        - HTTP GET timeout (default: 10s)
    COMFYUI_TIMEOUT_POST       - HTTP POST timeout (default: 30s)
    COMFYUI_TIMEOUT_HEALTH     - Server health check (default: 2s)
    COMFYUI_TIMEOUT_GENERATE   - Full generation wait (default: 300s)
    COMFYUI_TIMEOUT_VALIDATE   - validate_and_fix loop (default: 180s)
    COMFYUI_TIMEOUT_STARTUP    - Server startup wait (default: 90s)

    COMFYUI_MAX_ATTEMPTS       - Retry attempts in batch_generate (default: 5)
    COMFYUI_MAX_ITERATIONS     - validate_and_fix loop cap (default: 5)

Usage:
    from config import COMFYUI_URL, TIMEOUT_VALIDATE, MAX_ITERATIONS

    # Override example:
    #   COMFYUI_HOST=remote.dgx.local python tools/comfyui/validate_and_fix.py
"""
import os


# ═══════════════════════════════════════════════════════════════════
# Server connection
# ═══════════════════════════════════════════════════════════════════

COMFYUI_HOST: str = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT: int = int(os.environ.get("COMFYUI_PORT", "8188"))

# Allow full URL override (takes precedence over HOST+PORT)
COMFYUI_URL: str = os.environ.get(
    "COMFYUI_URL",
    f"http://{COMFYUI_HOST}:{COMFYUI_PORT}",
)


# ═══════════════════════════════════════════════════════════════════
# Timeouts (seconds)
# ═══════════════════════════════════════════════════════════════════

TIMEOUT_HTTP_GET: int = int(os.environ.get("COMFYUI_TIMEOUT_GET", "10"))
TIMEOUT_HTTP_POST: int = int(os.environ.get("COMFYUI_TIMEOUT_POST", "30"))
TIMEOUT_HEALTH: int = int(os.environ.get("COMFYUI_TIMEOUT_HEALTH", "2"))

TIMEOUT_GENERATE: int = int(os.environ.get("COMFYUI_TIMEOUT_GENERATE", "300"))
"""Full image generation wait — covers SDXL 20-step inference (~30s on GB10)
plus safety margin for queue, VAE decode, pixelization, and save."""

TIMEOUT_VALIDATE: int = int(os.environ.get("COMFYUI_TIMEOUT_VALIDATE", "180"))
"""validate_and_fix execute-validate-fix loop timeout. Shorter than GENERATE
because validate stage uses dummy prompts for structural checks only."""

TIMEOUT_STARTUP: int = int(os.environ.get("COMFYUI_TIMEOUT_STARTUP", "90"))
"""Server boot + model load time. SDXL base loading takes ~60s on GB10."""


# ═══════════════════════════════════════════════════════════════════
# Retry / iteration caps
# ═══════════════════════════════════════════════════════════════════

MAX_ATTEMPTS: int = int(os.environ.get("COMFYUI_MAX_ATTEMPTS", "5"))
"""Retry attempts when a single variant generation fails.
Used by batch_generate.py's per-variant retry loop."""

MAX_ITERATIONS: int = int(os.environ.get("COMFYUI_MAX_ITERATIONS", "5"))
"""Maximum execute-validate-fix iterations before giving up on a workflow.
Used by validate_and_fix.py's main loop."""
