"""Local subprocess lifecycle manager for ComfyUI server.

Starts, monitors, and waits for a ComfyUI server running on the same machine.
Uses only stdlib (subprocess, urllib, json, time, pathlib) — no external deps.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path


class ServerStartupError(Exception):
    """Raised when the ComfyUI server cannot be started or reached."""
    pass


class ServerManager:
    """Manage the lifecycle of a local ComfyUI server process.

    The server is treated as a **shared resource** — once started it is never
    killed.  ``ensure_running()`` is idempotent: it reuses an already-running
    server, waits for one that is mid-startup, or launches a fresh one via
    ``serve.sh``.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        serve_script: Path | None = None,
        startup_timeout_s: int = 90,
        poll_interval_s: float = 2.0,
        log_path: Path = Path("/tmp/comfyui_batch.log"),
    ):
        """
        Args:
            base_url: ComfyUI server URL.
            serve_script: Path to serve.sh.  Defaults to
                ``tools/comfyui/serve.sh`` relative to the repository root.
            startup_timeout_s: Max seconds to wait for server startup.
            poll_interval_s: Seconds between health-check polls.
            log_path: Where to redirect server stdout/stderr.
        """
        self.base_url = base_url.rstrip("/")
        if serve_script is None:
            # Walk up from this file to the repo root (4 levels).
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            serve_script = repo_root / "tools" / "comfyui" / "serve.sh"
        self.serve_script = Path(serve_script)
        self.startup_timeout_s = startup_timeout_s
        self.poll_interval_s = poll_interval_s
        self.log_path = Path(log_path)
        self._child_pid: int | None = None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_alive(self) -> bool:
        """GET ``/system_stats`` with a 2 s timeout.  True if 200 OK."""
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/system_stats", timeout=2
            ) as r:
                return r.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def ensure_running(self) -> bool:
        """Ensure the ComfyUI server is running and ready.

        Returns:
            ``True`` when the server is confirmed reachable.

        Raises:
            ServerStartupError: If the server cannot be reached within
                *startup_timeout_s* seconds, or ``serve.sh`` is missing.

        Workflow:
            1. ``is_alive()`` already? → reuse immediately.
            2. ``pgrep`` for an existing server process (mid-startup) →
               skip straight to the poll loop.
            3. Otherwise launch ``serve.sh`` as a detached background process.
            4. Poll ``is_alive()`` every *poll_interval_s* until ready.
            5. On timeout, surface the last 50 log lines and raise.
        """

        # Step 1 — fast path: server is already responsive ----------------
        if self.is_alive():
            print("[setup] Reusing existing ComfyUI server")
            return True

        # Step 2 — look for a server process that hasn't finished booting -
        need_launch = True
        proc = subprocess.run(
            ["pgrep", "-f", "python main.py.*--port 8188"],
            capture_output=True,
            text=True,
        )
        if proc.stdout.strip():
            print("[setup] Server process found (mid-startup), waiting for ready signal...")
            need_launch = False

        # Step 3 — launch serve.sh ----------------------------------------
        if need_launch:
            if not self.serve_script.exists():
                raise ServerStartupError(
                    f"serve.sh not found at {self.serve_script}"
                )
            log_file = open(self.log_path, "ab")  # noqa: SIM115
            child = subprocess.Popen(
                ["bash", str(self.serve_script)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # detach from Claude Code's group
            )
            self._child_pid = child.pid
            print(
                f"[setup] Launched ComfyUI (pid={child.pid}, log={self.log_path})"
            )

        # Step 4 — poll readiness -----------------------------------------
        start = time.monotonic()
        deadline = start + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.is_alive():
                elapsed = time.monotonic() - start
                print(f"[setup] Server ready after {elapsed:.1f}s")
                return True
            time.sleep(self.poll_interval_s)

        # Step 5 — timeout ------------------------------------------------
        tail = self.tail_log(50)
        raise ServerStartupError(
            f"Could not start ComfyUI server on {self.base_url} within "
            f"{self.startup_timeout_s}s.\n"
            f"Last {min(50, len(tail.splitlines()))} log lines:\n{tail}\n\n"
            f"You can try launching manually:\n"
            f"    bash {self.serve_script}\n"
            f"And verify with:\n"
            f"    curl -f {self.base_url}/system_stats"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def tail_log(self, n_lines: int = 50) -> str:
        """Return last *n_lines* of the log file for debugging."""
        try:
            with open(self.log_path, "r") as f:
                lines = f.readlines()
            return "".join(lines[-n_lines:])
        except FileNotFoundError:
            return "(log file not found)"

    def server_version(self) -> str:
        """Get the ComfyUI version from ``/system_stats``, or ``'unknown'``."""
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/system_stats", timeout=5
            ) as r:
                data = json.loads(r.read())
            return data.get("system", {}).get("comfyui_version", "unknown")
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ServerManager(base_url={self.base_url!r}, "
            f"serve_script={self.serve_script!r})"
        )
