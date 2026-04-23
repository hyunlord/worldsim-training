"""REST client wrapper for the ComfyUI API.

Provides a clean interface for submitting workflows, polling for completion,
and downloading output images.  Uses ``urllib.request`` (no third-party deps)
to stay consistent with the rest of the comfyui tooling.
"""
import json
import time
import uuid
import urllib.request
import urllib.error
from pathlib import Path


class ComfyClientError(Exception):
    """Base exception for ComfyUI client errors."""
    pass


class ComfyClient:
    """Thin wrapper around the ComfyUI REST API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        client_id: str | None = None,
    ):
        """
        Args:
            base_url: ComfyUI server URL (no trailing slash).
            client_id: Unique client identifier.  Auto-generated if *None*.
        """
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or uuid.uuid4().hex[:12]

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, timeout: int = 10) -> dict:
        """HTTP GET, returns parsed JSON."""
        url = f"{self.base_url}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())

    def _post(self, path: str, data: dict, timeout: int = 30) -> dict:
        """HTTP POST with JSON body, returns parsed JSON.

        Raises :class:`ComfyClientError` on HTTP errors.
        """
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            resp_body = e.read().decode(errors="replace")
            raise ComfyClientError(
                f"POST {path} failed (HTTP {e.code}): {resp_body}"
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow (API format) to ComfyUI.

        POST ``/prompt`` with ``{"prompt": workflow, "client_id": ...}``.

        Returns:
            The ``prompt_id`` string assigned by the server.

        Raises:
            ComfyClientError: If submission fails or the response lacks a
                ``prompt_id``.
        """
        resp = self._post("/prompt", {
            "prompt": workflow,
            "client_id": self.client_id,
        })
        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            raise ComfyClientError(
                f"Server response missing prompt_id: {json.dumps(resp)[:500]}"
            )
        return prompt_id

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout_s: int = 300,
    ) -> dict:
        """Poll ``/history/{prompt_id}`` until the job completes.

        Returns:
            The history entry dict containing ``outputs`` and ``status``.

        Raises:
            ComfyClientError: On timeout or if ComfyUI reports an execution
                error.
        """
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                history = self._get(f"/history/{prompt_id}")
            except Exception:
                time.sleep(2)
                continue

            if prompt_id not in history:
                time.sleep(2)
                continue

            entry = history[prompt_id]
            status = entry.get("status", {})

            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                raise ComfyClientError(
                    f"Execution error for {prompt_id}: {msgs}"
                )

            if status.get("completed") or status.get("status_str") == "success":
                return entry

            time.sleep(2)

        raise ComfyClientError(
            f"Timeout waiting for {prompt_id} after {timeout_s}s"
        )

    def download_output(
        self,
        filename: str,
        subfolder: str,
        type_: str,
        dest_path: Path,
    ) -> None:
        """Download an output image from ComfyUI.

        GET ``/view?filename=...&subfolder=...&type=...`` and write the raw
        bytes to *dest_path*.  Parent directories are created automatically.
        """
        url = (
            f"{self.base_url}/view"
            f"?filename={urllib.request.quote(filename)}"
            f"&subfolder={urllib.request.quote(subfolder)}"
            f"&type={urllib.request.quote(type_)}"
        )
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            dest_path.write_bytes(r.read())

    def get_output_images(self, history_entry: dict) -> list[dict]:
        """Extract image info dicts from a history entry.

        Returns:
            List of ``{"filename": str, "subfolder": str, "type": str}``
            dicts.
        """
        images = []
        for node_output in history_entry.get("outputs", {}).values():
            for img in node_output.get("images", []):
                images.append({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
        return images

    def object_info(self, node_type: str) -> dict:
        """GET ``/object_info/{node_type}``.

        Returns:
            The schema dict for *node_type*.
        """
        return self._get(f"/object_info/{node_type}")

    def system_stats(self) -> dict:
        """GET ``/system_stats``.

        Returns:
            Server statistics dict (devices, VRAM, etc.).
        """
        return self._get("/system_stats")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ComfyClient(base_url={self.base_url!r}, "
            f"client_id={self.client_id!r})"
        )
