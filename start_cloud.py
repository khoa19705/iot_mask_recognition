"""Fetch verified weights if absent, then serve the app with one model worker."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request


def prepare_model():
    target = Path(os.getenv("MASK_MODEL_PATH", "best.pt"))
    if target.is_file():
        return
    url = os.getenv("MODEL_URL", "")
    expected = os.getenv("MODEL_SHA256", "").lower()
    if not url.startswith("https://") or len(expected) != 64:
        raise RuntimeError("Provide best.pt or set HTTPS MODEL_URL and MODEL_SHA256.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            if not response.url.startswith("https://"):
                raise RuntimeError("Model download must stay on HTTPS.")
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                digest = hashlib.sha256()
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 512 * 1024 * 1024:
                        raise RuntimeError("Model exceeds 512 MB limit.")
                    output.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError("Model SHA256 mismatch.")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    prepare_model()
    port = int(os.getenv("PORT", "10000"))
    # A single worker keeps model memory and event-log locks in one process.
    command = [sys.executable, "-m", "gunicorn", "--bind", f"0.0.0.0:{port}",
               "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]
    raise SystemExit(subprocess.call(command))
