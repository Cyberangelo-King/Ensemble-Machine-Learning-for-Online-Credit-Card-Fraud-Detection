"""Download and verify a model bundle before atomically publishing it.

Only HTTPS URLs are accepted. The existing model is left untouched unless the
download completes and its SHA-256 digest matches the configured trusted value.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

MODEL_URL = os.environ.get("MODEL_URL", "").strip()
MODEL_SHA256 = os.environ.get("MODEL_SHA256", "").strip().lower()
OUTPUT = Path(os.environ.get("MODEL_OUTPUT", "results/stacking_model.pkl"))
MAX_MODEL_BYTES = int(os.environ.get("MODEL_MAX_BYTES", str(2 * 1024 * 1024 * 1024)))
CHUNK_SIZE = 1024 * 1024


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def download_verified_model(
    model_url: str,
    expected_sha256: str,
    output_path: Path,
    max_bytes: int = MAX_MODEL_BYTES,
) -> int:
    """Stream an HTTPS model download, verify it, then atomically replace output."""
    parsed = urlparse(model_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("MODEL_URL must be an absolute HTTPS URL.")
    if not _valid_digest(expected_sha256):
        raise ValueError("MODEL_SHA256 must be a 64-character hexadecimal digest.")
    if max_bytes <= 0:
        raise ValueError("MODEL_MAX_BYTES must be greater than zero.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".part")
    digest = hashlib.sha256()
    downloaded = 0

    try:
        request = Request(model_url, headers={"User-Agent": "fraud-research-model-fetcher/1.0"})
        with urlopen(request, timeout=60) as response:
            final_url = urlparse(response.geturl())
            if final_url.scheme != "https":
                raise ValueError("The model download redirected away from HTTPS.")
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError("The model artifact exceeds MODEL_MAX_BYTES.")

            with temporary_path.open("wb") as target:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError("The model artifact exceeds MODEL_MAX_BYTES.")
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())

        if downloaded == 0:
            raise ValueError("The downloaded model artifact is empty.")
        if digest.hexdigest() != expected_sha256:
            raise ValueError("MODEL_SHA256 integrity check failed.")
        os.replace(temporary_path, output_path)
        return downloaded
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    if not MODEL_URL:
        raise SystemExit("MODEL_URL is required. Supply a trusted HTTPS model artifact URL.")
    if not _valid_digest(MODEL_SHA256):
        raise SystemExit("MODEL_SHA256 must be a 64-character hexadecimal SHA-256 digest.")
    try:
        print(f"Downloading verified model artifact to {OUTPUT} ...")
        size = download_verified_model(MODEL_URL, MODEL_SHA256, OUTPUT)
    except Exception as exc:
        raise SystemExit(f"Model download failed; deployment aborted: {exc}") from exc
    print(f"Model ready: {OUTPUT} ({size:,} bytes, SHA-256 verified)")


if __name__ == "__main__":
    main()
