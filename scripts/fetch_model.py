"""Fetch a trained model bundle for deployment with an optional SHA-256 integrity check."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

MODEL_URL = os.environ.get("MODEL_URL", "").strip()
MODEL_SHA256 = os.environ.get("MODEL_SHA256", "").strip().lower()
OUTPUT = Path(os.environ.get("MODEL_OUTPUT", "results/stacking_model.pkl"))

if not MODEL_URL:
    print("MODEL_URL not configured; leaving model absent. Live inference will remain unavailable.")
    raise SystemExit(0)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
print(f"Downloading trained model to {OUTPUT} ...")
with urlopen(MODEL_URL, timeout=60) as response, OUTPUT.open("wb") as target:
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        target.write(chunk)

if MODEL_SHA256:
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        OUTPUT.unlink(missing_ok=True)
        raise SystemExit("MODEL_SHA256 integrity check failed; deployment aborted.")

print(f"Model ready: {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
