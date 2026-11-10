"""Tests for verified, atomic model artifact downloads."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts import fetch_model


class FakeResponse:
    def __init__(self, payload: bytes, final_url: str = "https://artifacts.example/model.pkl"):
        self.payload = payload
        self.offset = 0
        self.headers: dict[str, str] = {"Content-Length": str(len(payload))}
        self.final_url = final_url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def geturl(self) -> str:
        return self.final_url

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.payload) - self.offset
        chunk = self.payload[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


def test_download_verified_model_writes_only_after_digest_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"trusted model artifact"
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "models" / "stacking_model.pkl"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"previous-good-artifact")
    monkeypatch.setattr(fetch_model, "urlopen", lambda request, timeout: FakeResponse(payload))

    size = fetch_model.download_verified_model(
        "https://artifacts.example/model.pkl", expected, destination, max_bytes=1024
    )

    assert size == len(payload)
    assert destination.read_bytes() == payload
    assert not destination.with_name(destination.name + ".part").exists()


def test_digest_mismatch_preserves_existing_model_and_removes_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "stacking_model.pkl"
    destination.write_bytes(b"previous-good-artifact")
    monkeypatch.setattr(fetch_model, "urlopen", lambda request, timeout: FakeResponse(b"tampered"))

    with pytest.raises(ValueError, match="SHA-256"):
        fetch_model.download_verified_model(
            "https://artifacts.example/model.pkl", "0" * 64, destination, max_bytes=1024
        )

    assert destination.read_bytes() == b"previous-good-artifact"
    assert not destination.with_name(destination.name + ".part").exists()


def test_insecure_urls_are_rejected_before_network_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network access must not occur for an HTTP URL.")

    monkeypatch.setattr(fetch_model, "urlopen", forbidden_network)
    with pytest.raises(ValueError, match="HTTPS"):
        fetch_model.download_verified_model(
            "http://artifacts.example/model.pkl", "0" * 64,
            tmp_path / "stacking_model.pkl", max_bytes=1024
        )


def test_oversized_artifact_is_rejected_without_replacing_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "stacking_model.pkl"
    destination.write_bytes(b"previous-good-artifact")
    payload = b"x" * 20
    monkeypatch.setattr(fetch_model, "urlopen", lambda request, timeout: FakeResponse(payload))

    with pytest.raises(ValueError, match="MODEL_MAX_BYTES"):
        fetch_model.download_verified_model(
            "https://artifacts.example/model.pkl",
            hashlib.sha256(payload).hexdigest(),
            destination,
            max_bytes=10,
        )

    assert destination.read_bytes() == b"previous-good-artifact"
