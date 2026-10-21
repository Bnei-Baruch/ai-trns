import subprocess
from types import SimpleNamespace as NS

import pytest

from ai_trns import protocol
from ai_trns.words import build_document

API_KEY, WORKER_KEY = "client-key", "worker-key"
H = {"X-API-Key": API_KEY}
W = {protocol.KEY_HEADER: WORKER_KEY, protocol.VERSION_HEADER: str(protocol.VERSION)}


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "in.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                    "-y", str(path)], check=True)
    return path


@pytest.fixture
def api_config(tmp_path, monkeypatch):
    from ai_trns.api import config
    monkeypatch.setattr(config, "JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(config, "API_KEYS", {"tests": API_KEY})
    monkeypatch.setattr(config, "OIDC_ISSUER", "")
    monkeypatch.setattr(config, "WORKER_KEY", WORKER_KEY)
    monkeypatch.setattr(config, "ALLOWED_URL_HOSTS", ["cdn.example.org"])
    monkeypatch.setattr(config, "TRANSCRIBE_UPLOADS", True)
    return config


@pytest.fixture
def client(api_config):
    from fastapi.testclient import TestClient
    from ai_trns.api.app import app
    with TestClient(app) as c:
        yield c


def words_doc(duration_s: float = 4.0) -> dict:
    seg = NS(start=0.0, end=1.0, text=" שלום עולם.",
             words=[NS(word=" שלום", start=0.0, end=0.4, probability=0.9),
                    NS(word=" עולם.", start=0.5, end=0.9, probability=0.9)])
    return build_document([seg], duration_s=duration_s, language="he", model="fake")
