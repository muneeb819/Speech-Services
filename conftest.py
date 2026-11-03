import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

KEY = "test-secret"
H = {"Authorization": f"Bearer {KEY}"}


def wav_bytes(seconds=1.0, freq=440.0, sr=16000, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    s = Settings(backend="fake", data_dir=tmp_path, api_keys=f"alice:{KEY},bob:other-secret",
                 sync_max_seconds=5, max_upload_mb=1, rate_limit_per_min=1000, live_endpoint_ms=400)
    with TestClient(create_app(s)) as c:
        yield c
