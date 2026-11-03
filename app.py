"""Chatterbox Turbo voice-cloning worker. Runs in its own container (pinned PyTorch stack differs from the main service).

Safeguards: only voices registered through the main API (with a consent record) can be used, the shared secret is
required, and the model's default Perth watermark is left enabled."""
from __future__ import annotations

import hmac
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

DATA_DIR = Path(os.environ.get("CLONE_DATA_DIR", "/data"))
TOKEN = os.environ.get("CLONE_TOKEN", "")
FAKE = os.environ.get("CLONE_FAKE") == "1"
MAX_CHARS = int(os.environ.get("CLONE_MAX_CHARS", "2000"))

_model = None
_sr = 24000
_lock = threading.Lock()  # one generation at a time per GPU worker
_prepared: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    load()
    yield


def load() -> None:
    global _model, _sr
    if not TOKEN:
        raise RuntimeError("CLONE_TOKEN must be set")
    if FAKE:
        return
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    model_dir = os.environ.get("CLONE_MODEL_DIR")
    _model = ChatterboxTurboTTS.from_local(model_dir, device="cuda") if model_dir else \
        ChatterboxTurboTTS.from_pretrained(device="cuda")
    _sr = _model.sr


app = FastAPI(title="clone-worker", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


class Req(BaseModel):
    voice_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    text: str = Field(min_length=1)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/synthesize")
def synthesize(req: Req, x_clone_token: str = Header("")):
    global _prepared
    if not hmac.compare_digest(x_clone_token.encode(), TOKEN.encode()):
        raise HTTPException(401, "bad token")
    if len(req.text) > MAX_CHARS:
        raise HTTPException(413, "text too long")
    ref = DATA_DIR / "voices" / req.voice_id / "reference.wav"
    if not ref.is_file():  # voice deleted (consent withdrawn) or never registered
        raise HTTPException(404, "voice not found")
    buf = io.BytesIO()
    if FAKE:
        t = np.arange(int(0.3 * _sr)) / _sr
        sf.write(buf, (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), _sr, format="WAV")
        return Response(buf.getvalue(), media_type="audio/wav")
    import torch

    with _lock, torch.inference_mode():
        if hasattr(_model, "prepare_conditionals"):  # cache reference conditioning between requests
            if _prepared != req.voice_id:
                _model.prepare_conditionals(str(ref))
                _prepared = req.voice_id
            wav = _model.generate(req.text)
        else:
            wav = _model.generate(req.text, audio_prompt_path=str(ref))
    sf.write(buf, wav.detach().cpu().numpy().reshape(-1), _sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")
