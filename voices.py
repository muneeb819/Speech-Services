from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import shutil
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from app.audio import AudioError

log = logging.getLogger("voices")
VOICE_ID = re.compile(r"^[0-9a-f]{32}$")


class CloneDisabled(Exception):
    pass


class VoiceStore:
    """Registry of consented reference voices. Each voice carries its consent record; deleting a voice
    removes the reference audio (consent withdrawal)."""

    def __init__(self, root: Path):
        self.root = root
        self.audit = root.parent / "clone_audit.jsonl"

    def _dir(self, vid: str) -> Path:
        if not VOICE_ID.match(vid):
            raise KeyError(vid)
        return self.root / vid

    def create(self, *, owner: str, name: str, speaker_name: str, consent_statement: str,
               pcm24k: np.ndarray) -> dict:
        dur = len(pcm24k) / 24000
        if not 5 <= dur <= 60:
            raise AudioError("Reference audio must be between 5 and 60 seconds of clean speech")
        vid = uuid.uuid4().hex
        d = self._dir(vid)
        d.mkdir(parents=True)
        sf.write(d / "reference.wav", pcm24k, 24000, subtype="PCM_16")
        meta = {"id": vid, "name": name, "speaker_name": speaker_name, "consent_statement": consent_statement,
                "owner": owner, "created": time.time(), "duration": round(dur, 2),
                "sha256": hashlib.sha256((d / "reference.wav").read_bytes()).hexdigest()}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        self._log("voice_created", owner, vid)
        return meta

    def get(self, vid: str, owner: str | None = None) -> dict:
        try:
            meta = json.loads((self._dir(vid) / "meta.json").read_text())
        except (KeyError, FileNotFoundError):
            raise KeyError(vid) from None
        if owner is not None and meta["owner"] != owner:
            raise KeyError(vid)
        return meta

    def list(self, owner: str) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*/meta.json")):
            m = json.loads(p.read_text())
            if m["owner"] == owner:
                out.append({k: m[k] for k in ("id", "name", "speaker_name", "created", "duration")})
        return out

    def delete(self, vid: str, owner: str) -> None:
        self.get(vid, owner)
        shutil.rmtree(self._dir(vid))
        self._log("voice_deleted", owner, vid)

    def _log(self, event: str, owner: str, vid: str, chars: int | None = None) -> None:
        with self.audit.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, "owner": owner, "voice": vid, "chars": chars}) + "\n")

    def audit_generation(self, owner: str, vid: str, chars: int) -> None:
        self._log("synthesis", owner, vid, chars)


class CloneClient:
    def __init__(self, url: str, token: str):
        self.enabled = bool(url)
        self._c = httpx.AsyncClient(base_url=url or "http://disabled", timeout=httpx.Timeout(120, connect=5),
                                    headers={"X-Clone-Token": token})

    async def synthesize(self, voice_id: str, text: str) -> tuple[np.ndarray, int]:
        if not self.enabled:
            raise CloneDisabled()
        r = await self._c.post("/synthesize", json={"voice_id": voice_id, "text": text})
        if r.status_code != 200:
            raise RuntimeError(f"clone service error {r.status_code}")
        audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        return audio, sr

    async def healthy(self) -> bool:
        if not self.enabled:
            return True
        try:
            return (await self._c.get("/healthz")).status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self._c.aclose()
