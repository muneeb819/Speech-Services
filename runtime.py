from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from app import metrics
from app.auth import Authenticator
from app.config import Settings
from app.engines.base import Busy, Gate, aiter_blocking
from app.engines.vad import build_vad
from app.jobs import JobManager
from app.voices import CloneClient, VoiceStore

log = logging.getLogger("runtime")


class Runtime:
    """Process-wide state: models are loaded once at startup and shared by every request."""

    def __init__(self, s: Settings):
        self.settings = s
        self.auth = Authenticator(s)
        self.ready = False
        self.stt = self.tts = self.vad = None
        self.stt_gate = Gate(s.stt_concurrency, s.gate_timeout_seconds)
        self.tts_gate = Gate(s.tts_concurrency, s.gate_timeout_seconds)
        self.voices = VoiceStore(s.data_dir / "voices")
        self.clone = CloneClient(s.clone_url, s.clone_token)
        self.jobs = JobManager(self)

    async def startup(self) -> None:
        s = self.settings
        if s.backend == "fake":
            from app.engines.fake import FakeSTT, FakeTTS

            self.stt, self.tts = FakeSTT(), FakeTTS()
        else:
            from app.engines.stt_whisper import WhisperSTT
            from app.engines.tts_kokoro import KokoroTTS

            log.info("loading models", extra={"event": "load"})
            self.stt = await asyncio.to_thread(WhisperSTT, s)
            self.tts = await asyncio.to_thread(KokoroTTS, s)
        self.vad = build_vad("energy" if s.backend == "fake" else s.vad, s.vad_energy_threshold)
        await self._warmup()
        await self.jobs.start()
        self.ready = True
        log.info("ready", extra={"event": "ready"})

    async def _warmup(self) -> None:
        s = self.settings
        await asyncio.to_thread(self.stt.transcribe, np.zeros(16000, dtype=np.float32), language=s.stt_language,
                                word_timestamps=False, vad=False, beam_size=1, batch_size=0, prompt=None)
        await asyncio.to_thread(lambda: list(self.tts.stream("Ready.", s.tts_voice, 1.0)))

    async def shutdown(self) -> None:
        self.ready = False
        await self.jobs.stop()
        await self.clone.close()

    # ---- STT ----
    async def transcribe(self, audio: np.ndarray, *, mode: str = "sync", wait_forever: bool = False,
                         language: str | None, word_timestamps: bool = False, vad_filter: bool = True,
                         beam_size: int = 5, prompt: str | None = None, batch_size: int | None = None) -> dict:
        s = self.settings
        gate = self.stt_gate
        t0 = time.perf_counter()

        async def run() -> dict:
            return await asyncio.to_thread(
                self.stt.transcribe, audio, language=language, word_timestamps=word_timestamps, vad=vad_filter,
                beam_size=beam_size, batch_size=s.stt_batch_size if batch_size is None else batch_size, prompt=prompt)

        try:
            if wait_forever:
                async with gate._sem:  # background jobs queue behind interactive traffic without timing out
                    result = await run()
            else:
                async with gate:
                    result = await run()
        except Busy:
            metrics.BUSY_REJECTIONS.labels("stt").inc()
            raise
        metrics.STT_AUDIO_SECONDS.labels(mode).inc(len(audio) / 16000)
        metrics.STT_PROC_SECONDS.labels(mode).observe(time.perf_counter() - t0)
        return result

    # ---- TTS ----
    async def tts_chunks(self, text: str, voice: str, speed: float, owner: str):
        """Async iterator of (float32 chunk, sample_rate). Handles stock Kokoro voices and clone:<id> voices."""
        if voice.startswith("clone:"):
            vid = voice[6:]
            self.voices.get(vid, owner)  # KeyError if missing / not owned
            self.voices.audit_generation(owner, vid, len(text))
            metrics.TTS_CHARS.labels("clone").inc(len(text))
            try:
                async with self.tts_gate:
                    audio, sr = await self.clone.synthesize(vid, text)
            except Busy:
                metrics.BUSY_REJECTIONS.labels("tts").inc()
                raise
            yield audio, sr
            return
        if voice not in self.tts.voices():
            raise ValueError(f"Unknown voice '{voice}'")
        metrics.TTS_CHARS.labels("stock").inc(len(text))
        t0 = time.perf_counter()
        first = True
        try:
            async with self.tts_gate:
                async for chunk in aiter_blocking(lambda: self.tts.stream(text, voice, speed)):
                    if first:
                        metrics.TTS_FIRST_CHUNK.observe(time.perf_counter() - t0)
                        first = False
                    yield chunk, self.tts.sample_rate
        except Busy:
            metrics.BUSY_REJECTIONS.labels("tts").inc()
            raise
