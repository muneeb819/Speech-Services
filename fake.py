"""CPU test doubles so the whole service can be exercised without a GPU. Never use in production."""
from __future__ import annotations

from typing import Iterator

import numpy as np


class FakeSTT:
    name = "fake-stt"

    def transcribe(self, audio, *, language, word_timestamps, vad, beam_size, batch_size, prompt) -> dict:
        dur = len(audio) / 16000
        seg = {"id": 0, "start": 0.0, "end": round(dur, 3), "text": f"fake transcript {dur:.1f}s",
               "avg_logprob": -0.1, "no_speech_prob": 0.01}
        if word_timestamps:
            seg["words"] = [{"word": "fake", "start": 0.0, "end": round(dur, 3), "probability": 0.9}]
        return {"language": language or "en", "language_probability": 1.0, "duration": dur,
                "text": seg["text"], "segments": [seg]}


class FakeTTS:
    name = "fake-tts"
    sample_rate = 24000

    def voices(self) -> list[str]:
        return ["af_heart", "am_adam"]

    def stream(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        if voice not in self.voices():
            raise ValueError(f"Unknown voice '{voice}'")
        for _ in [p for p in text.replace("\n", ". ").split(".") if p.strip()]:
            t = np.arange(int(0.2 * self.sample_rate)) / self.sample_rate
            yield (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
