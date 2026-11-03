from __future__ import annotations

from typing import Iterator

import numpy as np

from app.config import Settings

KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova",
    "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


class KokoroTTS:
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, s: Settings):
        from kokoro import KPipeline

        self._device, self._repo = s.tts_device, s.tts_repo
        self._pipes = {"a": KPipeline(lang_code="a", repo_id=s.tts_repo, device=s.tts_device)}
        self._extra = [v.strip() for v in s.extra_voices.split(",") if v.strip()]

    def voices(self) -> list[str]:
        return KOKORO_VOICES + self._extra

    def _pipe(self, voice: str):
        lang = voice[0]
        if lang not in self._pipes:
            from kokoro import KPipeline

            self._pipes[lang] = KPipeline(lang_code=lang, repo_id=self._repo, device=self._device,
                                          model=self._pipes["a"].model)
        return self._pipes[lang]

    def stream(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        if voice not in self.voices():
            raise ValueError(f"Unknown voice '{voice}'")
        for _, _, audio in self._pipe(voice)(text, voice=voice, speed=speed):
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
            if len(pcm):
                yield pcm
