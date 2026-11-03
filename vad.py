from __future__ import annotations

import numpy as np

SR = 16000


class EnergyVAD:
    """Dependency-free fallback VAD (RMS threshold over 30 ms frames)."""

    def __init__(self, threshold: float = 0.012):
        self.threshold = threshold

    def spans(self, audio: np.ndarray) -> list[tuple[int, int]]:
        frame = int(0.03 * SR)
        n = len(audio) // frame
        if n == 0:
            return []
        rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
        voiced = rms > self.threshold
        out, start = [], None
        for i, v in enumerate(voiced):
            if v and start is None:
                start = i
            elif not v and start is not None:
                out.append((start * frame, i * frame))
                start = None
        if start is not None:
            out.append((start * frame, n * frame))
        return out


class SileroVAD:
    """Silero VAD as bundled with faster-whisper (no extra dependency)."""

    def __init__(self):
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        self._fn = get_speech_timestamps
        self._opts = VadOptions(min_silence_duration_ms=100, speech_pad_ms=30)

    def spans(self, audio: np.ndarray) -> list[tuple[int, int]]:
        if len(audio) < SR // 10:
            return []
        return [(t["start"], t["end"]) for t in self._fn(audio, self._opts, sampling_rate=SR)]


def build_vad(kind: str, energy_threshold: float):
    if kind == "silero":
        return SileroVAD()
    return EnergyVAD(energy_threshold)
