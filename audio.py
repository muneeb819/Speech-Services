from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


class AudioError(ValueError):
    """Bad, unsupported or too-long audio supplied by a client."""


ALLOWED_RATES = {8000, 16000, 22050, 24000, 44100, 48000}
FORMAT_MIME = {
    "wav": "audio/wav", "pcm": "audio/L16", "ulaw": "audio/basic",
    "flac": "audio/flac", "mp3": "audio/mpeg", "opus": "audio/ogg",
}


def decode_file(path: Path, max_seconds: int, rate: int = 16000) -> np.ndarray:
    """Decode any ffmpeg-readable media to mono float32 at `rate`. Output is capped to bound memory."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate),
           "-t", str(max_seconds + 1), "-f", "f32le", "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Audio decoding timed out") from exc
    if proc.returncode != 0 or not proc.stdout:
        raise AudioError("Could not decode audio: " + proc.stderr.decode(errors="replace")[:200])
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if len(audio) / rate > max_seconds:
        raise AudioError(f"Audio longer than the {max_seconds}s limit")
    return audio


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")


def to_ulaw(pcm16: np.ndarray) -> bytes:
    """G.711 mu-law encode (8 kHz telephony: Asterisk/VICIdial friendly)."""
    s = pcm16.astype(np.int32)
    sign = (s < 0).astype(np.int32)
    s = np.minimum(np.abs(s), 32635) + 0x84
    exp = np.clip(np.floor(np.log2(s)).astype(np.int32) - 7, 0, 7)
    mant = (s >> (exp + 3)) & 0x0F
    return ((~((sign << 7) | (exp << 4) | mant)) & 0xFF).astype(np.uint8).tobytes()


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    return x if src == dst else soxr.resample(x, src, dst, quality="HQ").astype(np.float32)


def encode_audio(x: np.ndarray, src_rate: int, fmt: str, out_rate: int | None) -> tuple[bytes, str]:
    out_rate = out_rate or (8000 if fmt == "ulaw" else src_rate)
    if out_rate not in ALLOWED_RATES:
        raise AudioError(f"sample_rate must be one of {sorted(ALLOWED_RATES)}")
    y = resample(x, src_rate, out_rate)
    if fmt == "pcm":
        return float_to_pcm16(y).tobytes(), FORMAT_MIME[fmt]
    if fmt == "ulaw":
        return to_ulaw(float_to_pcm16(y)), FORMAT_MIME[fmt]
    if fmt in ("wav", "flac"):
        buf = io.BytesIO()
        sf.write(buf, y, out_rate, format=fmt.upper(), subtype="PCM_16")
        return buf.getvalue(), FORMAT_MIME[fmt]
    if fmt in ("mp3", "opus"):
        codec = (["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"] if fmt == "mp3"
                 else ["-c:a", "libopus", "-b:a", "48k", "-f", "ogg"])
        proc = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "s16le", "-ar", str(out_rate), "-ac", "1",
                               "-i", "pipe:0", *codec, "pipe:1"], input=float_to_pcm16(y).tobytes(), capture_output=True)
        if proc.returncode != 0:
            raise AudioError("Encoding failed")
        return proc.stdout, FORMAT_MIME[fmt]
    raise AudioError(f"Unsupported format {fmt}")


class StreamEncoder:
    """Stateful chunk encoder for streaming (pcm / ulaw only)."""

    def __init__(self, fmt: str, src_rate: int, out_rate: int | None):
        if fmt not in ("pcm", "ulaw"):
            raise AudioError("Streaming supports response_format pcm or ulaw")
        self.fmt = fmt
        self.out_rate = out_rate or (8000 if fmt == "ulaw" else src_rate)
        if self.out_rate not in ALLOWED_RATES:
            raise AudioError(f"sample_rate must be one of {sorted(ALLOWED_RATES)}")
        self._rs = None if src_rate == self.out_rate else soxr.ResampleStream(src_rate, self.out_rate, 1, dtype="float32")

    def _emit(self, y: np.ndarray) -> bytes:
        p = float_to_pcm16(y)
        return p.tobytes() if self.fmt == "pcm" else to_ulaw(p)

    def feed(self, x: np.ndarray) -> bytes:
        y = x if self._rs is None else self._rs.resample_chunk(x, last=False)
        return self._emit(y) if len(y) else b""

    def finish(self) -> bytes:
        if self._rs is None:
            return b""
        y = self._rs.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        return self._emit(y) if len(y) else b""
