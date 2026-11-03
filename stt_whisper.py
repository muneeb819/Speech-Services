from __future__ import annotations

import numpy as np

from app.config import Settings


def _segments_to_result(segments, info, words: bool) -> dict:
    rows = []
    for s in segments:
        row = {"id": s.id, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
               "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob}
        if words:
            row["words"] = [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3),
                             "probability": w.probability} for w in (s.words or [])]
        rows.append(row)
    return {"language": info.language, "language_probability": info.language_probability,
            "duration": info.duration, "text": " ".join(r["text"] for r in rows).strip(), "segments": rows}


class WhisperSTT:
    name = "faster-whisper"

    def __init__(self, s: Settings):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.model = WhisperModel(s.stt_model, device=s.stt_device, compute_type=s.stt_compute_type,
                                  cpu_threads=s.stt_cpu_threads, num_workers=max(1, s.stt_concurrency))
        self.batched = BatchedInferencePipeline(model=self.model)

    def transcribe(self, audio: np.ndarray, *, language, word_timestamps, vad, beam_size, batch_size, prompt) -> dict:
        opts = dict(language=language, task="transcribe", beam_size=beam_size, vad_filter=vad,
                    word_timestamps=word_timestamps, condition_on_previous_text=False, initial_prompt=prompt)
        if batch_size > 0:
            segments, info = self.batched.transcribe(audio, batch_size=batch_size, **opts)
        else:
            segments, info = self.model.transcribe(audio, **opts)
        return _segments_to_result(segments, info, word_timestamps)  # consumes the generator
