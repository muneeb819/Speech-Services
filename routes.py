from __future__ import annotations

import io
import json
import logging
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app import formats
from app.audio import AudioError, StreamEncoder, decode_file, encode_audio, FORMAT_MIME
from app.auth import Principal, require_auth
from app.engines.base import Busy
from app.voices import CloneDisabled

log = logging.getLogger("routes")
router = APIRouter()
RESULT_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}


def rt(request: Request):
    return request.app.state.rt


async def save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as f:
            while chunk := await upload.read(1 << 20):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"Upload exceeds {max_bytes // (1 << 20)} MB limit")
                f.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Empty upload")


def render(result: dict, fmt: str):
    if fmt == "json":
        return JSONResponse({"text": result["text"]})
    if fmt == "verbose_json":
        return JSONResponse(result)
    if fmt == "text":
        return PlainTextResponse(formats.to_text(result))
    if fmt == "srt":
        return PlainTextResponse(formats.to_srt(result), media_type="application/x-subrip")
    return PlainTextResponse(formats.to_vtt(result), media_type="text/vtt")


def _lang(request: Request, language: str | None) -> str | None:
    if language is None or language == "":
        return rt(request).settings.stt_language
    return None if language == "auto" else language


# ------------------------------------------------------------------ health
@router.get("/healthz", include_in_schema=False)
async def healthz():
    return {"status": "ok"}


@router.get("/readyz", include_in_schema=False)
async def readyz(request: Request):
    r = rt(request)
    if not r.ready or not await r.clone.healthy():
        raise HTTPException(503, "not ready")
    return {"status": "ready", "stt": r.stt.name, "tts": r.tts.name, "clone": r.clone.enabled}


@router.get("/metrics", include_in_schema=False)
async def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------------ STT
@router.post("/v1/audio/transcriptions")
async def transcribe(request: Request, file: UploadFile = File(...), model: str = Form("whisper-1"),
                     language: str | None = Form(None), response_format: str = Form("json"),
                     word_timestamps: bool = Form(False), vad_filter: bool = Form(True),
                     prompt: str | None = Form(None), beam_size: int = Form(5, ge=1, le=10),
                     _: Principal = Depends(require_auth)):
    r = rt(request)
    s = r.settings
    if response_format not in RESULT_FORMATS:
        raise HTTPException(400, f"response_format must be one of {sorted(RESULT_FORMATS)}")
    tmp = s.data_dir / "tmp" / uuid.uuid4().hex
    try:
        await save_upload(file, tmp, s.max_upload_mb << 20)
        import asyncio
        audio = await asyncio.to_thread(decode_file, tmp, s.max_audio_seconds)
    finally:
        tmp.unlink(missing_ok=True)
    if len(audio) / 16000 > s.sync_max_seconds:
        raise HTTPException(413, f"Audio over {s.sync_max_seconds}s: submit it to POST /v1/jobs/transcriptions")
    result = await r.transcribe(audio, language=_lang(request, language), word_timestamps=word_timestamps or
                                response_format == "verbose_json", vad_filter=vad_filter, beam_size=beam_size,
                                prompt=prompt)
    return render(result, response_format)


# ------------------------------------------------------------------ jobs
@router.post("/v1/jobs/transcriptions", status_code=202)
async def create_job(request: Request, file: UploadFile = File(...), language: str | None = Form(None),
                     word_timestamps: bool = Form(True), vad_filter: bool = Form(True),
                     prompt: str | None = Form(None), beam_size: int = Form(5, ge=1, le=10),
                     p: Principal = Depends(require_auth)):
    r = rt(request)
    s = r.settings
    if r.jobs.queue.full():
        raise HTTPException(503, "Job queue full", headers={"Retry-After": "30"})
    params = {"language": _lang(request, language), "word_timestamps": word_timestamps,
              "vad_filter": vad_filter, "prompt": prompt, "beam_size": beam_size}
    import asyncio
    jid = await asyncio.to_thread(r.jobs.store.create, p.key_id, file.filename or "upload", params)
    try:
        await save_upload(file, r.jobs.upload_path(jid), s.max_upload_mb << 20)
    except BaseException:
        await asyncio.to_thread(r.jobs.store.delete, jid)
        raise
    if not r.jobs.submit(jid):
        r.jobs.upload_path(jid).unlink(missing_ok=True)
        await asyncio.to_thread(r.jobs.store.delete, jid)
        raise HTTPException(503, "Job queue full", headers={"Retry-After": "30"})
    return {"id": jid, "status": "queued"}


async def _owned_job(request: Request, jid: str, p: Principal) -> dict:
    import asyncio
    job = await asyncio.to_thread(rt(request).jobs.store.get, jid)
    if not job or job["owner"] != p.key_id:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/v1/jobs/{jid}")
async def job_status(request: Request, jid: str, p: Principal = Depends(require_auth)):
    j = await _owned_job(request, jid, p)
    return {k: j[k] for k in ("id", "status", "created", "updated", "filename", "error")}


@router.get("/v1/jobs/{jid}/result")
async def job_result(request: Request, jid: str, format: str = "verbose_json", p: Principal = Depends(require_auth)):
    j = await _owned_job(request, jid, p)
    if j["status"] != "completed":
        raise HTTPException(409, f"Job is {j['status']}")
    if format not in RESULT_FORMATS:
        raise HTTPException(400, f"format must be one of {sorted(RESULT_FORMATS)}")
    return render(json.loads(rt(request).jobs.result_path(jid).read_text()), format)


@router.delete("/v1/jobs/{jid}", status_code=204)
async def job_delete(request: Request, jid: str, p: Principal = Depends(require_auth)):
    import asyncio
    j = await _owned_job(request, jid, p)
    if j["status"] in ("queued", "running"):
        raise HTTPException(409, "Job still active")
    r = rt(request)
    r.jobs.result_path(jid).unlink(missing_ok=True)
    await asyncio.to_thread(r.jobs.store.delete, jid)


# ------------------------------------------------------------------ TTS
class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str = Field(min_length=1)
    voice: str = Field("af_heart", pattern=r"^(clone:[0-9a-f]{32}|[a-z]{2}_[a-z0-9]+)$")
    response_format: Literal["wav", "pcm", "ulaw", "flac", "mp3", "opus"] = "wav"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    sample_rate: int | None = None
    stream: bool = False


@router.get("/v1/voices/stock")
async def stock_voices(request: Request, _: Principal = Depends(require_auth)):
    return {"voices": rt(request).tts.voices()}


@router.post("/v1/audio/speech")
async def speech(request: Request, body: SpeechRequest, p: Principal = Depends(require_auth)):
    r = rt(request)
    if len(body.input) > r.settings.max_tts_chars:
        raise HTTPException(413, f"input exceeds {r.settings.max_tts_chars} characters")
    if body.voice.startswith("clone:") and not r.clone.enabled:
        raise HTTPException(501, "Voice cloning is not enabled on this deployment")
    try:
        if body.stream:
            enc_probe = StreamEncoder(body.response_format, 24000, body.sample_rate)  # validates format early
            gen = r.tts_chunks(body.input, body.voice, body.speed, p.key_id)
            first = await gen.__anext__()  # surfaces unknown voice / busy as proper HTTP errors

            async def body_iter():
                enc = StreamEncoder(body.response_format, first[1], body.sample_rate)
                try:
                    out = enc.feed(first[0])
                    if out:
                        yield out
                    async for chunk, sr in gen:
                        out = enc.feed(chunk)
                        if out:
                            yield out
                    tail = enc.finish()
                    if tail:
                        yield tail
                finally:
                    await gen.aclose()

            _ = enc_probe
            return StreamingResponse(body_iter(), media_type=FORMAT_MIME[body.response_format],
                                     headers={"X-Sample-Rate": str(enc_probe.out_rate)})
        parts, sr = [], 24000
        async for chunk, sr in r.tts_chunks(body.input, body.voice, body.speed, p.key_id):
            parts.append(chunk)
        import asyncio
        data, mime = await asyncio.to_thread(encode_audio, np.concatenate(parts), sr, body.response_format,
                                             body.sample_rate)
        return Response(data, media_type=mime)
    except StopAsyncIteration:
        raise HTTPException(500, "No audio produced")
    except KeyError:
        raise HTTPException(404, "Voice not found")
    except ValueError as exc:  # unknown stock voice, bad sample_rate/format
        raise HTTPException(400, str(exc))
    except Busy:
        raise HTTPException(503, "TTS capacity exhausted", headers={"Retry-After": "2"})
    except CloneDisabled:
        raise HTTPException(501, "Voice cloning is not enabled on this deployment")


# ------------------------------------------------------------------ voice cloning registry
@router.post("/v1/voices", status_code=201)
async def create_voice(request: Request, file: UploadFile = File(...), name: str = Form(..., max_length=80),
                       speaker_name: str = Form(..., max_length=120),
                       consent_confirmed: bool = Form(...),
                       consent_statement: str = Form(..., min_length=10, max_length=1000),
                       p: Principal = Depends(require_auth)):
    """Register a reference voice. Requires an explicit consent record from the speaker."""
    r = rt(request)
    if not r.clone.enabled:
        raise HTTPException(501, "Voice cloning is not enabled on this deployment")
    if not consent_confirmed:
        raise HTTPException(400, "consent_confirmed must be true: the speaker must have authorised this voice")
    import asyncio
    tmp = r.settings.data_dir / "tmp" / uuid.uuid4().hex
    try:
        await save_upload(file, tmp, 50 << 20)
        pcm = await asyncio.to_thread(decode_file, tmp, 120, 24000)
    except AudioError as exc:
        raise HTTPException(400, str(exc))
    finally:
        tmp.unlink(missing_ok=True)
    try:
        meta = await asyncio.to_thread(r.voices.create, owner=p.key_id, name=name, speaker_name=speaker_name,
                                       consent_statement=consent_statement, pcm24k=pcm)
    except AudioError as exc:
        raise HTTPException(400, str(exc))
    return {"id": meta["id"], "voice": f"clone:{meta['id']}", "name": name, "duration": meta["duration"]}


@router.get("/v1/voices")
async def list_voices(request: Request, p: Principal = Depends(require_auth)):
    return {"voices": rt(request).voices.list(p.key_id)}


@router.delete("/v1/voices/{vid}", status_code=204)
async def delete_voice(request: Request, vid: str, p: Principal = Depends(require_auth)):
    try:
        rt(request).voices.delete(vid, p.key_id)
    except KeyError:
        raise HTTPException(404, "Voice not found")
