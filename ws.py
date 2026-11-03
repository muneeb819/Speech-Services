"""Realtime websockets.

/v1/stream/stt  client -> JSON config, then binary PCM16 mono frames (default 16 kHz); server -> partial/final JSON.
/v1/stream/tts  client -> JSON {"type":"speak","text":..}, {"type":"cancel"}; server -> binary audio + JSON events.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import metrics
from app.audio import AudioError, StreamEncoder, pcm16_to_float, resample
from app.auth import authenticate_ws
from app.engines.base import Busy

log = logging.getLogger("ws")
ws_router = APIRouter()
SR = 16000


async def _open(ws: WebSocket, kind: str):
    p = authenticate_ws(ws)
    rt = ws.app.state.rt
    if p is None or not rt.ready:
        await ws.close(code=4401 if p is None else 1013)
        return None
    if not rt.auth.ws_acquire(p):
        await ws.close(code=4429)
        return None
    await ws.accept()
    metrics.WS_ACTIVE.labels(kind).inc()
    return p, rt


# ---------------------------------------------------------------- live STT
@ws_router.websocket("/v1/stream/stt")
async def stream_stt(ws: WebSocket):
    opened = await _open(ws, "stt")
    if not opened:
        return
    p, rt = opened
    s = rt.settings
    cfg = {"language": s.stt_language, "sample_rate": SR, "endpoint_ms": s.live_endpoint_ms,
           "partials": True, "prompt": None}
    try:
        first = await asyncio.wait_for(ws.receive(), 10)
        if first.get("text"):
            cfg.update({k: v for k, v in json.loads(first["text"]).items() if k in cfg})
            pending = None
        else:
            pending = first
        in_rate = int(cfg["sample_rate"])
        if in_rate not in (8000, 16000, 24000, 48000):
            await ws.send_json({"type": "error", "message": "sample_rate must be 8000/16000/24000/48000"})
            return
        language = None if cfg["language"] == "auto" else cfg["language"]
        endpoint = int(cfg["endpoint_ms"]) * SR // 1000
        await ws.send_json({"type": "ready", "sample_rate": in_rate})

        idle = np.zeros(0, dtype=np.float32)       # rolling pre-roll while waiting for speech
        utt = np.zeros(0, dtype=np.float32)        # current utterance (includes pre-roll)
        in_speech, since_check, total = False, 0, 0
        last_partial = time.monotonic()
        started_at = 0.0
        max_utt = s.live_max_utterance_s * SR

        async def finalize() -> None:
            nonlocal utt, in_speech, idle
            audio, t_start = utt, started_at
            utt, idle, in_speech = np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), False
            if len(audio) < int(0.3 * SR):
                return
            try:
                res = await rt.transcribe(audio, mode="live", language=language, vad_filter=False,
                                          beam_size=3, prompt=cfg["prompt"], batch_size=0)
            except Busy:
                await ws.send_json({"type": "error", "message": "busy"})
                return
            if res["text"]:
                await ws.send_json({"type": "final", "text": res["text"], "start": round(t_start, 2),
                                    "end": round(t_start + len(audio) / SR, 2)})

        msg = pending
        while True:
            if msg is None:
                msg = await asyncio.wait_for(ws.receive(), s.live_idle_timeout_s)
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):
                cmd = json.loads(msg["text"]).get("type")
                if cmd == "flush":
                    await finalize()
                elif cmd == "close":
                    await finalize()
                    break
                msg = None
                continue
            data = msg.get("bytes")
            msg = None
            if not data:
                continue
            x = resample(pcm16_to_float(data), in_rate, SR)
            total += len(x)
            since_check += len(x)
            if in_speech:
                utt = np.concatenate([utt, x])
            else:
                idle = np.concatenate([idle, x])[-int(0.5 * SR):]
            if since_check < int(0.2 * SR):
                continue
            since_check = 0
            if not in_speech:
                sp = rt.vad.spans(idle)
                if sp and sum(e - b for b, e in sp) >= int(0.1 * SR):
                    in_speech, utt = True, idle.copy()
                    started_at = (total - len(utt)) / SR
                    last_partial = time.monotonic()
                continue
            tail = utt[-2 * SR:]
            sp = rt.vad.spans(tail)
            trailing = len(tail) - (sp[-1][1] if sp else 0)
            if trailing >= endpoint or len(utt) >= max_utt:
                await finalize()
            elif (cfg["partials"] and not rt.stt_gate.busy
                  and (time.monotonic() - last_partial) * 1000 >= s.live_partial_interval_ms):
                last_partial = time.monotonic()
                try:
                    res = await rt.transcribe(utt, mode="live", language=language, vad_filter=False,
                                              beam_size=1, prompt=cfg["prompt"], batch_size=0)
                    if res["text"]:
                        await ws.send_json({"type": "partial", "text": res["text"]})
                except Busy:
                    pass
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except (ValueError, KeyError, json.JSONDecodeError, AudioError) as exc:
        try:
            await ws.send_json({"type": "error", "message": f"bad request: {exc}"})
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        log.exception("live stt crashed")
    finally:
        metrics.WS_ACTIVE.labels("stt").dec()
        rt.auth.ws_release(p)
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- streaming TTS with barge-in
@ws_router.websocket("/v1/stream/tts")
async def stream_tts(ws: WebSocket):
    opened = await _open(ws, "tts")
    if not opened:
        return
    p, rt = opened
    s = rt.settings
    cfg = {"voice": s.tts_voice, "speed": 1.0, "format": "pcm", "sample_rate": None}
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    current: dict = {"task": None}

    async def speak(item: dict) -> None:
        sid = item.get("id")
        try:
            enc = StreamEncoder(cfg["format"], 24000, cfg["sample_rate"])
            started = False
            async for chunk, sr in rt.tts_chunks(item["text"], cfg["voice"], float(cfg["speed"]), p.key_id):
                if not started or sr != 24000:
                    enc = StreamEncoder(cfg["format"], sr, cfg["sample_rate"])
                    if not started:
                        await ws.send_json({"type": "start", "id": sid, "sample_rate": enc.out_rate,
                                            "format": cfg["format"]})
                        started = True
                out = enc.feed(chunk)
                if out:
                    await ws.send_bytes(out)
            tail = enc.finish() if started else b""
            if tail:
                await ws.send_bytes(tail)
            await ws.send_json({"type": "done", "id": sid})
        except asyncio.CancelledError:
            raise
        except Busy:
            await ws.send_json({"type": "error", "id": sid, "message": "busy"})
        except (ValueError, KeyError, AudioError) as exc:
            await ws.send_json({"type": "error", "id": sid, "message": str(exc)})

    async def consumer() -> None:
        while True:
            item = await queue.get()
            current["task"] = asyncio.create_task(speak(item))
            try:
                await current["task"]
            except asyncio.CancelledError:
                if not current["task"].cancelled():
                    raise  # consumer itself cancelled
            finally:
                current["task"] = None

    worker = asyncio.create_task(consumer())
    try:
        while True:
            msg = await asyncio.wait_for(ws.receive_text(), s.live_idle_timeout_s)
            req = json.loads(msg)
            t = req.get("type")
            if t == "config":
                cfg.update({k: v for k, v in req.items() if k in cfg})
                await ws.send_json({"type": "configured"})
            elif t == "speak":
                text = str(req.get("text", "")).strip()
                if not text or len(text) > s.max_tts_chars:
                    await ws.send_json({"type": "error", "message": "invalid text length"})
                elif queue.full():
                    await ws.send_json({"type": "error", "message": "too many queued requests"})
                else:
                    queue.put_nowait({"id": req.get("id"), "text": text})
            elif t == "cancel":  # barge-in: drop queued speech and stop the current utterance now
                while not queue.empty():
                    queue.get_nowait()
                if current["task"]:
                    current["task"].cancel()
                await ws.send_json({"type": "cancelled"})
            elif t == "close":
                break
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except (json.JSONDecodeError, AttributeError):
        try:
            await ws.send_json({"type": "error", "message": "messages must be JSON objects"})
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        log.exception("tts stream crashed")
    finally:
        worker.cancel()
        if current["task"]:
            current["task"].cancel()
        metrics.WS_ACTIVE.labels("tts").dec()
        rt.auth.ws_release(p)
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
