# Self-hosted Speech Service (STT + TTS + consented voice cloning)

Production wrapper around your starter: **faster-whisper** (STT), **Kokoro** (TTS), **Chatterbox Turbo** (cloning, optional, isolated container).

## Run

```bash
cp .env.example .env            # set SPEECH_API_KEYS and SPEECH_DOMAIN
docker compose --profile tls up -d --build            # API + automatic HTTPS
docker compose --profile tls --profile clone up -d    # + voice cloning (set SPEECH_CLONE_URL/TOKEN first)
SPEECH_URL=https://your-domain SPEECH_KEY=... ./scripts/smoke.sh
```
Requires an NVIDIA GPU + NVIDIA Container Toolkit. First start downloads models into the `model-cache` volume.
Local, GPU-less API exploration: `make dev` (fake engines, docs at `/docs`). Tests: `make test`.

## API (Bearer or `X-API-Key`)

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/transcriptions` | OpenAI-compatible file STT (`json`, `verbose_json`, `text`, `srt`, `vtt`), up to `SPEECH_SYNC_MAX_SECONDS` |
| `POST /v1/jobs/transcriptions`, `GET /v1/jobs/{id}`, `/result`, `DELETE` | Durable queue for long recordings (survives restarts, source audio deleted after processing) |
| `POST /v1/audio/speech` | OpenAI-compatible TTS: `wav pcm ulaw flac mp3 opus`, `sample_rate`, `speed`, `stream:true` (pcm/ulaw) |
| `WS /v1/stream/stt` | Live STT: JSON config, then PCM16 mono frames → `partial` / `final` (VAD endpointing) |
| `WS /v1/stream/tts` | Streaming TTS with barge-in: `{"type":"speak","text":..}`, `{"type":"cancel"}` |
| `POST/GET/DELETE /v1/voices` | Register a cloned voice **with consent record**; use as `voice: "clone:<id>"` |
| `/healthz` `/readyz` `/metrics` | Liveness, readiness (503 until models warm), Prometheus |

Telephony (VICIdial/Asterisk): request `response_format: "ulaw"` (8 kHz G.711) or `pcm` with `sample_rate: 8000`; send live audio to `/v1/stream/stt` with `{"sample_rate": 8000}`.

## What makes it production-grade
- Models load once, warm up, and gate readiness; bounded GPU concurrency with load shedding (503 + `Retry-After`) instead of unbounded queues.
- API-key auth (constant-time, per-key identity), per-key rate limits and websocket caps, upload/duration/text limits, per-owner job and voice isolation.
- Cloning: consent required at registration, consent record + audit log (`clone_audit.jsonl`), Perth watermark left on, deleting a voice deletes the reference audio, cloning runs in a separate container behind a shared secret.
- JSON logs with request IDs, Prometheus metrics, retention janitor, non-root container, healthchecks, TLS via Caddy (`/metrics` and `/readyz` hidden from the internet).

## Operating notes
- **Keep `--workers 1` per container**; scale by adding containers/GPUs behind a load balancer (websockets and jobs are per-instance).
- Raise `SPEECH_STT_CONCURRENCY` / `SPEECH_TTS_CONCURRENCY` only after load testing your GPU.
- Back up the `speech-data` volume if you need job history and voice consent records.
- Pin exact model revisions and `CHATTERBOX_SPEC` for reproducible builds; keep `THIRD_PARTY_NOTICES.md` current.

## Validation status
Verified on CPU with fake engines: all 14 tests (auth, formats, limits, jobs, live-STT endpointing, TTS barge-in, cloning consent/audit/deletion) and a real uvicorn boot. **Not run here: GPU inference, model downloads, the Docker build, Chatterbox.** Validate those on your GPU host (`scripts/smoke.sh`), and load-test before setting concurrency.
