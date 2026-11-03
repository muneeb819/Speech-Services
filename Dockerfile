# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 10001 app && mkdir -p /data /models && chown app:app /data /models
WORKDIR /srv

# ---- main API: faster-whisper + Kokoro ----
FROM base AS api
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app
COPY deploy/entrypoint.sh /entrypoint.sh
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 CMD curl -fs http://localhost:8000/healthz || exit 1
ENTRYPOINT ["/entrypoint.sh"]
# One worker on purpose: models and concurrency gates live in-process. Scale with more containers/GPUs.
CMD ["uvicorn", "app.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30"]

# ---- optional voice-cloning worker: Chatterbox Turbo (isolated dependency stack) ----
FROM base AS clone
ARG CHATTERBOX_SPEC="chatterbox-tts"
RUN pip install "${CHATTERBOX_SPEC}" fastapi==0.115.6 "uvicorn[standard]==0.34.0" soundfile
COPY clone_service ./clone_service
USER app
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 CMD curl -fs http://localhost:8100/healthz || exit 1
CMD ["uvicorn", "clone_service.app:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
