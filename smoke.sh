#!/bin/sh
# Usage: SPEECH_URL=https://speech.example.com SPEECH_KEY=... ./scripts/smoke.sh [audio-file]
set -eu
URL="${SPEECH_URL:-http://127.0.0.1:8000}"; KEY="${SPEECH_KEY:-}"
AUTH=""; [ -n "$KEY" ] && AUTH="Authorization: Bearer $KEY"
curl -fsS "$URL/readyz"; echo
curl -fsS -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"input":"Smoke test of the speech service.","response_format":"wav"}' "$URL/v1/audio/speech" -o /tmp/smoke.wav
echo "TTS ok: $(wc -c </tmp/smoke.wav) bytes"
curl -fsS -H "$AUTH" -F "file=@${1:-/tmp/smoke.wav}" -F response_format=text "$URL/v1/audio/transcriptions"
