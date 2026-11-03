from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("speech_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("speech_http_request_seconds", "HTTP latency", ["route"],
                         buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300))
STT_AUDIO_SECONDS = Counter("speech_stt_audio_seconds_total", "Audio seconds transcribed", ["mode"])
STT_PROC_SECONDS = Histogram("speech_stt_processing_seconds", "STT processing time", ["mode"],
                             buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 300, 900))
TTS_CHARS = Counter("speech_tts_chars_total", "Characters synthesized", ["voice_kind"])
TTS_FIRST_CHUNK = Histogram("speech_tts_first_chunk_seconds", "Time to first audio chunk",
                            buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 6))
WS_ACTIVE = Gauge("speech_ws_active", "Open websockets", ["kind"])
JOBS_QUEUED = Gauge("speech_jobs_queued", "Jobs waiting")
JOBS_TOTAL = Counter("speech_jobs_total", "Jobs finished", ["status"])
BUSY_REJECTIONS = Counter("speech_busy_rejections_total", "Requests shed because GPU slots were full", ["engine"])
