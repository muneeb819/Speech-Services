import json
import time

import numpy as np

from tests.conftest import H, KEY, wav_bytes


def test_health_and_ready(client):
    assert client.get("/healthz").status_code == 200
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert b"speech_http_requests_total" in client.get("/metrics").content


def test_auth_required(client):
    assert client.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401
    assert client.post("/v1/audio/speech", json={"input": "hi"},
                       headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/v1/audio/speech", json={"input": "hi"}, headers={"X-API-Key": KEY}).status_code == 200


def test_transcription_formats(client):
    f = {"file": ("a.wav", wav_bytes(2), "audio/wav")}
    r = client.post("/v1/audio/transcriptions", files=f, headers=H)
    assert r.status_code == 200 and "fake transcript" in r.json()["text"]
    r = client.post("/v1/audio/transcriptions", files=f, data={"response_format": "srt"}, headers=H)
    assert "00:00:00,000 -->" in r.text
    r = client.post("/v1/audio/transcriptions", files=f, data={"response_format": "verbose_json"}, headers=H)
    assert r.json()["segments"][0]["words"]
    r = client.post("/v1/audio/transcriptions", files=f, data={"response_format": "bogus"}, headers=H)
    assert r.status_code == 400


def test_transcription_limits_and_bad_audio(client):
    long_ = {"file": ("a.wav", wav_bytes(8), "audio/wav")}  # > sync_max_seconds=5
    assert client.post("/v1/audio/transcriptions", files=long_, headers=H).status_code == 413
    junk = {"file": ("a.wav", b"not audio at all" * 10, "audio/wav")}
    assert client.post("/v1/audio/transcriptions", files=junk, headers=H).status_code == 400
    big = {"file": ("a.bin", b"\0" * (2 << 20), "application/octet-stream")}  # > 1 MB
    assert client.post("/v1/audio/transcriptions", files=big, headers=H).status_code == 413


def test_speech_formats(client):
    base = {"input": "Hello there. How are you."}
    r = client.post("/v1/audio/speech", json=base, headers=H)
    assert r.content[:4] == b"RIFF"
    r = client.post("/v1/audio/speech", json={**base, "response_format": "ulaw"}, headers=H)
    assert r.status_code == 200 and len(r.content) == 8000 * 0.4  # 2 sentences x 0.2s @ 8 kHz, 1 byte/sample
    r = client.post("/v1/audio/speech", json={**base, "response_format": "pcm", "sample_rate": 16000}, headers=H)
    assert len(r.content) == 16000 * 0.4 * 2
    r = client.post("/v1/audio/speech", json={**base, "response_format": "mp3"}, headers=H)
    assert r.status_code == 200 and len(r.content) > 100
    assert client.post("/v1/audio/speech", json={**base, "voice": "zz_nobody"}, headers=H).status_code == 400
    assert client.post("/v1/audio/speech", json={**base, "sample_rate": 12345}, headers=H).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": ""}, headers=H).status_code == 422


def test_speech_streaming(client):
    r = client.post("/v1/audio/speech", headers=H,
                    json={"input": "One. Two. Three.", "stream": True, "response_format": "ulaw"})
    assert r.status_code == 200 and r.headers["x-sample-rate"] == "8000"
    assert abs(len(r.content) - 8000 * 0.6) <= 8
    bad = client.post("/v1/audio/speech", headers=H, json={"input": "x.", "stream": True, "response_format": "wav"})
    assert bad.status_code == 400
    bad = client.post("/v1/audio/speech", headers=H, json={"input": "x.", "stream": True, "voice": "zz_no"})
    assert bad.status_code == 400


def test_ulaw_encoder_known_values():
    from app.audio import to_ulaw
    out = to_ulaw(np.array([0, 32767, -32768, 1000], dtype="<i2"))
    assert out[0] == 0xFF and out[1] == 0x80 and out[2] == 0x00
    assert out[3] == 0xCE  # mu-law of +1000


def test_jobs_flow_and_isolation(client):
    f = {"file": ("long.wav", wav_bytes(8), "audio/wav")}
    r = client.post("/v1/jobs/transcriptions", files=f, headers=H)
    assert r.status_code == 202
    jid = r.json()["id"]
    for _ in range(50):
        st = client.get(f"/v1/jobs/{jid}", headers=H).json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert st["status"] == "completed", st
    res = client.get(f"/v1/jobs/{jid}/result", headers=H).json()
    assert res["duration"] == 8.0
    assert "-->" in client.get(f"/v1/jobs/{jid}/result?format=vtt", headers=H).text
    other = {"Authorization": "Bearer other-secret"}
    assert client.get(f"/v1/jobs/{jid}", headers=other).status_code == 404
    assert client.delete(f"/v1/jobs/{jid}", headers=H).status_code == 204
    assert client.get(f"/v1/jobs/{jid}", headers=H).status_code == 404


def test_job_bad_audio_fails_cleanly(client):
    r = client.post("/v1/jobs/transcriptions", files={"file": ("x.wav", b"garbage" * 50, "audio/wav")}, headers=H)
    jid = r.json()["id"]
    for _ in range(50):
        st = client.get(f"/v1/jobs/{jid}", headers=H).json()
        if st["status"] != "queued" and st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "failed" and "decode" in st["error"].lower()


def test_ws_tts_and_barge_in(client):
    with client.websocket_connect("/v1/stream/tts", headers=H) as ws:
        ws.send_json({"type": "config", "format": "ulaw"})
        assert ws.receive_json()["type"] == "configured"
        ws.send_json({"type": "speak", "id": "a", "text": "One. Two."})
        assert ws.receive_json()["type"] == "start"
        got = 0
        while True:
            m = ws.receive()
            if m.get("bytes"):
                got += len(m["bytes"])
            else:
                assert json.loads(m["text"])["type"] == "done"
                break
        assert got == 3200
        ws.send_json({"type": "speak", "id": "b", "text": "Again. And again. And again."})
        ws.send_json({"type": "cancel"})
        seen = set()
        while "cancelled" not in seen:
            m = ws.receive()
            if m.get("text"):
                seen.add(json.loads(m["text"])["type"])


def test_ws_requires_auth(client):
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/stream/stt") as ws:
            ws.receive_json()


def test_ws_live_stt_endpointing(client):
    sr = 16000
    tone = (0.3 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr) * 32767).astype("<i2")
    silence = np.zeros(sr, dtype="<i2")
    with client.websocket_connect("/v1/stream/stt", headers=H) as ws:
        ws.send_json({"sample_rate": 16000, "partials": False})
        assert ws.receive_json()["type"] == "ready"
        for chunk in np.split(np.concatenate([silence[:4000], tone, silence]), 24)[:]:
            ws.send_bytes(chunk.tobytes())
        msg = ws.receive_json()
        assert msg["type"] == "final" and "fake transcript" in msg["text"]


def test_voice_cloning_disabled_by_default(client):
    r = client.post("/v1/audio/speech", json={"input": "hi", "voice": "clone:" + "a" * 32}, headers=H)
    assert r.status_code == 501


def _enable_clone(client, tmp_path):
    import httpx
    import os
    os.environ.update({"CLONE_TOKEN": "tok", "CLONE_FAKE": "1", "CLONE_DATA_DIR": str(tmp_path)})
    import importlib
    import clone_service.app as ca
    importlib.reload(ca)
    rt = client.app.state.rt
    rt.clone.enabled = True
    rt.clone._c = httpx.AsyncClient(transport=httpx.ASGITransport(app=ca.app), base_url="http://clone",
                                    headers={"X-Clone-Token": "tok"})


def test_clone_requires_consent_and_audits(client, tmp_path):
    _enable_clone(client, tmp_path)
    ref = {"file": ("ref.wav", wav_bytes(8, sr=24000), "audio/wav")}
    form = {"name": "demo", "speaker_name": "Jane Doe", "consent_statement": "I, Jane Doe, authorise this use.",
            "consent_confirmed": "false"}
    assert client.post("/v1/voices", files=ref, data=form, headers=H).status_code == 400
    short = {"file": ("ref.wav", wav_bytes(2, sr=24000), "audio/wav")}
    assert client.post("/v1/voices", files=short, data={**form, "consent_confirmed": "true"}, headers=H).status_code == 400
    r = client.post("/v1/voices", files=ref, data={**form, "consent_confirmed": "true"}, headers=H)
    assert r.status_code == 201
    voice = r.json()["voice"]
    assert len(client.get("/v1/voices", headers=H).json()["voices"]) == 1
    other = {"Authorization": "Bearer other-secret"}
    assert client.get("/v1/voices", headers=other).json()["voices"] == []
    assert client.post("/v1/audio/speech", json={"input": "hello", "voice": voice}, headers=other).status_code == 404
    out = client.post("/v1/audio/speech", json={"input": "hello", "voice": voice, "response_format": "ulaw"}, headers=H)
    assert out.status_code == 200 and len(out.content) > 1000
    audit = (tmp_path / "clone_audit.jsonl").read_text()
    assert "voice_created" in audit and "synthesis" in audit
    vid = voice[6:]
    assert client.delete(f"/v1/voices/{vid}", headers=H).status_code == 204
    assert client.post("/v1/audio/speech", json={"input": "hello", "voice": voice}, headers=H).status_code == 404
