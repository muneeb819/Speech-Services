from __future__ import annotations


def _ts(sec: float, comma: bool) -> str:
    ms = max(0, round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{',' if comma else '.'}{ms:03}"


def to_srt(result: dict) -> str:
    return "\n".join(f"{i}\n{_ts(s['start'], True)} --> {_ts(s['end'], True)}\n{s['text']}\n"
                     for i, s in enumerate(result["segments"], 1))


def to_vtt(result: dict) -> str:
    body = "\n".join(f"{_ts(s['start'], False)} --> {_ts(s['end'], False)}\n{s['text']}\n" for s in result["segments"])
    return "WEBVTT\n\n" + body


def to_text(result: dict) -> str:
    return "\n".join(s["text"] for s in result["segments"]) + "\n"
