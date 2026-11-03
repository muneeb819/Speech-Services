from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict
from dataclasses import dataclass

from fastapi import HTTPException, Request, WebSocket

from app.config import Settings


@dataclass(frozen=True)
class Principal:
    key_id: str


class Authenticator:
    def __init__(self, s: Settings):
        self.disabled = s.auth_disabled
        self._keys: list[tuple[str, bytes]] = []
        for raw in filter(None, (x.strip() for x in s.api_keys.split(","))):
            name, _, secret = raw.partition(":")
            if not secret:
                name, secret = hashlib.sha256(raw.encode()).hexdigest()[:8], raw
            self._keys.append((name, hashlib.sha256(secret.encode()).digest()))
        self._rate = s.rate_limit_per_min
        self._buckets: dict[str, tuple[float, float]] = defaultdict(lambda: (float(self._rate), time.monotonic()))
        self._ws_open: dict[str, int] = defaultdict(int)
        self._ws_max = s.max_ws_per_key

    def verify(self, token: str | None) -> Principal | None:
        if self.disabled:
            return Principal("anonymous")
        if not token:
            return None
        digest = hashlib.sha256(token.encode()).digest()
        found = None
        for name, d in self._keys:  # check all keys: constant work per attempt
            if hmac.compare_digest(d, digest):
                found = name
        return Principal(found) if found else None

    def check_rate(self, p: Principal) -> bool:
        tokens, last = self._buckets[p.key_id]
        now = time.monotonic()
        tokens = min(float(self._rate), tokens + (now - last) * self._rate / 60.0)
        if tokens < 1:
            self._buckets[p.key_id] = (tokens, now)
            return False
        self._buckets[p.key_id] = (tokens - 1, now)
        return True

    def ws_acquire(self, p: Principal) -> bool:
        if self._ws_open[p.key_id] >= self._ws_max:
            return False
        self._ws_open[p.key_id] += 1
        return True

    def ws_release(self, p: Principal) -> None:
        self._ws_open[p.key_id] = max(0, self._ws_open[p.key_id] - 1)


def _token_from_headers(headers) -> str | None:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return headers.get("x-api-key")


async def require_auth(request: Request) -> Principal:
    a: Authenticator = request.app.state.rt.auth
    p = a.verify(_token_from_headers(request.headers))
    if p is None:
        raise HTTPException(401, "Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})
    if not a.check_rate(p):
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "5"})
    request.state.key_id = p.key_id
    return p


def authenticate_ws(ws: WebSocket) -> Principal | None:
    """Headers (Authorization / X-API-Key) or ?token= for browser clients."""
    a: Authenticator = ws.app.state.rt.auth
    return a.verify(_token_from_headers(ws.headers) or ws.query_params.get("token"))
