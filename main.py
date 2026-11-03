from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__, metrics
from app.audio import AudioError
from app.config import Settings, get_settings
from app.engines.base import Busy
from app.logging_setup import setup_logging
from app.routes import router
from app.runtime import Runtime
from app.ws import ws_router

log = logging.getLogger("api")


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    setup_logging(s.log_level)
    s.validate_for_start()
    for d in ("tmp", "uploads", "results", "voices"):
        (s.data_dir / d).mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rt = rt = Runtime(s)
        await rt.startup()
        yield
        await rt.shutdown()

    app = FastAPI(title="Self-hosted Speech Service", version=__version__, lifespan=lifespan,
                  docs_url="/docs" if s.auth_disabled else None, redoc_url=None,
                  openapi_url="/openapi.json" if s.auth_disabled else None)
    if s.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in s.cors_origins.split(",")],
                           allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            log.exception("unhandled", extra={"request_id": rid, "path": request.url.path})
            response = JSONResponse({"error": {"message": "Internal server error"}}, status_code=500)
        ms = (time.perf_counter() - t0) * 1000
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.HTTP_REQUESTS.labels(request.method, route, response.status_code).inc()
        metrics.HTTP_LATENCY.labels(route).observe(ms / 1000)
        if route not in ("/healthz", "/metrics"):
            log.info("request", extra={"request_id": rid, "path": request.url.path, "status": response.status_code,
                                       "ms": round(ms, 1), "key_id": getattr(request.state, "key_id", None)})
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(AudioError)
    async def audio_error(_, exc: AudioError):
        return JSONResponse({"error": {"message": str(exc), "type": "invalid_audio"}}, status_code=400)

    @app.exception_handler(Busy)
    async def busy(_, exc: Busy):
        return JSONResponse({"error": {"message": "Capacity exhausted, retry shortly", "type": "busy"}},
                            status_code=503, headers={"Retry-After": "2"})

    app.include_router(router)
    app.include_router(ws_router)
    return app


def app_factory() -> FastAPI:  # uvicorn --factory app.main:app_factory
    return create_app()
