from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from app import metrics
from app.audio import AudioError, decode_file

log = logging.getLogger("jobs")


class JobStore:
    """Tiny durable job table (SQLite, WAL). Survives restarts; queued/running jobs are re-queued."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
                updated REAL NOT NULL, filename TEXT, params TEXT NOT NULL, error TEXT)""")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def create(self, owner: str, filename: str, params: dict) -> str:
        jid, now = uuid.uuid4().hex, time.time()
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,NULL)",
                      (jid, owner, "queued", now, now, filename, json.dumps(params)))
        return jid

    def get(self, jid: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return dict(r) if r else None

    def set_status(self, jid: str, status: str, error: str | None = None) -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE jobs SET status=?, updated=?, error=? WHERE id=?", (status, time.time(), error, jid))

    def incomplete(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY created")]

    def expired(self, older_than: float) -> list[str]:
        with self._conn() as c:
            return [r["id"] for r in c.execute(
                "SELECT id FROM jobs WHERE updated<? AND status NOT IN ('queued','running')", (older_than,))]

    def delete(self, jid: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM jobs WHERE id=?", (jid,))


class JobManager:
    def __init__(self, rt):
        self.rt = rt
        s = rt.settings
        self.store = JobStore(s.data_dir / "jobs.sqlite3")
        self.upload_dir = s.data_dir / "uploads"
        self.result_dir = s.data_dir / "results"
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=s.job_queue_max)
        self._tasks: list[asyncio.Task] = []

    def upload_path(self, jid: str) -> Path:
        return self.upload_dir / jid

    def result_path(self, jid: str) -> Path:
        return self.result_dir / f"{jid}.json"

    async def start(self) -> None:
        for job in await asyncio.to_thread(self.store.incomplete):
            if self.upload_path(job["id"]).exists():
                self.queue.put_nowait(job["id"])
                log.info("requeued job after restart", extra={"event": "job_requeue"})
            else:
                self.store.set_status(job["id"], "failed", "upload lost during restart")
        metrics.JOBS_QUEUED.set(self.queue.qsize())
        self._tasks = [asyncio.create_task(self._worker()) for _ in range(self.rt.settings.job_workers)]
        self._tasks.append(asyncio.create_task(self._janitor()))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def submit(self, jid: str) -> bool:
        try:
            self.queue.put_nowait(jid)
        except asyncio.QueueFull:
            return False
        metrics.JOBS_QUEUED.set(self.queue.qsize())
        return True

    async def _worker(self) -> None:
        s = self.rt.settings
        while True:
            jid = await self.queue.get()
            metrics.JOBS_QUEUED.set(self.queue.qsize())
            job = await asyncio.to_thread(self.store.get, jid)
            if not job:
                continue
            await asyncio.to_thread(self.store.set_status, jid, "running")
            try:
                p = json.loads(job["params"])
                audio = await asyncio.to_thread(decode_file, self.upload_path(jid), s.max_audio_seconds)
                result = await self.rt.transcribe(audio, mode="job", wait_forever=True, **p)
                await asyncio.to_thread(self.result_path(jid).write_text, json.dumps(result, ensure_ascii=False))
                await asyncio.to_thread(self.store.set_status, jid, "completed")
                metrics.JOBS_TOTAL.labels("completed").inc()
            except AudioError as exc:
                await asyncio.to_thread(self.store.set_status, jid, "failed", str(exc))
                metrics.JOBS_TOTAL.labels("failed").inc()
            except Exception as exc:  # noqa: BLE001
                log.exception("job failed", extra={"event": "job_failed"})
                await asyncio.to_thread(self.store.set_status, jid, "failed", "internal error")
                metrics.JOBS_TOTAL.labels("failed").inc()
                _ = exc
            finally:
                self.upload_path(jid).unlink(missing_ok=True)  # source audio is not retained

    async def _janitor(self) -> None:
        while True:
            await asyncio.sleep(600)
            try:
                cutoff = time.time() - self.rt.settings.retention_hours * 3600
                for jid in await asyncio.to_thread(self.store.expired, cutoff):
                    self.result_path(jid).unlink(missing_ok=True)
                    await asyncio.to_thread(self.store.delete, jid)
            except Exception:  # noqa: BLE001
                log.exception("janitor error")
