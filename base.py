from __future__ import annotations

import asyncio
import threading
from typing import Callable, Iterator, Protocol

import numpy as np


class STTEngine(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, *, language: str | None, word_timestamps: bool, vad: bool,
                   beam_size: int, batch_size: int, prompt: str | None) -> dict: ...


class TTSEngine(Protocol):
    name: str
    sample_rate: int

    def voices(self) -> list[str]: ...
    def stream(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]: ...


class Busy(Exception):
    """No inference slot became free in time."""


class Gate:
    """Bounded concurrency in front of a GPU model, with a wait timeout (load shedding)."""

    def __init__(self, n: int, timeout: float):
        self._sem = asyncio.Semaphore(n)
        self._timeout = timeout
        self.waiting = 0

    @property
    def busy(self) -> bool:
        return self._sem.locked()

    async def __aenter__(self) -> "Gate":
        self.waiting += 1
        try:
            await asyncio.wait_for(self._sem.acquire(), self._timeout)
        except asyncio.TimeoutError as exc:
            raise Busy() from exc
        finally:
            self.waiting -= 1
        return self

    async def __aexit__(self, *exc) -> None:
        self._sem.release()


_DONE = object()


async def aiter_blocking(factory: Callable[[], Iterator]):
    """Run a blocking generator in a worker thread and yield its items asynchronously.
    Closing/cancelling the consumer stops the producer after the current chunk."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def worker() -> None:
        try:
            for item in factory():
                if stop.is_set():
                    break
                loop.call_soon_threadsafe(q.put_nowait, item)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            loop.call_soon_threadsafe(q.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _DONE)

    fut = loop.run_in_executor(None, worker)
    try:
        while True:
            item = await q.get()
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        await asyncio.shield(fut)
