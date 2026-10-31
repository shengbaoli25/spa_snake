"""Bounded asynchronous CSV logging; never performs robot I/O.

The control thread only enqueues immutable row copies and checks failures.
The caller must stop motion before finish(), whose wait is strictly bounded.
"""
from __future__ import annotations

import csv
from pathlib import Path
import queue
import threading


class LogQueueFull(RuntimeError):
    pass


class AsyncCsvWriter:
    def __init__(self, path: str | Path, fieldnames: list[str], capacity: int = 256):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("CSV queue capacity must be a positive integer")
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._finish_requested = threading.Event()
        self._finished = threading.Event()
        self._error: Exception | None = None
        self._submission_error: Exception | None = None
        self._submitted = 0
        self._written = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="spa-trial-csv")

    def start(self, timeout_s: float = 1.0):
        self._thread.start()
        if not self._ready.wait(timeout=max(0.0, timeout_s)):
            raise TimeoutError("CSV 日志初始化超时")
        self.check()

    def check(self):
        with self._lock:
            error = self._error or self._submission_error
        if error is not None:
            raise RuntimeError(f"CSV 日志失败: {error}") from error

    def submit(self, row: dict):
        """Never block waiting for disk or queue space; no silent data loss."""
        self.check()
        if self._finish_requested.is_set():
            raise RuntimeError("CSV 日志已停止接收")
        try:
            self._queue.put_nowait(dict(row))
        except queue.Full as exc:
            error = LogQueueFull("CSV 日志队列已满，试验必须停止")
            with self._lock:
                self._submission_error = error
            raise error from exc
        with self._lock:
            self._submitted += 1

    def finish(self, timeout_s: float = 1.0):
        """Drain and close within the caller's bound, or report failure."""
        self._finish_requested.set()
        if not self._finished.wait(timeout=max(0.0, timeout_s)):
            raise TimeoutError("CSV 日志收尾超时，数据未完整保存")
        self.check()
        with self._lock:
            if self._submitted != self._written:
                raise RuntimeError("CSV 日志行数不一致，数据未完整保存")

    def diagnostics(self):
        with self._lock:
            return {"submitted_rows": self._submitted, "written_rows": self._written,
                    "finished": self._finished.is_set(),
                    "error": str(self._error or self._submission_error or "")}

    def _run(self):
        try:
            with self.path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
                writer.writeheader()
                stream.flush()
                self._ready.set()
                while not (self._finish_requested.is_set() and self._queue.empty()):
                    try:
                        row = self._queue.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    writer.writerow(row)
                    stream.flush()
                    with self._lock:
                        self._written += 1
        except Exception as exc:
            with self._lock:
                self._error = exc
        finally:
            self._ready.set()
            self._finished.set()
