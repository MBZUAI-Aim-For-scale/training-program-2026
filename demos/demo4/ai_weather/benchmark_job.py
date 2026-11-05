"""Session-owned benchmark work with cooperative process cancellation."""

import atexit
import copy
import os
import signal
import threading
import traceback
from collections import deque

_workers = set()
_workers_lock = threading.Lock()
_output_locks = {}
_output_locks_guard = threading.Lock()


def track_worker(pid):
    with _workers_lock:
        _workers.add(pid)


def untrack_worker(pid):
    with _workers_lock:
        _workers.discard(pid)


def _stop_remaining_workers():
    with _workers_lock:
        for pid in _workers:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


atexit.register(_stop_remaining_workers)


class BenchmarkJob:
    def __init__(self, runner, plans, signature, resource_key=None):
        self.runner = runner
        self.plans = copy.deepcopy(plans)
        self.signature = signature
        self.resource_key = resource_key
        self._started = False
        self.cancel_event = threading.Event()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.logs = deque(maxlen=2000)
        self.runs = []
        self.status = "running"

    def report(self, line):
        with self.lock:
            self.logs.append(str(line))

    def start(self):
        with self.lock:
            if self._started:
                return self
            self._started = True
            threading.Thread(target=self._run, daemon=True, name="benchmark-job").start()
        return self

    def cancel(self):
        self.cancel_event.set()

    def snapshot(self):
        with self.lock:
            return {"status": self.status, "logs": list(self.logs), "runs": list(self.runs),
                    "done": self.done.is_set(), "stopping": self.cancel_event.is_set()}

    def _run(self):
        if self.resource_key is None:
            self._run_plans()
            return
        with _output_locks_guard:
            output_lock = _output_locks.setdefault(self.resource_key, threading.Lock())
        acquired = output_lock.acquire(blocking=False)
        if not acquired:
            self.report("Waiting for another task writing the same output directory…")
        while not acquired and not self.cancel_event.is_set():
            acquired = output_lock.acquire(timeout=0.1)
        try:
            self._run_plans()
        finally:
            if acquired:
                output_lock.release()

    def _run_plans(self):
        try:
            for plan in self.plans:
                if self.cancel_event.is_set():
                    break
                result = self.runner(plan["mode"], plan["model_key"], plan["config"],
                                     progress=self.report, cancel_event=self.cancel_event)
                if self.cancel_event.is_set():
                    result["cancelled"] = True
                result["model"] = plan["label"]
                with self.lock:
                    self.runs.append(result)
            with self.lock:
                self.status = ("cancelled" if self.cancel_event.is_set() else
                               "completed" if self.runs and all(r.get("ok") for r in self.runs) else "failed")
        except Exception:  # noqa: BLE001 - thread boundary reports failures to the UI.
            with self.lock:
                self.runs.append({"ok": False, "returncode": 1, "stderr": traceback.format_exc()})
                self.status = "cancelled" if self.cancel_event.is_set() else "failed"
        finally:
            self.done.set()
