"""Small session-owned tasks; workers never access Streamlit or session state."""

import copy
import threading
import traceback


class BackgroundTask:
    def __init__(self, function, *args, **kwargs):
        self._function = function
        self._args = copy.deepcopy(args)
        self._kwargs = copy.deepcopy(kwargs)
        self._lock = threading.Lock()
        self._started = False
        self.done = threading.Event()
        self._result = None
        self._error = None

    def start(self):
        # A rerun or repeated submit must never start this task twice.
        with self._lock:
            if self._started:
                return self
            self._started = True
            threading.Thread(target=self._run, daemon=True, name="dashboard-task").start()
        return self

    def _run(self):
        try:
            result = self._function(*self._args, **self._kwargs)
            with self._lock:
                self._result = result
        except Exception:
            with self._lock:
                self._error = traceback.format_exc()
        finally:
            self.done.set()

    def snapshot(self):
        with self._lock:
            done = self.done.is_set()
            return {"done": done, "result": self._result, "error": self._error,
                    "status": "failed" if done and self._error else "completed" if done else "running"}
