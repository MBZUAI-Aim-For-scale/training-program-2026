"""Keep benchmark handles across browser sessions in one dashboard server."""

import secrets
import threading
import time


_jobs = {}
_lock = threading.Lock()
_retention_seconds = 24 * 60 * 60


def _prune(now):
    for token, (job, accessed) in list(_jobs.items()):
        if job.done.is_set() and now - accessed > _retention_seconds:
            del _jobs[token]


def register_benchmark_job(job):
    """Return an unguessable reconnect token; do not start or restart the job."""
    with _lock:
        now = time.monotonic()
        _prune(now)
        token = secrets.token_urlsafe(32)
        _jobs[token] = (job, now)
        return token


def find_benchmark_job(token):
    with _lock:
        now = time.monotonic()
        _prune(now)
        entry = _jobs.get(token)
        if entry is None:
            return None
        job, _ = entry
        _jobs[token] = (job, now)
        return job
