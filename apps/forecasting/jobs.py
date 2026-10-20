"""Launch model runs as background processes and track their status.

Each run executes `runners/run_model.py` with the Python interpreter of the
model's environment, detached from Streamlit so a page refresh doesn't kill
it. The runner reports progress in {JOBS_DIR}/{job_id}.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .catalog import BY_KEY, ENV_ROOT
from .store import JOBS_DIR, RunRequest

APP_ROOT = Path(__file__).resolve().parent.parent
RUNNER = APP_ROOT / "runners" / "run_model.py"


def _python_for(req: RunRequest) -> str:
    if req.synthetic:
        return sys.executable
    py = ENV_ROOT / BY_KEY[req.model].env / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(f"environment for {req.model} not found at {py}")
    return str(py)


def launch(req: RunRequest) -> str:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"{req.model}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    status_path = JOBS_DIR / f"{job_id}.json"
    log_path = JOBS_DIR / f"{job_id}.log"
    out = req.path()
    out.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({"state": "queued", "message": "Starting…",
                                       "started": time.time(), "output": str(out)}))
    cmd = [_python_for(req), str(RUNNER), "--model", req.model,
           "--init", req.init.strftime("%Y-%m-%dT%H"), "--lead-hours", str(req.lead_hours),
           "--members", str(req.members), "--out", str(out), "--status", str(status_path)]
    if req.synthetic:
        cmd.append("--synthetic")
    with open(log_path, "w") as log:
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=APP_ROOT,
                         start_new_session=True)
    return job_id


def status(job_id: str) -> dict:
    try:
        s = json.loads((JOBS_DIR / f"{job_id}.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"state": "unknown", "message": "Status not available yet."}
    s["elapsed_s"] = (s.get("finished") or time.time()) - s.get("started", time.time())
    return s


def log_tail(job_id: str, lines: int = 20) -> str:
    try:
        return "".join((JOBS_DIR / f"{job_id}.log").read_text().splitlines(keepends=True)[-lines:])
    except FileNotFoundError:
        return ""
