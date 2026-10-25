"""Run one model and write its output in the shared format (see demo3/contract.py).

Invoked by the app (demo3/jobs.py) with the interpreter of the model's own
environment:

    python runners/run_model.py --model graphcast --init 2025-07-01T00 \
        --lead-hours 240 --members 1 --out out.nc --status job.json

A real model is added by creating runners/runner_<model_key>.py with

    def run(init: pd.Timestamp, lead_hours: int, members: int, report) -> xr.Dataset

returning tp [mm per 6 h], t2m [K], z500 [m2 s-2] on dims
(ensemble, lead_time[h], lat, lon). `report(message, fraction)` updates the
progress shown in the app.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

# JAX (GraphCast, NeuralGCM, FGN) grabs 75% of GPU memory up front by default. On the
# Spark that memory is shared with the CPU, so the rest of the machine starts swapping.
# The largest JAX model needs ~18 GB; 35% (~42 GB) leaves room for everything else.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "runners"))

from demo3 import contract  # noqa: E402
from demo3.catalog import BY_KEY  # noqa: E402


class Reporter:
    def __init__(self, path: Path):
        self.path = path
        self.state = json.loads(path.read_text()) if path.exists() else {"started": time.time()}

    def __call__(self, message: str, fraction: float | None = None, **extra):
        self.state.update(message=message, state=extra.pop("state", "running"), **extra)
        if fraction is not None:
            self.state["progress"] = max(0.0, min(1.0, fraction))
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        tmp.replace(self.path)
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(BY_KEY))
    ap.add_argument("--init", required=True, help="YYYY-MM-DDTHH (UTC)")
    ap.add_argument("--lead-hours", type=int, required=True)
    ap.add_argument("--members", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--status", required=True, type=Path)
    a = ap.parse_args()

    report = Reporter(a.status)
    model = BY_KEY[a.model]
    init = pd.Timestamp(a.init)
    t0 = time.time()
    try:
        report(f"Loading {model.name}…", 0.0)
        try:
            runner = importlib.import_module(f"runner_{a.model}")
        except ModuleNotFoundError:
            raise NotImplementedError(f"No runner yet for {model.name} (runners/runner_{a.model}.py)")
        ds = runner.run(init, a.lead_hours, a.members, report)
        t_post = time.time()
        report("Post-processing and saving…", 0.97)
        source = ds.attrs.get("fgn_init_source") or model.init_source  # FGN: sample or converter
        ds = contract.finalize(ds, model_key=a.model, model_name=model.name, init=init,
                               init_source=source)
        contract.write(ds, a.out)
        post_s = round(time.time() - t_post, 1)
        report("Done", 1.0, state="done", finished=time.time(), runtime_s=round(time.time() - t0, 1),
               post_s=post_s)
        # how this run was made, shown next to the results (also for saved runs)
        stages = {k: report.state.get(k) for k in ("load_s", "fetch_s", "run_s", "post_s",
                                                    "runtime_s", "peak_gpu_gb")}
        stages.update(model=a.model, init=a.init, lead_hours=a.lead_hours, members=a.members,
                      output_mb=round(a.out.stat().st_size / 1e6, 1),
                      init_source=source,
                      when=time.strftime("%Y-%m-%dT%H:%M"))
        a.out.with_suffix(".json").write_text(json.dumps(stages, indent=1))
        # measured Spark runtimes, source for timings.json
        rec = {k: report.state.get(k) for k in ("load_s", "fetch_s", "run_s", "post_s",
                                                "runtime_s", "peak_gpu_gb")}
        rec.update(model=a.model, init=a.init, lead_hours=a.lead_hours, members=a.members,
                   output_mb=round(a.out.stat().st_size / 1e6, 1), when=time.strftime("%Y-%m-%dT%H:%M"))
        with open(a.out.parent.parent / "_timings_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return 0
    except Exception as e:  # surfaced in the app
        traceback.print_exc()
        report(f"{type(e).__name__}: {e}", None, state="failed", finished=time.time())
        return 1


if __name__ == "__main__":
    sys.exit(main())
