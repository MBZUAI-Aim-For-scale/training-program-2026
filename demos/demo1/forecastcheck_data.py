#!/usr/bin/env python3
"""Download this Spark's country data for Demo 5 (ForecastCheck).

  python3 forecastcheck_data.py      download it (the Demo 5 page's button runs this in the background)
  python3 forecastcheck_data.py --country rwanda   another of ForecastCheck's countries (Demo 5's stays as it is)
  python3 forecastcheck_data.py --setup --country rwanda
                                     downloads its data, then runs ForecastCheck's full pipeline (`fcstcheck
                                     pipeline --full`, up to a few hours) and waits for it. The country Demo 5
                                     shows stays as it is (`fcstcheck country`, or the Demo 5 page's picker)

Only for the countries ForecastCheck has (its deployments/: Kenya, Rwanda, Senegal and Sierra
Leone): this Spark's own, from its name (senegal-spark: Senegal), as country_data.py finds it, or
on a Spark whose country ForecastCheck doesn't have, the one chosen for Demo 5 (the Demo 5 page's
picker, or `fcstcheck country <name>`: ForecastCheck's .deployment). It runs ForecastCheck's own `fetch --apply` (demos/demo5/run.sh, the course's
`fcstcheck`), so the data lands in Demo 5's data folder in ForecastCheck's layout, ready for its
pipeline. The country is given to that one command (VERIFICATION_DEPLOYMENT), so the country a
trainee chose with `fcstcheck country` stays as it is. Running it again fetches only what's missing.

Its progress is in bytes: the files the bucket holds for the country (listed first, as the fetch
filters them) against those already in Demo 5's data folder, with the same size.

Standard library only, like spark_setup.py. Progress goes to state/forecastcheck_data.json and
state/logs/forecastcheck_data.log; dashboard.py reads both.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import country_data

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE = HERE / "state"
STATUS = STATE / "forecastcheck_data.json"
LOG = STATE / "logs" / "forecastcheck_data.log"
LOCK = STATE / "forecastcheck_data.lock"
RESULTS = STATE / "forecastcheck_data_results.json"  # each country's latest finished download
DATA = REPO / "demos" / "demo5" / "data"  # Demo 5's data folder: one folder per country (fcstcheck where)
CHECKOUT = REPO / "demos" / "demo5" / "forecastcheck"
RUN = REPO / "demos" / "demo5" / "run.sh"
IMAGE = "forecastcheck"
PIPELINE = "forecastcheck-pipeline"  # ForecastCheck's pipeline container (its docker/run.sh)


def supported() -> list[str]:
    """ForecastCheck's countries (its deployments/*.toml), as it names them: kenya, sierra-leone, ...
    Empty until Demo 5's code is here."""
    return sorted(p.stem for p in (CHECKOUT / "deployments").glob("*.toml"))


def spark_country() -> str | None:
    """This Spark's country as ForecastCheck names it, if ForecastCheck has it."""
    found = country_data.country_from_hostname(country_data.hostname())  # (Sierra_Leone, ...)
    name = found.lower().replace("_", "-") if found else None
    return name if name in supported() else None


def chosen_country() -> str | None:
    """The country chosen for Demo 5 (ForecastCheck's .deployment, which `fcstcheck country` and the
    Demo 5 page's picker set), if ForecastCheck has it."""
    try:
        lines = (CHECKOUT / ".deployment").read_text().splitlines()
    except OSError:
        return None
    name = next((l.strip() for l in lines if l.strip() and not l.strip().startswith("#")), None)
    return name if name in supported() else None


def shown_country() -> str | None:
    """The country Demo 5's dashboard shows: the one chosen for it (.deployment), or before any is chosen,
    this Spark's own (Demo 5's start chooses that, when ForecastCheck has it)."""
    return chosen_country() or spark_country()


def demo_country() -> str | None:
    """The country Demo 5's data is downloaded for: this Spark's own if ForecastCheck has it, else the
    one chosen for Demo 5."""
    return spark_country() or chosen_country()


def choose(name: str) -> tuple[bool, str]:
    """Chooses Demo 5's country, as `fcstcheck country <name>` does (and a running Demo 5 restarts
    with it). (ok, message)"""
    if name not in supported():
        return False, f"ForecastCheck has no {name}"
    try:
        r = subprocess.run(["bash", str(RUN), "country", name], cwd=REPO, capture_output=True, text=True,
                           timeout=300, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    out = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, out[-1] if out else ""


def title(name: str) -> str:
    return name.replace("-", " ").title()


def image_ready() -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# --- What the fetch brings (ForecastCheck's scripts/fetch_country.sh, as it filters the bucket) -------

VARIABLES = {"precipitation": "precipitation", "temperature_2m_max": "temperature_max",
             "temperature_2m_min": "temperature_min"}
MODELS = ("graphcast", "ifs")  # (the fetch's default: the two its pipeline runs)
SHAPES = ("gridded", "station")


def data_source(country: str) -> str:
    """The country's folder in the bucket: its deployments/<country>.toml's data_source."""
    with open(CHECKOUT / "deployments" / f"{country}.toml", "rb") as f:
        return tomllib.load(f)["data_source"]


def local_name(rel: str) -> str | None:
    """Where the fetch puts a file of the country's bucket folder (path inside it), in the country's data
    folder; None for one it doesn't fetch."""
    p = rel.split("/")
    if not rel.endswith(".nc") or ".." in p:
        return None
    if (len(p) == 6 and p[0] == "raw" and p[1] in SHAPES and p[2] in VARIABLES and re.fullmatch(r"\d{4}", p[3])
            and p[4] in MODELS):
        return "/".join(["forecast", p[1], VARIABLES[p[2]], *p[3:]])
    if len(p) >= 5 and p[0] == "target" and p[1] in SHAPES and p[2] in VARIABLES:
        return "/".join(["truth", p[3].lower(), p[1], VARIABLES[p[2]], *p[4:]])
    if rel == "assets/dem.nc":
        return rel
    return None


def plan(country: str, token=None) -> list[tuple[str, int]]:
    """(path in the country's data folder, size) of every file the fetch brings for the country."""
    bucket, _, prefix = data_source(country).removeprefix("gs://").partition("/")
    token = token or country_data.Token()
    files = []
    for o in country_data.list_objects(token, prefix.rstrip("/") + "/", bucket=bucket):
        name = local_name(o["name"][len(prefix.rstrip("/")) + 1:])
        if name:
            files.append((name, int(o["size"])))
    return files


def arrived(country: str, files: list[tuple[str, int]]) -> tuple[int, int]:
    """(files, bytes) of these already in the country's data folder, whole (the same size)."""
    n = b = 0
    for name, size in files:
        try:
            if (DATA / country / name).stat().st_size == size:
                n, b = n + 1, b + size
        except OSError:
            pass
    return n, b


def write_status(**fields):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(fields, indent=1))
    tmp.replace(STATUS)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--country", help="another of ForecastCheck's countries (default: Demo 5's)")
    p.add_argument("--setup", action="store_true",
                   help="download, then run the full pipeline and wait for it (Demo 5 shows what it shows)")
    args = p.parse_args(argv)
    if args.country and args.country not in supported():
        print(f"ForecastCheck has no {args.country!r}; it has: {', '.join(supported())}", file=sys.stderr)
        return 2
    country = args.country or demo_country()
    if not country:
        print(f"ForecastCheck has no data for this Spark ({country_data.hostname()}) and no country was chosen "
              f"for Demo 5 (the Demo 5 page, or fcstcheck country <name>); it has: "
              f"{', '.join(title(c) for c in supported()) or 'nothing yet (set up Demo 5 first)'}",
              file=sys.stderr)
        return 2
    LOG.parent.mkdir(parents=True, exist_ok=True)
    # Held until this process ends, however it ends; a second click can't start a second download
    lock = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("A download is already running.", file=sys.stderr)
        return 1
    job = {"state": "running", "pid": os.getpid(), "country": country, "started": time.time(), "finished": None,
           "detail": "Listing the bucket", "total_files": 0, "total_bytes": 0, "done_files": 0, "done_bytes": 0,
           "setup": args.setup, "phase": "download"}
    saving = threading.Lock()

    def save(**fields):
        with saving:
            job.update(fields)
            write_status(**job)
    save()
    try:
        files = plan(country)
    except (OSError, ValueError, KeyError, country_data.bucket_access.BucketError) as e:
        # (no bucket access, say: the fetch still runs and says why; progress is unknown)
        files = []
        print(f"(progress unknown: couldn't list the bucket: {type(e).__name__}: {e})", flush=True)
    total = sum(size for _, size in files)
    n, b = arrived(country, files)
    save(detail="Fetching", total_files=len(files), total_bytes=total, done_files=n, done_bytes=b)
    finished = threading.Event()

    def watch():  # the bytes in place, every few seconds (gcloud prints no byte counts of its own)
        while not finished.wait(3):
            n, b = arrived(country, files)
            save(done_files=n, done_bytes=b)
    if files:
        threading.Thread(target=watch, daemon=True).start()

    env = {**os.environ, "VERIFICATION_DEPLOYMENT": country}
    last = ""
    with open(LOG, "a") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M')}: {title(country)}: fcstcheck fetch --apply\n")
        log.flush()
        proc = subprocess.Popen(["bash", str(RUN), "fetch", "--apply"], env=env, cwd=REPO, text=True,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                errors="replace", start_new_session=True)

        def stop(*_):  # (Stop on the page: the fetch and its container go too)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        signal.signal(signal.SIGTERM, stop)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.strip().strip("."):
                last = line.strip()
                save(detail=last[:200])
        rc = proc.wait()
    finished.set()
    ok = rc == 0
    n, b = arrived(country, files)
    state = "done" if ok else ("interrupted" if rc < 0 or rc == 143 else "failed")
    if ok and args.setup:
        save(done_files=n, done_bytes=b)
        return pipeline(country, job, save)
    save(state=state, finished=time.time(), done_files=n, done_bytes=b,
         detail=(f"{title(country)}'s data is in Demo 5's data folder" if ok else last[:200] or f"exit {rc}"))
    record(job)
    return 0 if ok else 1


def pipeline_log(country: str) -> Path:
    return DATA / "logs" / f"pipeline-{country}.log"


def pipeline_running() -> bool:
    try:
        return subprocess.run(["docker", "container", "inspect", PIPELINE], capture_output=True,
                              timeout=30, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return True  # (Docker didn't answer: look again)


# The full pipeline's shape (ForecastCheck's scripts/run_pipeline.sh --full): one step per truth product
# and variable on disk (each shape: gridded, and station when there are station forecasts), and in each
# step three parts, every one a line per model, lead day and test year in its output.
PARTS = ("Raw metrics", "Calibration", "Calibrated metrics")
SCORED = re.compile(r"^(\S+) \| lead \d+d \(\d+h\) \|")      # verification.run: model | lead | ...
CALIBRATED = re.compile(r"^  lead \d+d \(\d+h\)( ->|: SKIP)")   # calibration.run(_station): one a lead


def pipeline_plan(country: str) -> dict:
    """{pipeline_steps, pipeline_part_size}: the steps the full pipeline runs for the country's data on
    disk, and the lines each part of a step prints (models x lead days x test years)."""
    try:
        script = (CHECKOUT / "scripts" / "run_pipeline.sh").read_text()
    except OSError:
        script = ""
    slug = country.replace("-", "_")
    studied = next((m.group(2).split() for m in re.finditer(r'^\s*([\w|]+)\)\s*\n\s*all_variables="([^"]*)"',
                                                           script, re.MULTILINE) if slug in m.group(1).split("|")),
                   list(VARIABLES.values()))
    full = re.search(r'YEARS="\$\{YEARS:-([\d,]+)\}"; LEADS="\$\{LEADS:-([\d,]+)\}"', script)
    years, leads = (len(full.group(1).split(",")), len(full.group(2).split(","))) if full else (2, 4)
    data, steps = DATA / country, 0
    for shape in SHAPES:
        if shape == "station" and not (data / "forecast" / "station").is_dir():
            continue
        for product in sorted(p.parent.name for p in data.glob(f"truth/*/{shape}")):
            steps += sum((data / "truth" / product / shape / v).is_dir() for v in studied)
    return {"pipeline_steps": steps, "pipeline_part_size": len(MODELS) * years * leads}


def pipeline_progress(lines: list[str], plan_: dict) -> dict:
    """Where the full pipeline is, from its output: the step, its part, and how far each has got."""
    heads = [i for i, l in enumerate(lines) if l.startswith("──")]
    if not heads:
        return {"pipeline_step": 0, "pipeline_pct": 0}
    size, total = plan_["pipeline_part_size"] or 1, max(plan_["pipeline_steps"], len(heads))
    current = lines[heads[-1] + 1:]
    raw = sum(bool((m := SCORED.match(l)) and not m.group(1).endswith("-idr")) for l in current)
    cal = sum(bool(CALIBRATED.match(l)) for l in current)
    idr = sum(bool((m := SCORED.match(l)) and m.group(1).endswith("-idr")) for l in current)
    part = 2 if idr else 1 if cal or raw >= size else 0
    counts = [min(raw, size), min(cal, size), min(idr, size)]
    done = (len(heads) - 1) * 3 * size + sum(counts)
    return {"pipeline_step": len(heads), "pipeline_steps": total, "pipeline_step_name": lines[heads[-1]][2:].strip(),
            "pipeline_part": part, "pipeline_part_done": counts[part],
            "pipeline_pct": round(100 * done / (total * 3 * size))}


def pipeline(country: str, job: dict, save) -> int:
    """ForecastCheck's full pipeline for the country (`fcstcheck pipeline --full`, in its own container),
    followed until it ends. Stop stops it too. When it ends ForecastCheck restarts Demo 5 to show it."""
    plan_ = pipeline_plan(country)
    save(phase="pipeline", detail="Starting the full pipeline", pipeline_started=time.time(), **plan_)
    log = pipeline_log(country)
    since = log.stat().st_mtime if log.exists() else 0
    r = subprocess.run(["bash", str(RUN), "pipeline", "--full"], cwd=REPO, capture_output=True, text=True,
                       check=False, stdin=subprocess.DEVNULL, env={**os.environ, "VERIFICATION_DEPLOYMENT": country})
    with open(LOG, "a") as out:
        out.write(r.stdout + r.stderr)
    if r.returncode != 0:  # (one already running, say: its message says so)
        lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
        save(state="failed", finished=time.time(), detail=(lines[-1] if lines else f"exit {r.returncode}")[:200])
        record(job)
        return 1

    def stop(*_):
        subprocess.run(["docker", "stop", PIPELINE], capture_output=True, timeout=120, check=False)
        save(state="interrupted", finished=time.time(), detail="The pipeline was stopped")
        record(job)
        sys.exit(1)
    signal.signal(signal.SIGTERM, stop)
    lines: list[str] = []
    while True:
        time.sleep(5)
        try:
            if log.stat().st_mtime >= since:
                lines = [l.strip() for l in log.read_text(errors="replace").splitlines() if l.strip()]
        except OSError:
            pass
        if not pipeline_running():
            break
        save(detail=(lines[-1] if lines else "Starting")[:200], **pipeline_progress(lines, plan_))
    end = next((l for l in reversed(lines) if l.startswith("pipeline finished with exit status")), "")
    ok = end.endswith(" 0")
    save(state="done" if ok else "failed", finished=time.time(),
         detail=(f"{title(country)} is scored" if ok
                 else (lines[-2] if len(lines) > 1 and end else lines[-1] if lines else "the pipeline ended")[:200]))
    record(job)
    return 0 if ok else 1


def record(job: dict):
    """Keep a finished download's outcome by country, so the dashboard can say which are here."""
    results = read_results()
    rec = results.get(job["country"]) or {}
    if job.get("phase") == "pipeline":  # (its data was downloaded in full, before the pipeline)
        rec.update(state="done", pipeline=job.get("state"), pipeline_finished=job.get("finished"))
    else:
        rec.update({k: job.get(k) for k in ("state", "finished", "total_files", "total_bytes")})
    results[job["country"]] = rec
    tmp = RESULTS.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=1))
    tmp.replace(RESULTS)


# --- For the dashboard --------------------------------------------------------------------

def read_results() -> dict:
    try:
        results = json.loads(RESULTS.read_text())
    except (OSError, ValueError):
        return {}
    return results if isinstance(results, dict) else {}


def scored(country: str) -> bool:
    """The pipeline has written scores for the country (what Demo 5's pages read)."""
    return country_data.has_files(DATA / country / "diagnostics") or any(
        (DATA / country / "diagnostics").glob("**/*.parquet"))


def on_this_spark() -> dict[str, str]:
    """ForecastCheck's countries with data here: "scored" (downloaded, and the pipeline has scored it),
    "complete" (its latest download finished) or "partial" (files here, but no finished download)."""
    results, found = read_results(), {}
    latest = read_status()  # (a download from before this record was kept)
    if latest and latest.get("country") and latest["country"] not in results and latest["state"] != "running":
        results[latest["country"]] = latest
    for c in supported():
        if (results.get(c) or {}).get("state") == "done" and (DATA / c).is_dir():
            found[c] = "scored" if scored(c) else "complete"
        elif country_data.has_files(DATA / c):
            found[c] = "partial"
    return found


def read_status() -> dict | None:
    """The latest download's status, with `state` corrected if its process has died."""
    try:
        job = json.loads(STATUS.read_text())
    except (OSError, ValueError):
        return None
    if job["state"] == "running" and not is_ours(job["pid"]):
        job["state"] = "interrupted"
    return job


def is_ours(pid) -> bool:
    """True if this pid is a running forecastcheck_data.py, never some other process that reused the pid."""
    try:
        return "forecastcheck_data.py" in Path(f"/proc/{int(pid)}/cmdline").read_text()
    except (OSError, TypeError, ValueError):
        return False


def stop():
    """Stop a running download. What it fetched stays; the next run fetches only what's missing."""
    job = read_status()
    if job and job["state"] == "running":
        try:
            os.kill(job["pid"], signal.SIGTERM)
        except ProcessLookupError:  # it finished just now
            pass


if __name__ == "__main__":
    sys.exit(main())
