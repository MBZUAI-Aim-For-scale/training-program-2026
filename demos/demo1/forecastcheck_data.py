#!/usr/bin/env python3
"""Download this Spark's country data for Demo 5 (ForecastCheck).

  python3 forecastcheck_data.py      download it (the Demo 5 page's button runs this in the background)
  python3 forecastcheck_data.py --country rwanda   another of ForecastCheck's countries (Demo 5's stays as it is)

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
           "detail": "Listing the bucket", "total_files": 0, "total_bytes": 0, "done_files": 0, "done_bytes": 0}
    saving = threading.Lock()

    def save(**fields):
        with saving:
            job.update(fields)
            write_status(**job)
    save()
    try:
        files = plan(country)
    except Exception as e:  # (no bucket access, say: the fetch still runs and says why; progress is unknown)
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
    save(state="done" if ok else ("interrupted" if rc < 0 or rc == 143 else "failed"), finished=time.time(),
         done_files=n, done_bytes=b,
         detail=(f"{title(country)}'s data is in Demo 5's data folder" if ok else last[:200] or f"exit {rc}"))
    record(job)
    return 0 if ok else 1


def record(job: dict):
    """Keep a finished download's outcome by country, so the dashboard can say which are here."""
    results = read_results()
    results[job["country"]] = {k: job.get(k) for k in ("state", "finished", "total_files", "total_bytes")}
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


def on_this_spark() -> dict[str, str]:
    """ForecastCheck's countries with data here: "complete" (its latest download finished) or "partial"
    (files here, but no finished download)."""
    results, found = read_results(), {}
    latest = read_status()  # (a download from before this record was kept)
    if latest and latest.get("country") and latest["country"] not in results and latest["state"] != "running":
        results[latest["country"]] = latest
    for c in supported():
        if (results.get(c) or {}).get("state") == "done" and (DATA / c).is_dir():
            found[c] = "complete"
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
