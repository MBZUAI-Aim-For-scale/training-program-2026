#!/usr/bin/env python3
"""Download this Spark's country data for Demo 5 (ForecastCheck).

  python3 forecastcheck_data.py      download it (the Demo 5 page's button runs this in the background)

Only for the countries ForecastCheck has (its deployments/: Kenya, Rwanda, Senegal and Sierra
Leone): this Spark's own, from its name (senegal-spark: Senegal), as country_data.py finds it, or
on a Spark whose country ForecastCheck doesn't have, the one chosen for Demo 5 (the Demo 5 page's
picker, or `fcstcheck country <name>`: ForecastCheck's .deployment). It runs ForecastCheck's own `fetch --apply` (demos/demo5/run.sh, the course's
`fcstcheck`), so the data lands in Demo 5's data folder in ForecastCheck's layout, ready for its
pipeline. The country is given to that one command (VERIFICATION_DEPLOYMENT), so the country a
trainee chose with `fcstcheck country` stays as it is. Running it again fetches only what's missing.

Standard library only, like spark_setup.py. Progress goes to state/forecastcheck_data.json and
state/logs/forecastcheck_data.log; dashboard.py reads both.
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import country_data

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE = HERE / "state"
STATUS = STATE / "forecastcheck_data.json"
LOG = STATE / "logs" / "forecastcheck_data.log"
LOCK = STATE / "forecastcheck_data.lock"
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


def write_status(**fields):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(fields, indent=1))
    tmp.replace(STATUS)


def main() -> int:
    country = demo_country()
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
    started = time.time()
    write_status(state="running", pid=os.getpid(), country=country, started=started, finished=None,
                 detail="Fetching")
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
            if line.strip():
                last = line.strip()
                write_status(state="running", pid=os.getpid(), country=country, started=started,
                             finished=None, detail=last[:200])
        rc = proc.wait()
    ok = rc == 0
    write_status(state="done" if ok else ("interrupted" if rc < 0 or rc == 143 else "failed"),
                 pid=os.getpid(), country=country, started=started, finished=time.time(),
                 detail=(f"{title(country)}'s data is in Demo 5's data folder" if ok else last[:200] or f"exit {rc}"))
    return 0 if ok else 1


# --- For the dashboard --------------------------------------------------------------------

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
