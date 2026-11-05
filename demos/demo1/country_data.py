#!/usr/bin/env python3
"""Download this Spark's country data from the program's Google bucket.

  python3 country_data.py          download it (the dashboard's button runs this in the background)
  python3 country_data.py --plan   show the country, file count and size, and download nothing

The country comes from the Spark's name: kenya-spark -> Kenya, sierra-leone-spark -> Sierra_Leone.
"The data" is the country's 312 files in the bucket layout (see layout_names): CHIRPS v3 rainfall,
CHIRTS temperature and the AI model forecasts, one file a year for 2000-2025. Nothing else in the
bucket is downloaded. A listed file that's missing from the bucket counts as a failed file.

Files keep their bucket paths under ~/<country>_data, such as ~/kenya_data ($SPARK_DATA_DIR to change it). Running
it again resumes: files already there, with the bucket's size and date, are skipped. A file cut off
part-way starts over, since it's only moved into place once it has been downloaded and checked.

Standard library only, like spark_setup.py. Progress goes to state/country_data.json and
state/logs/country_data.log; dashboard.py reads both.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import verify_bucket_access as bucket_access

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
STATUS = STATE / "country_data.json"
LOG = STATE / "logs" / "country_data.log"
LOCK = STATE / "country_data.lock"

API = "https://storage.googleapis.com/storage/v1/b"
BUCKET = bucket_access.BUCKET.removeprefix("gs://")
WORKERS = 8
ATTEMPTS = 3
CHUNK = 1024 * 1024
TOKEN_MAX_AGE = 1800  # Google's tokens last an hour; each request starts with one at most half that old
RESERVE_BYTES = 10 * 10**9  # leave this much disk free after the download

# The bucket layout
COUNTRIES = ("Bangladesh", "Bolivia", "Chile", "Colombia", "Ethiopia", "Indonesia", "Kenya", "Nigeria",
             "Pakistan", "Peru", "Philippines", "Rwanda", "Senegal", "Sierra_Leone")
YEARS = range(2000, 2026)
TEMPERATURES = ("Tmax", "Tmin")
MODELS = ("AIFS_ENS_v2", "AIFS_v2", "graphcast")
FORECAST_VARS = ("tp", "t2m_max", "t2m_min")


class DownloadError(Exception):
    pass


def data_dir(country: str) -> Path:
    return Path(os.environ.get("SPARK_DATA_DIR") or Path.home() / f"{country.lower()}_data").expanduser()


def hostname() -> str:
    return socket.gethostname()


def country_from_hostname(name: str) -> str | None:
    """The one country named in a hostname, ignoring case and punctuation; None if there isn't exactly one."""
    squashed = "".join(c for c in name.split(".")[0].lower() if c.isalnum())
    found = [c for c in COUNTRIES if c.replace("_", "").lower() in squashed]
    return found[0] if len(found) == 1 else None


# --- The bucket ---------------------------------------------------------------------------

class Token:
    """An OAuth token from the Spark's read-only key, renewed before it can expire mid-download."""

    def __init__(self):
        with open(bucket_access.key_path) as f:
            self.key = json.load(f)
        self.lock = threading.Lock()
        self.value, self.taken = None, 0.0

    def get(self) -> str:
        with self.lock:
            if not self.value or time.time() - self.taken > TOKEN_MAX_AGE:
                self.value, self.taken = bucket_access.access_token(self.key), time.time()
            return self.value


def request(token: Token, url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"Authorization": f"Bearer {token.get()}"})


def list_objects(token: Token) -> list[dict]:
    objects, page = [], None
    while True:
        query = {"maxResults": 1000, "fields": "nextPageToken,items(name,size,updated,md5Hash)"}
        if page:
            query["pageToken"] = page
        with urllib.request.urlopen(request(token, f"{API}/{BUCKET}/o?{urllib.parse.urlencode(query)}"),
                                    timeout=60) as r:
            data = json.load(r)
        objects += data.get("items", [])
        page = data.get("nextPageToken")
        if not page:
            return objects


def layout_names(country: str) -> list[str]:
    """Every file the bucket layout lists for a country."""
    names = []
    for year in YEARS:
        names.append(f"CHIRPSv3_Country_0p25deg/{country}/CHIRPSv3_{country}_{year}_box025.nc")
        for t in TEMPERATURES:
            names.append(f"CHIRTS_Country_0p25deg/{t}/{country}/CHIRTS-ERA5_{t}_{country}_{year}_box025.nc")
        for model in MODELS:
            for var in FORECAST_VARS:
                names.append(f"Forecasts_Country_0p25deg/{model}/{var}/{country}/"
                             f"{model}_{var}_{country}_{year}_box025.nc")
    return names


def select(objects: list[dict], country: str) -> list[dict]:
    wanted = set(layout_names(country))
    return [o for o in objects if o["name"] in wanted]


def plan(country: str, token: Token | None = None) -> list[dict]:
    return select(list_objects(token or Token()), country)


# --- Downloading --------------------------------------------------------------------------

def local_path(dest: Path, name: str) -> Path:
    if ".." in name.split("/") or name.startswith("/"):
        raise DownloadError(f"unsafe object name: {name}")
    return dest / name


def bucket_time(obj: dict) -> float:
    return datetime.fromisoformat(obj["updated"]).timestamp()


def up_to_date(path: Path, obj: dict) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    return st.st_size == int(obj["size"]) and int(st.st_mtime) == int(bucket_time(obj))


def fetch(token: Token, obj: dict, path: Path, on_bytes) -> None:
    """Download one object, check it, then move it into place, so `path` only ever holds a complete file."""
    url = f"{API}/{BUCKET}/o/{urllib.parse.quote(obj['name'], safe='')}?alt=media"
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(usedforsecurity=False)
    size = 0
    with urllib.request.urlopen(request(token, url), timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(CHUNK):
            f.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            on_bytes(len(chunk))
    if size != int(obj["size"]):
        raise DownloadError(f"got {size} bytes, expected {obj['size']}")
    # Composite objects have no MD5 in the bucket; those are checked by size alone
    if obj.get("md5Hash") and base64.b64encode(digest.digest()).decode() != obj["md5Hash"]:
        raise DownloadError("MD5 checksum doesn't match the bucket's")
    stamp = bucket_time(obj)
    os.utime(part, (stamp, stamp))
    part.replace(path)


class Job:
    """Progress of one download, saved to STATUS for the dashboard (at most every 2 seconds while busy)."""

    def __init__(self, country: str, dest: Path):
        self.lock = threading.Lock()
        self.saved = 0.0
        self.data = {"pid": os.getpid(), "country": country, "dest": str(dest), "state": "running",
                     "detail": "Listing the bucket", "started": time.time(), "finished": None,
                     "total_files": 0, "total_bytes": 0, "done_files": 0, "done_bytes": 0,
                     "skipped_files": 0, "failed": []}
        self.save()

    def save(self):
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(STATUS)
        self.saved = time.time()

    def update(self, **fields):
        with self.lock:
            self.data.update(fields)
            self.save()

    def add_bytes(self, n: int):
        with self.lock:
            self.data["done_bytes"] += n
            if time.time() - self.saved > 2:
                self.save()

    def file_done(self, skipped: bool = False):
        with self.lock:
            self.data["done_files"] += 1
            self.data["skipped_files"] += skipped
            self.save()

    def file_failed(self, name: str, why: str):
        with self.lock:
            self.data["failed"].append({"name": name, "error": why})
            self.save()


def say(message: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}", flush=True)


def size_text(n: float) -> str:
    return f"{n / 1e9:,.1f} GB" if n >= 1e9 else f"{n / 1e6:,.0f} MB"


def download_one(job: Job, token: Token, obj: dict, dest: Path):
    name, wanted = obj["name"], int(obj["size"])
    try:
        path = local_path(dest, name)
    except DownloadError as e:
        job.file_failed(name, str(e))
        say(f"FAILED {name}: {e}")
        return
    for attempt in range(1, ATTEMPTS + 1):
        got = [0]

        def on_bytes(n):
            got[0] += n
            job.add_bytes(n)

        try:
            fetch(token, obj, path, on_bytes)
            job.file_done()
            say(f"ok     {name} ({size_text(wanted)})")
            return
        except Exception as e:  # network, disk or checksum: try again, then record it and carry on
            job.add_bytes(-got[0])
            error = f"{type(e).__name__}: {e}"
            say(f"retry  {name}, attempt {attempt}/{ATTEMPTS}: {error}")
            if attempt < ATTEMPTS:
                time.sleep(2 ** attempt)
    job.file_failed(name, error)
    say(f"FAILED {name}: {error}")


def download(job: Job, token: Token, country: str, dest: Path, objects: list[dict] | None = None):
    """Fetch the country's files into `dest`; `job` holds the outcome. Raises if it can't get started."""
    missing = []
    if objects is None:
        objects = plan(country, token)
        found = {o["name"] for o in objects}
        missing = [n for n in layout_names(country) if n not in found]
    if not objects:
        raise DownloadError(f"no files for {country} in gs://{BUCKET}")
    for name in missing:
        job.file_failed(name, "not in the bucket")
        say(f"FAILED {name}: not in the bucket")
    for o in objects:
        local_path(dest, o["name"])
    todo = [o for o in objects if not up_to_date(dest / o["name"], o)]
    done = len(objects) - len(todo)
    total = sum(int(o["size"]) for o in objects)
    needed = sum(int(o["size"]) for o in todo)
    say(f"{country}: {len(objects)} files, {size_text(total)}; {done} already here, "
        f"{size_text(needed)} to download into {dest}")

    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if needed + RESERVE_BYTES > free:
        raise DownloadError(f"not enough disk: {size_text(needed)} to download plus {size_text(RESERVE_BYTES)} "
                            f"to keep free, but only {size_text(free)} free under {dest}")

    job.update(detail="Downloading", total_files=len(objects) + len(missing), total_bytes=total, done_files=done,
               done_bytes=total - needed, skipped_files=done)
    with ThreadPoolExecutor(WORKERS) as pool:
        # list() so a bug in a worker is raised here instead of being dropped with its future
        list(pool.map(lambda o: download_one(job, token, o, dest), todo))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", action="store_true", help="show what would be downloaded, and download nothing")
    args = p.parse_args(argv)

    country = country_from_hostname(hostname())
    if not country:
        print(f"No single country in this Spark's name ({hostname()}). Known: {', '.join(COUNTRIES)}",
              file=sys.stderr)
        return 2
    if args.plan:
        files = plan(country)
        print(f"{country}: {len(files)} files, {size_text(sum(int(o['size']) for o in files))} -> {data_dir(country)}")
        return 0

    LOG.parent.mkdir(parents=True, exist_ok=True)
    # Held until this process ends, however it ends; a second click can't start a second download
    lock = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("A download is already running.", file=sys.stderr)
        return 1

    dest = data_dir(country)
    job = Job(country, dest)
    try:
        download(job, Token(), country, dest)
    except Exception as e:
        job.update(state="failed", detail=str(e) or type(e).__name__, finished=time.time())
        say(f"FAILED {e}")
        return 1
    failed = len(job.data["failed"])
    finished = job.data["total_files"] - failed
    summary = (f"{failed} of {job.data['total_files']} files failed; run it again to retry them" if failed
               else f"{finished} files, {size_text(job.data['total_bytes'])} in {dest}")
    job.update(state="failed" if failed else "done", detail=summary, finished=time.time())
    say(summary)
    return 1 if failed else 0


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
    """True if this pid is a running country_data.py, so never some other process that reused the pid."""
    try:
        return "country_data.py" in Path(f"/proc/{int(pid)}/cmdline").read_text()
    except (OSError, TypeError, ValueError):
        return False


def stop():
    """Stop a running download. Finished files stay; the next run skips them."""
    job = read_status()
    if job and job["state"] == "running":
        try:
            os.kill(job["pid"], signal.SIGTERM)
        except ProcessLookupError:  # it finished just now
            pass


if __name__ == "__main__":
    sys.exit(main())
