#!/usr/bin/env python3
"""Make this Spark's country data into AI Almanac datasets, and register them (with the country's region).

  spark-setup register-ai-almanac            this Spark's country (from its name)
  spark-setup register-ai-almanac --country Kenya   another country whose data is on this Spark
  spark-setup register-ai-almanac --dry-run  show what it would do, change nothing
  (python3 ai_almanac_data.py ... is the same)

AI Almanac (~/ai-almanac) benchmarks rainfall forecasts against observations, so this takes the
country data's rainfall: CHIRPS v3 observations and the AI models' total precipitation (tp). It
lays them out the way AI Almanac's discovery reads a dataset folder, in AI Almanac's dataset folder
~/ai-almanac-data ($ALMANAC_DATASETS_DIR; /datasets in its container):

  ~/ai-almanac-data/<country>/obs/CHIRPSv3_<Country>_<YYYY>_box025.nc
  ~/ai-almanac-data/<country>/models/<Model>/<Model>_tp_<Country>_<YYYY>_box025.nc

as hard links to the files in ~/<country>_data (no extra space; a copy where a hard link can't be
made). The fleet mirrors ~/ai-almanac-data from the program's bucket and deletes what the bucket
doesn't have, except in a folder holding .fleet-demos-keep: each country folder gets one.

Then, in AI Almanac's own container (the running one, or a one-off one if it isn't running, so AI
Almanac needn't be open), with AI Almanac's own code:
  - the country's region, its bounds the data's lat/lon box, read as AI Almanac reads a source's
    (nothing changes if a region with that name is there);
  - ai-almanac sources discover /datasets --register --names <country>/ai-almanac-names.yaml:
    the sources, named as the shared datasets' are (AIFSv2, AIFS-ENSv2, GraphCast, CHIRPS v3) and
    linked to the live forecast models (ones already registered are left as they are).
So running it again only adds what's new. A country whose AI Almanac datasets come from the bucket
(Ethiopia's) is left as it is.

Standard library only, like spark_setup.py.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import country_data

ALMANAC = Path.home() / "ai-almanac"
ROOT = Path(os.environ.get("ALMANAC_DATASETS_DIR") or Path.home() / "ai-almanac-data")
IN_CONTAINER = "/datasets"
NAMES_FILE = "ai-almanac-names.yaml"
KEEP = ".fleet-demos-keep"  # (hcwfscale's fleet-demos: a folder holding it is left out of the bucket sync)
# Countries whose AI Almanac datasets are the shared ones from the program's bucket: left as they are.
BUCKET_COUNTRIES = {"Ethiopia"}

# Country data folder -> AI Almanac source folder, and how it's registered (as ~/ai-almanac's
# deploy/spark-datasets.yaml names the shared datasets'). Only rainfall: what AI Almanac benchmarks.
OBS = ("CHIRPSv3_Country_0p25deg", "obs", {"name": "CHIRPS v3", "obs_date_marks": "start"})
MODELS = {
    "AIFS_v2": {"name": "AIFSv2", "forecast_model_id": "aifs2"},
    "AIFS_ENS_v2": {"name": "AIFS-ENSv2", "forecast_model_id": "aifs2ens"},
    "graphcast": {"name": "GraphCast", "forecast_model_id": "graphcast_operational"},
}

# Run in AI Almanac's container: adds the region (argv: display name, a folder of its NetCDF data).
ADD_REGION = r'''
import asyncio, sys
from pathlib import Path
import xarray as xr
from ai_almanac.settings import reload_settings
from ai_almanac.server.app import _apply_migrations
from ai_almanac.server import db
from ai_almanac.server.services import region_catalog
from ai_almanac.server.services.data_sources import _spatial_bounds

name, folder = sys.argv[1], Path(sys.argv[2])
reload_settings()
_apply_migrations()
with xr.open_dataset(sorted(folder.glob("*.nc"))[0]) as ds:
    box = _spatial_bounds(ds)

async def main():
    try:
        await region_catalog.seed_packaged_regions()
        slug = region_catalog.region_slug(name)
        for r in await region_catalog.list_regions():
            if r["id"] == slug or str(r.get("display_name") or "").lower() == name.lower():
                print(f"region {r['id']!r} ({r['display_name']}) is already there")
                return
        r = await region_catalog.create_region(display_name=name, land_only=False,
                                               description=f"{name}: the training program's country data", **box)
        print(f"added region {r['id']!r} ({name}): lat {box['lat_min']} to {box['lat_max']}, "
              f"lon {box['lon_min']} to {box['lon_max']}")
    finally:
        await db.engine.dispose()

asyncio.run(main())
'''


def say(msg: str):
    print(msg, flush=True)


def display(country: str) -> str:
    return country.replace("_", " ")


def slug(country: str) -> str:
    """The region id AI Almanac gives the display name (region_catalog.region_slug): the folder is
    named so, for discovery to put its sources in that region."""
    return re.sub(r"[^a-z0-9]+", "-", display(country).strip().lower()).strip("-")


def sources(country: str) -> list[tuple[Path, Path, dict]]:
    """(the country data folder, its AI Almanac folder, how to register it), for each that's there."""
    data, dest = country_data.data_dir(country), ROOT / slug(country)
    out = [(data / OBS[0] / country, dest / OBS[1], OBS[2])]
    out += [(data / "Forecasts_Country_0p25deg" / m / "tp" / country, dest / "models" / m, meta)
            for m, meta in MODELS.items()]
    return [(src, d, meta) for src, d, meta in out if src.is_dir() and any(src.glob("*.nc"))]


def link_files(src: Path, dest: Path, dry: bool) -> tuple[int, int]:
    """Hard-links src's NetCDF files into dest (a copy across disks); a file of the same size is kept
    (a copy made by hand, say), one that changed is replaced. Returns (linked or copied, kept)."""
    done = same = 0
    if not dry:
        dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.nc")):
        target = dest / f.name
        try:
            if target.exists() and (os.path.samefile(f, target) or target.stat().st_size == f.stat().st_size):
                same += 1
                continue
        except OSError:
            pass
        done += 1
        if dry:
            continue
        tmp = dest / f".{f.name}.part"
        tmp.unlink(missing_ok=True)
        try:
            os.link(f, tmp)
        except OSError:  # (another disk, say)
            shutil.copy2(f, tmp)
        os.replace(tmp, target)
    return done, same


def names_yaml(found: list[tuple[Path, Path, dict]]) -> str:
    """The --names file for `sources discover`: paths relative to /datasets (the scanned folder)."""
    lines = ["# Written by the training program's demos/demo1/ai_almanac_data.py: the names its",
             "# country datasets register under (paths relative to /datasets).", "sources:"]
    for _, dest, meta in found:
        lines.append(f"  {dest.relative_to(ROOT).as_posix()}:")
        lines += [f"    {k}: {v!r}" if " " in str(v) else f"    {k}: {v}" for k, v in meta.items()]
    return "\n".join(lines) + "\n"


def is_copy_of_country_data(folder: Path, country: str) -> bool:
    """A folder made by hand from the country data (every NetCDF file in it is one of the country
    data's, by name and size), like Nigeria's first one."""
    data = country_data.data_dir(country)
    ours = {f.name: f.stat().st_size for f in data.rglob("*.nc")} if data.is_dir() else {}
    files = list(folder.rglob("*.nc"))
    return bool(files) and all(ours.get(f.name) == f.stat().st_size for f in files)


def almanac(args: list[str], stdin: str | None, dry: bool) -> int:
    """Runs args in AI Almanac's container (~/ai-almanac's compose.spark.yaml, as scripts/spark.sh
    runs it): in the running one, or a one-off one when it isn't running."""
    if dry:
        return 0
    uid = os.getuid()
    env = dict(os.environ)
    env.setdefault("ALMANAC_DATA_DIR", str(Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
                                           / "ai-almanac-spark"))
    env.setdefault("ALMANAC_DATASETS_DIR", str(ROOT))
    env.setdefault("ALMANAC_PORT", "8765")
    sock = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}") / "docker.sock"
    if "DOCKER_HOST" not in env and sock.is_socket():
        env["DOCKER_HOST"] = f"unix://{sock}"
    compose = ["docker", "compose", "-f", "compose.spark.yaml"]
    up = subprocess.run([*compose, "ps", "--status", "running", "-q", "almanac"], cwd=ALMANAC, env=env,
                        capture_output=True, text=True).stdout.strip()
    cmd = [*compose, "exec", "-T", "almanac", *args] if up else [*compose, "run", "--rm", "--no-deps", "-T",
                                                                  "almanac", *args]
    return subprocess.run(cmd, cwd=ALMANAC, env=env, input=stdin, text=True).returncode


def run(country: str, dry: bool = False) -> int:
    name, dest = display(country), ROOT / slug(country)
    say(f"AI Almanac datasets for {name}{' (dry run: nothing changes)' if dry else ''}")
    if not (ALMANAC / "compose.spark.yaml").is_file():
        say(f"  AI Almanac isn't in {ALMANAC} on this Spark: nothing to register with")
        return 1
    if country in BUCKET_COUNTRIES:
        say(f"  {name}'s datasets already come from the program's bucket: left as they are")
        return 0
    if dest.is_dir() and not (dest / KEEP).exists() and not is_copy_of_country_data(dest, country):
        say(f"  {dest} is already there and isn't made from the country data: left as it is")
        return 0
    found = sources(country)
    if not found:
        say(f"  no {name} rainfall data in {country_data.data_dir(country)} yet: download the country "
            "data first (the dashboard's Country data, or python3 country_data.py)")
        return 1
    for src, d, meta in found:
        done, same = link_files(src, d, dry)
        say(f"  {meta['name']:<11} {d}: {done} file(s) {'to link' if dry else 'linked'}, {same} already there")
    if not dry:
        (dest / KEEP).write_text("This folder is made on this Spark (the training program's ai_almanac_data.py)\n"
                                 "from its country data: the fleet's sync of ~/ai-almanac-data leaves it alone.\n")
        (dest / NAMES_FILE).write_text(names_yaml(found))
    obs = next((d for _, d, m in found if m is OBS[2]), found[0][1])
    say(f"  adding the region {name} (its bounds: the data's)")
    rc = almanac(["python", "-", name, f"{IN_CONTAINER}/{obs.relative_to(ROOT).as_posix()}"], ADD_REGION, dry)
    if rc != 0:
        say("  the region wasn't added (see above): nothing registered")
        return rc
    say("  registering the sources")
    return almanac(["ai-almanac", "sources", "discover", IN_CONTAINER, "--register", "--names",
                    f"{IN_CONTAINER}/{slug(country)}/{NAMES_FILE}"], None, dry)


def add_arguments(p: argparse.ArgumentParser):
    p.add_argument("--country", help="a country whose data is on this Spark (default: this Spark's)")
    p.add_argument("--dry-run", action="store_true", help="show what it would do; change nothing")


def main_args(a) -> int:
    country = country_data.country_named(a.country) if a.country else country_data.country_from_hostname(
        country_data.hostname())
    if not country:
        print(f"No country for this Spark ({country_data.hostname()}); give --country, one of: "
              f"{', '.join(country_data.COUNTRIES)}", file=sys.stderr)
        return 2
    return run(country, a.dry_run)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_arguments(p)
    return main_args(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
