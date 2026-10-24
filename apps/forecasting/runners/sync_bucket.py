"""Share saved runs between Sparks through a cloud bucket.

    docker exec ai_weather /opt/envs/ui/bin/python apps/forecasting/runners/sync_bucket.py push --dry-run
    docker exec ai_weather /opt/envs/ui/bin/python apps/forecasting/runners/sync_bucket.py pull --models aurora15 atlas_crps

The bucket comes from --bucket or $DEMO3_BUCKET (e.g. gs://<bucket>/demo3), laid out
like outputs/: <model>/<YYYYMMDDTHH>_<lead>h_m<members>.nc plus its .json timings.
pull downloads runs this Spark doesn't have (a public bucket needs no credentials).
push uploads runs the bucket doesn't have yet. It needs a key that can write to the
bucket: the Sparks' own key is read-only, so their runs stay on them. Synthetic runs
are never pushed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apps.forecasting import store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("direction", choices=["push", "pull"])
    ap.add_argument("--bucket", default=os.environ.get("DEMO3_BUCKET", ""))
    ap.add_argument("--models", nargs="+", help="only these models (default: all)")
    ap.add_argument("--files", nargs="+", help="push only these saved runs (paths to .nc files)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be copied; copy nothing")
    a = ap.parse_args()
    if not a.bucket:
        ap.error("no bucket: pass --bucket gs://... or set DEMO3_BUCKET")
    store.BUCKET = a.bucket.rstrip("/")
    fs, root = store.bucket_fs()

    remote = {(m, n): size for m, n, size in store.remote_runs(max_age_s=0)}
    local = {(p.parent.name, p.name): p for p in store.OUTPUT_DIR.glob("*/*.nc")
             if store._NAME.match(p.name) and "_synthetic" not in p.name}
    if a.files:
        keep = {(Path(f).parent.name, Path(f).name) for f in a.files}
        local = {k: p for k, p in local.items() if k in keep}
    want = lambda m: not a.models or m in a.models
    if a.direction == "push":
        todo = [(k, p) for k, p in sorted(local.items()) if want(k[0]) and remote.get(k) != p.stat().st_size]
        total = sum(p.stat().st_size for _, p in todo)
    else:
        todo = [(k, remote[k]) for k in sorted(remote) if want(k[0]) and k not in local]
        total = sum(size for _, size in todo)
    for (m, n), _ in todo:
        print(f"  {a.direction} {m}/{n}")
    print(f"{len(todo)} file(s), {total / 1e9:.2f} GB {'to' if a.direction == 'push' else 'from'} {store.BUCKET}")
    if a.dry_run:
        return 0
    for i, ((m, n), item) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {m}/{n}", flush=True)
        if a.direction == "push":
            fs.makedirs(f"{root}/{m}", exist_ok=True)  # a no-op on cloud buckets
            side = item.with_suffix(".json")
            if side.exists():
                fs.put(str(side), f"{root}/{m}/{side.name}")
            fs.put(str(item), f"{root}/{m}/{n}")
        else:
            store.download(m, n)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
