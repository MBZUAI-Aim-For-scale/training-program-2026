"""Download every live model's weights into the cache, so a model's first run doesn't wait for them.

Run inside the image by docker/prefetch.sh (which spark_setup.py starts after the image is built,
in the background); the cache is the one docker/run.sh mounts (~/.cache/demo3 on the Spark).

    python runners/prefetch_weights.py             every live model's that isn't here yet
    python runners/prefetch_weights.py --missing   print the models whose weights aren't here yet
    python runners/prefetch_weights.py --one KEY   one model's (run in that model's environment)

Each model's weights come from where its runner gets them: Earth2Studio's package (into its
cache), NeuralGCM's checkpoint, FGN's weights and Google's sample case. A model whose download
finished is recorded in DONE and skipped next time; one that fails is tried again next time.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "runners"))

from demo3 import catalog

DONE = Path.home() / ".cache" / "demo3-weights.json"
# The models run through Earth2Studio, by their class in earth2studio.models.px
E2S = {"graphcast": "GraphCastOperational", "atlas_crps": "AtlasCRPS", "aurora15": "Aurora1p5",
       "aifs2_single": "AIFS2", "aifs2_ens": "AIFS2ENS"}


def fetch_one(key: str) -> None:
    if key in E2S:
        from earth2studio.models import px
        cls = getattr(px, E2S[key])
        # Loads the model on the CPU: that downloads every file it needs into Earth2Studio's cache,
        # as the runner's first load would.
        cls.load_model(cls.load_default_package())
    elif key == "neuralgcm":
        import runner_neuralgcm
        runner_neuralgcm.fetch_checkpoint()
    elif key == "fgn":
        import runner_fgn
        for local, remote in (runner_fgn.WEIGHTS, runner_fgn.SAMPLE):
            runner_fgn._fetch(local, remote, lambda message, *_a, **_k: print(message, flush=True))
    else:
        raise SystemExit(f"no weights to fetch for {key}")


def done() -> dict:
    try:
        return json.loads(DONE.read_text())
    except (OSError, ValueError):
        return {}


def missing() -> list[catalog.Model]:
    have = done()
    return [m for m in catalog.MODELS if m.live and m.key not in have]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    how = ap.add_mutually_exclusive_group()
    how.add_argument("--one", metavar="KEY", choices=sorted(catalog.BY_KEY))
    how.add_argument("--missing", action="store_true")
    a = ap.parse_args()
    if a.one:
        fetch_one(a.one)
        return 0
    todo = missing()
    if a.missing:
        print(" ".join(m.key for m in todo))
        return 0

    print(f"{len(todo)} model{'s' * (len(todo) != 1)} to fetch, about "
          f"{sum(m.weights_gb for m in todo):.0f} GB: {', '.join(m.name for m in todo) or 'none'}", flush=True)
    failed = []
    for i, m in enumerate(todo, 1):
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {m.name} (~{m.weights_gb:g} GB)…", flush=True)
        py = catalog.ENV_ROOT / m.env / "bin" / "python"
        rc = subprocess.run([str(py), __file__, "--one", m.key], cwd=APP_ROOT, check=False).returncode
        if rc:
            failed.append(m.name)
            print(f"[{i}/{len(todo)}] {m.name}: FAILED (exit code {rc}); tried again next time", flush=True)
            continue
        record = done()
        record[m.key] = time.strftime("%Y-%m-%dT%H:%M")
        tmp = DONE.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(DONE)
        print(f"[{i}/{len(todo)}] {m.name}: done in {time.time() - t0:.0f} s", flush=True)
    print(f"Weights for {', '.join(failed)} failed." if failed else "Every model's weights are here.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
