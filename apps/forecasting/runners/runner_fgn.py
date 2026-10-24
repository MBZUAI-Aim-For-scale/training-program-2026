"""FGN Mini (WeatherNext 2, 1 deg) from Google DeepMind.

Starting conditions:
  2024-10-07 00Z, up to 7.5 days   Google's sample file (HRES analyses)
  any other start                  built from ECMWF open data by fgn_convert.py,
                                   which runs in the e2s018 environment; cached
                                   in outputs/_fgn_inputs/

The full 0.25 deg FGN does not fit in a Spark's memory, so this is the 1 deg
Mini model, and the only Mini weights Google publishes are the cyclone-tuned
WeatherNextCyclones_Mini (<2024). Members are separate noise seeds. The weights
and the sample file (~1.1 GB) download on first use into DEMO3_FGN_DIR.
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from _e2s import peak_gpu_gb

APP_ROOT = Path(__file__).resolve().parent.parent
FGN_DIR = Path(os.environ.get("DEMO3_FGN_DIR", Path.home() / ".cache" / "fgn"))
INPUTS_DIR = APP_ROOT / "outputs" / "_fgn_inputs"
BUCKET = "https://storage.googleapis.com/dm_graphcast/weathernext2"
CONFIG = "WeatherNextCyclones_Mini"
WEIGHTS = ("WeatherNextCyclones_Mini_2024.npz", "params/WeatherNextCyclones_Mini_%3C2024.npz")
SAMPLE = ("hres_2024-10-07_1p0_steps30.nc",
          "dataset/source-hres_forecast_init-2024-10-07%2000:00:00_res-1.0_levels-13_steps-30.nc")
SAMPLE_INIT, SAMPLE_STEPS = pd.Timestamp("2024-10-07T00"), 30
CONVERTER_PY = Path(os.environ.get("DEMO3_ENV_ROOT", "/opt/envs")) / "e2s018" / "bin" / "python"


def _fetch(local: str, remote: str, report) -> Path:
    path = FGN_DIR / local
    if not path.exists():
        report(f"Downloading {local} from Google (first run only)…", 0.02)
        FGN_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(f"{BUCKET}/{remote}", tmp)
        tmp.replace(path)
    return path


def _inputs(init: pd.Timestamp, steps: int, sample: Path, report) -> tuple[xr.Dataset, str]:
    if init == SAMPLE_INIT and steps <= SAMPLE_STEPS:
        batch = xr.load_dataset(sample).isel(time=slice(0, steps + 2))
        return batch, "HRES analysis (Google sample case)"
    # the converter writes just the two input times; forecast frames are added here
    cached = INPUTS_DIR / f"{init:%Y%m%dT%H}.nc"
    if not cached.exists():
        report("Building starting conditions from ECMWF open data…", 0.05)
        cmd = [str(CONVERTER_PY), str(Path(__file__).with_name("fgn_convert.py")),
               "--init", f"{init:%Y-%m-%dT%H}", "--steps", "0", "--static", str(sample),
               "--out", str(cached), "--source", os.environ.get("DEMO3_IFS_SOURCE", "aws")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-4000:], sep="\n", flush=True)
        if r.returncode != 0:
            lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
            raise RuntimeError(lines[-1] if lines else "FGN input conversion failed")
    batch = xr.load_dataset(cached)
    times = pd.to_timedelta(np.arange(steps + 2) * 6, unit="h")
    start = batch.datetime.values[0, 0]
    batch = batch.reindex(time=times).assign_coords(
        datetime=(("batch", "time"), (start + times.values)[None, :]))
    return batch, "IFS analysis (ECMWF open data)"


def run(init, lead_hours, members, report):
    import haiku as hk
    import jax
    import xarray_jax
    from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
    from weathernext.weathernext2 import fgn

    steps = lead_hours // 6
    t0 = time.time()
    weights = _fetch(*WEIGHTS, report)
    sample = _fetch(*SAMPLE, report)
    report("Loading FGN Mini…", 0.03)
    config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{CONFIG}")
    with open(weights, "rb") as f:
        ckpt = checkpoint.load(f, fgn.CheckPoint)
    load_s = round(time.time() - t0, 1)
    report(f"Model loaded in {load_s:.0f} s; preparing starting conditions…", 0.05, load_s=load_s)

    t_in = time.time()
    batch, source = _inputs(pd.Timestamp(init), steps, sample, report)
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice("6h", f"{steps * 6}h"), **dataclasses.asdict(config.task))
    fetch_s = round(time.time() - t_in, 1)
    report(f"Initial conditions ready in {fetch_s:.0f} s", 0.1, fetch_s=fetch_s)

    # GPUs need the alternative attention implementation.
    config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"].keywords[
        "transformer_kwargs"]["attention_type"] = "triblockdiag_mha"
    config_inference = fgn.PredictorConfig(
        task=config.task, predictor_constructor=config.predictor_constructor,
        predictor_kwargs=config.predictor_kwargs,
        predictor_wrappers=config.predictor_wrappers[:-1])  # drop the ensemble wrapper

    @hk.transform
    def run_forward(inputs, targets_template, forcings):
        return fgn.construct_predictor(config_inference)(
            inputs, targets_template=targets_template, forcings=forcings)

    fwd = jax.jit(lambda rng, i, t, f: run_forward.apply(ckpt.params, rng, i, t, f))
    fwd_pmap = xarray_jax.pmap(fwd, dim="sample")
    rngs = np.stack([jax.random.fold_in(jax.random.PRNGKey(0), i) for i in range(members)])

    t1 = time.time()
    chunks, total = [], steps * members
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
            predictor_fn=fwd_pmap, rngs=rngs, inputs=inputs, targets_template=targets * np.nan,
            forcings=forcings, num_steps_per_chunk=1, num_samples=members,
            pmap_devices=jax.local_devices()):
        chunks.append(chunk.as_numpy())  # to host: keeps outputs off the GPU
        done = len(chunks)
        report(f"Step {done}/{total} (member {(done - 1) // steps + 1}/{members})",
               0.1 + 0.85 * done / total)
    run_s = round(time.time() - t1, 1)
    report(f"Forecast finished in {run_s:.0f} s", 0.95, run_s=run_s, peak_gpu_gb=peak_gpu_gb())

    p = xr.combine_by_coords(chunks).isel(batch=0)
    hours = (p.time.values / np.timedelta64(1, "h")).astype(int)
    out = xr.Dataset({
        "t2m": p["2m_temperature"],
        "z500": p["geopotential"].sel(level=500, drop=True),
        "tp": p["total_precipitation_6hr"] * 1000.0,
    }).rename(sample="ensemble").assign_coords(time=hours).rename(time="lead_time")
    out = out.drop_vars([c for c in out.coords if c not in ("ensemble", "lead_time", "lat", "lon")])
    out = out.assign_coords(ensemble=np.arange(members))
    # lead 0 from the starting analysis, like the other models
    t0_state = batch.isel(batch=0, time=1, drop=True)
    lead0 = xr.Dataset({"t2m": t0_state["2m_temperature"],
                        "z500": t0_state["geopotential"].sel(level=500, drop=True),
                        "tp": t0_state["2m_temperature"] * np.nan})
    lead0 = lead0.drop_vars([c for c in lead0.coords if c not in ("lat", "lon")])
    lead0 = lead0.expand_dims(ensemble=out.ensemble, lead_time=[0])
    out = xr.concat([lead0, out], dim="lead_time")
    out["tp"] = out.tp.clip(min=0).where(out.lead_time > 0)
    out.attrs["fgn_init_source"] = source
    return out
