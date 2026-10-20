"""Time FGN (WeatherNext 2) on the Spark using Google's sample HRES case.

FGN starts from ECMWF HRES analyses in DeepMind's format, which Google
publishes for a single date (2024-10-07 00Z, gs://dm_graphcast/weathernext2/).
Until we build a converter from ECMWF open data, this script runs that case:
checkpoint 1 of 4, N noise members, 30 six-hourly steps (7.5 days, the longest
sample file), and writes the forecast in the shared output format.

The sample file has no 100 m wind, which the WeatherNext2 checkpoints need, so
the default is the WeatherNextCyclones checkpoint: the same FGN architecture
and algorithm (per Google's README the only difference is 100 m wind), so the
compute and timing are equivalent.

    python runners/fgn_sample_timing.py --data-dir /workspace/e2s/fgn_data --members 3
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))
from demo3 import contract  # noqa: E402

DATA_FILE = "hres_2024-10-07_0p25_steps30.nc"
CHECKPOINTS = {  # config name -> local weights file (checkpoint 1 of 4)
    "WeatherNextCyclones": "WeatherNextCyclones_2025_model1.npz",
    "WeatherNext2": "WeatherNext2_model1.npz",
    # 1 deg lightweight version; the 0.25 deg models above exceed the Spark's
    # 121 GB shared memory on the GPU attention path (a single 33 GB allocation fails).
    "WeatherNextCyclones_Mini": "WeatherNextCyclones_Mini_2024.npz",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--config", choices=sorted(CHECKPOINTS), default="WeatherNextCyclones")
    ap.add_argument("--data-file", default=DATA_FILE,
                    help="sample file in --data-dir (steps-04 is enough to measure per-step time)")
    a = ap.parse_args()

    import haiku as hk
    import jax
    import xarray_jax
    from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
    from weathernext.weathernext2 import fgn

    t0 = time.time()
    config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{a.config}")
    with open(a.data_dir / CHECKPOINTS[a.config], "rb") as f:
        ckpt = checkpoint.load(f, fgn.CheckPoint)
    load_s = time.time() - t0

    t1 = time.time()
    batch = xr.load_dataset(a.data_dir / a.data_file).compute()
    data_s = time.time() - t1
    n_steps = batch.sizes["time"] - 2  # first two frames are the inputs (-6 h, 0 h)
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch, target_lead_times=slice("6h", f"{n_steps * 6}h"), **dataclasses.asdict(config.task))

    # GPUs need the alternative attention implementation (slower, more memory).
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
    rngs = np.stack([jax.random.fold_in(jax.random.PRNGKey(0), i) for i in range(a.members)])

    t2 = time.time()
    chunks, step_times = [], []
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
            predictor_fn=fwd_pmap, rngs=rngs, inputs=inputs, targets_template=targets * np.nan,
            forcings=forcings, num_steps_per_chunk=1, num_samples=a.members,
            pmap_devices=jax.local_devices()):
        # Copy to host: blocks until the step is really done (JAX is async, so
        # timings are otherwise meaningless) and keeps outputs off the GPU.
        chunk = chunk.as_numpy()
        chunks.append(chunk)
        step_times.append(time.time() - t2)
        # One device: all steps of member 1 come first, then member 2, ...
        leads = (chunk.time.values / np.timedelta64(1, "h")).astype(int).tolist()
        print(f"chunk {len(step_times)} leads={leads} samples={chunk.sizes.get('sample')} "
              f"at {step_times[-1]:.0f} s", flush=True)
    pred = xr.combine_by_coords(chunks)
    run_s = time.time() - t2
    first_step_s = step_times[0]  # includes JIT compilation
    per_step_s = (run_s - first_step_s) / max(1, len(step_times) - 1)  # one member, one 6 h step
    peak_gb = (jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e9

    init = pd.Timestamp(batch.datetime.isel(batch=0, time=1).values)
    p = pred.isel(batch=0)
    hours = (p.time.values / np.timedelta64(1, "h")).astype(int)
    out = xr.Dataset({
        "t2m": p["2m_temperature"],
        "z500": p["geopotential"].sel(level=500, drop=True),
        "tp": p["total_precipitation_6hr"] * 1000.0,
    }).rename(sample="ensemble").assign_coords(time=hours).rename(time="lead_time")
    out = out.assign_coords(ensemble=np.arange(a.members))
    # Prepend lead 0 from the analysis so the file starts at the init time like the others.
    lead0 = xr.Dataset({
        "t2m": batch["2m_temperature"].isel(batch=0, time=1, drop=True),
        "z500": batch["geopotential"].isel(batch=0, time=1, drop=True).sel(level=500, drop=True),
        "tp": batch["2m_temperature"].isel(batch=0, time=1, drop=True) * np.nan,
    }).expand_dims(ensemble=out.ensemble, lead_time=[0])
    out = xr.concat([lead0, out], dim="lead_time")
    out["tp"] = out.tp.clip(min=0).where(out.lead_time > 0)
    out = contract.finalize(out, model_key="fgn", model_name="FGN (WeatherNext 2)", init=init,
                            init_source=f"HRES analysis (Google sample case; {a.config} checkpoint)")
    dest = APP_ROOT / "outputs" / "fgn" / f"{init:%Y%m%dT%H}_{n_steps * 6}h_m{a.members}.nc"
    dest.parent.mkdir(parents=True, exist_ok=True)
    contract.write(out, dest)

    rec = dict(model="fgn", checkpoint=a.config, init=f"{init:%Y-%m-%dT%H}", lead_hours=n_steps * 6, members=a.members,
               load_s=round(load_s, 1), data_load_s=round(data_s, 1), run_s=round(run_s, 1),
               first_step_s=round(first_step_s, 1), per_step_s=round(per_step_s, 2),
               est_10day_run_s=round(first_step_s + per_step_s * (40 * a.members - 1), 1), peak_gpu_gb=round(peak_gb, 1),
               output_mb=round(dest.stat().st_size / 1e6, 1), when=time.strftime("%Y-%m-%dT%H:%M"))
    with open(APP_ROOT / "outputs" / "_timings_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
