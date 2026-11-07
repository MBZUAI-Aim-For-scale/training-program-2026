# Demo 3 — Running Your First AI Weather Forecast

**New to this? Follow the step-by-step [Beginner's Guide](../BEGINNER_GUIDE.md#demo-3--run-your-first-ai-weather-forecast).**

A Streamlit app, running on the DGX Spark, for running AI weather models. Participants choose up to four models, a start date, lead time, region and use case (heat, precipitation or rainy-season onset), run the forecast (or load a saved run), and compare it with observations.

**Status:**
- All 7 models run live on the Spark from one Docker image (`docker/`), and saved runs load instantly.
- The event movie (forecast vs observations) is Panchali's `demo3/event_movie/forecast_event_movie.py`.
- Results are Panchali's event movie (forecast next to observations, day by day: CHIRPS v3 rainfall, CHIRTS daily maximum temperature) and event-track map, for heat or precipitation; plus how the forecast was made (load, inputs, run, save, with times on the Spark), maps of z500 / 2 m temperature / rainfall, and a NetCDF download.
- Each model card lists what it needs on the Spark: weight size, peak GPU memory, output size, grid and inputs.
- **Onset** use case: onset maps (forecast vs observed), the rainfall series and an onset movie, by a choice of definitions (ICPAC, Moron–Robertson, Moron–Robertson with MOK filter), each adjustable.
- **Several models at once:** pick up to 4 models at the start. Saved runs load (or download from the shared bucket) and the rest run one after another. Results: how each forecast was made, event movies (or onset maps and movies) for the models you choose, a time-series comparison of all of them, and forecast maps.
- Loading saved runs from a bucket (`runners/sync_bucket.py`, `DEMO3_BUCKET`).

## Run it

**Full step-by-step setup (new Spark, GPU check, copying weights, troubleshooting): [SETUP.md](SETUP.md).**

**On the Spark** (aarch64; build once, about an hour, mostly compiling two libraries):

```bash
cd ~/training-program-2026/demos/demo3
docker build -t demo3 -f docker/Dockerfile .
bash docker/run.sh            # or DEV=1 bash docker/run.sh to use this folder's code live
```

`run.sh` keeps saved runs in `outputs/` and its cache (model weights, ~21 GB, plus the starting
conditions runs download; ~90 GB over time) in `~/.cache/demo3`; set `DEMO3_CACHE` to reuse an existing cache.

**Model weights:** `bash docker/prefetch.sh` downloads every model's weights into that cache in the
background (container `demo3-weights`: `docker logs -f demo3-weights`), so no model's first run waits
for them. `spark-setup` runs it after building the image, the Spark's own setup after each update
included; it skips the models already fetched (`runners/prefetch_weights.py --missing` lists the rest).

**From your laptop:** open a tunnel, then browse to http://localhost:8502.

```bash
ssh -L 8502:localhost:8502 <you>@rwanda-spark
```

**After a reboot**, if the container won't start with "unresolvable CDI devices", Docker started
before the GPU list existed: `systemctl --user restart docker && docker start demo3`.

**Environments in the image** (`/opt/envs/<env>`, pinned in `docker/locks/`):
`e2s018` Atlas CRPS, Aurora 1.5, AIFS v2 single · `e2s018ens` AIFS v2 ENS ·
`graphcast` GraphCast and NeuralGCM · `fgn` FGN · `ui` the page.

## Layout

```
app.py                 Streamlit page: steps 2–3 and results
demo3/catalog.py       the 7 models (type, environment, lead-time limit, members)
demo3/regions.py       program countries → padded plotting boxes
demo3/contract.py      the output-file format every runner writes
demo3/store.py         where outputs live; reusing saved runs
demo3/jobs.py          launches runs in the background and tracks progress
demo3/movie.py         runs the event movie in the background for the loaded run
demo3/results.py       pipeline stages, model requirements, forecast maps, NetCDF download
demo3/onset.py         onset definitions and detection (matches ROMP), observed + forecast series
demo3/onset_view.py    the onset tab: definition picker, maps, series, onset movie
demo3/obs.py           observations: CHIRPS v3 daily rainfall and CHIRTS daily Tmax (yearly files, downloaded once)
demo3/runs.py          gets the selected models' forecasts: saved, downloaded, or run one at a time
demo3/compare.py       the time-series comparison of the loaded models
demo3/data/            ROMP's MWset: local 5-day wet-spell amounts for Moron–Robertson
demo3/event_movie/     Panchali's forecast_event_movie.py (unchanged) + the Spark adapter
demo3/theme.py         styling, matching the Demo 5 platform
runners/run_model.py   runner entry point
runners/fgn_convert.py FGN starting conditions from ECMWF open data (see the FGN section)
runners/sync_bucket.py push/pull saved runs to/from a shared bucket
runners/prefetch_weights.py  downloads every model's weights ahead (docker/prefetch.sh runs it)
timings.json           measured Spark runtimes shown in brackets in the app
docker/                the image: Dockerfile, run.sh, per-environment package locks
outputs/               saved runs: outputs/{model}/{YYYYMMDDTHH}_{lead}h_m{members}.nc
```

## Output format (what the plots receive)

Each run is one NetCDF file covering the whole globe:

| | |
|---|---|
| dims | `ensemble, lead_time, lat, lon` (`ensemble` has size 1 for deterministic models) |
| `lead_time` | integer hours after the start time: 0, 6, 12, … |
| `lat` / `lon` | ascending; lon runs −180 to 180 |
| `tp` | precipitation accumulated over the previous 6 h, in **mm** (NaN at lead 0) |
| `t2m` | 2 m temperature, in **K** |
| `z500` | 500 hPa geopotential, in **m² s⁻²** (divide by 9.80665 to get height in m) |
| coords | `valid_time` (along lead_time), `init_time` |
| attrs | `model`, `model_name`, `init_source`, `members` |

## Results: the event movie

The results are Panchali's `demo3/event_movie/forecast_event_movie.py`, kept unchanged
(drop in new versions as they come). `run_event_movie.py` adapts it to the Spark: it reads
Demo 3's output files and takes its observations from `demo3/obs.py`. The use case picks the
movie: **Heat** = daily max 2 m temperature against CHIRTS, **Precipitation** = daily rainfall
against CHIRPS v3. Both observations cover land only, so the forecast is blanked over the sea
and outside the observed box, and the maps and regional series compare the same area. Each movie
is rendered once per request and saved under `outputs/_movies/`.

**Heat observations:** CHIRTS daily Tmax (CHIRTS-ERA5) from the yearly country files on the
models' 0.25° grid in `gs://aim4scale_training_2026/CHIRTS_Country_0p25deg/Tmax` (from 2000), read
from this Spark's country data first (every country Demo 1's country-data picker downloaded, mounted at
`/country_data`), else downloaded once into `outputs/_obs/chirts_0p25/`. Programme countries only; for other regions
the page falls back to ERA5 from Google's ARCO copy (cached in `outputs/_obs_cache/`). Two
things to know when reading heat results: the models give a value every 6 hours, so their daily
maximum misses part of the afternoon peak (0.3–1.3 °C in our cases, measured with hourly ERA5);
and CHIRTS reads warmer than ERA5 over land (0.4–3.6 °C in our cases), because it is built from
station temperatures at 5 km. The heat chart can show ERA5 as a second line to make that visible.

## Results: onset

The **Onset** use case finds when the rainy season starts, at every grid point, by a definition
chosen on the page (defaults: Moron–Robertson with MOK filter for India and Bangladesh, ICPAC
elsewhere). The definitions follow Narayana's note and the ROMP package
(`apps/benchmarking/ROMP`):

| Definition | Wet spell | Dry-spell check | Search from |
|---|---|---|---|
| ICPAC (Ethiopia; used across Africa) | 3 days, each ≥ 1 mm, total > 20 mm | no 7 consecutive days < 1 mm in the next 21 days | the region's rainy-season start |
| Moron–Robertson (India) | 5 days, each ≥ 1 mm, total > local climatological amount (ROMP's MWset) | no 10-day spell < 5 mm in total in the next 30 days | 1 April |
| Moron–Robertson with MOK filter | as above | none | 2 June (median monsoon onset over Kerala, 1901–2024) |

Every setting (thresholds, dry-spell length and window, search start) can be changed on the page.

India, Bangladesh and Pakistan default to Moron–Robertson with MOK filter (ROMP's MWset amounts
cover 97% of Pakistan's monsoon belt); everywhere else defaults to ICPAC. Colombia has no
published definition: ICPAC's rule reproduces its observed season starts (CHIRPS v3 1998–2025),
searching from mid-February and mid-August for the Andes' two seasons (`onset.REGION_NOTES`).

- **Rainfall used:** observed CHIRPS v3.0 at 0.25° (`demo3/obs.py`) from the search start to the
  day before the forecast, then the forecast's daily rainfall put on the CHIRPS grid. This is the
  "sat" version (days within each pentad split using IMERG). Programme countries read their yearly
  file on the models' own 0.25° grid (cells on whole and quarter degrees, as GraphCast's) from
  `gs://aim4scale_training_2026/CHIRPSv3_Country_0p25deg` (from 2000): this Spark's country data
  first, else the bucket, cached in `outputs/_obs/chirps_v3_0p25/`. Other regions download CHC's
  global 0.25° daily files (~0.9 MB a day, CHIRPS's own grid) into `outputs/_obs/chirps_v3/p25/`.
  Models on another grid (NeuralGCM's) are interpolated onto the observations'. CHIRPS covers
  land, 60°S–60°N, and runs about a month behind; days missing before the forecast are flagged on
  the page.
- **Possible onset:** confirming onset needs the wet spell plus the dry-spell window (24 days for
  ICPAC, 35 for Moron–Robertson), longer than a 10–15 day forecast. A wet spell whose check runs
  past the end of the data is shown as possible onset, not onset.
- **Observed onset** (for checking the forecast) uses CHIRPS alone, including days after the start.
- **Onset movie:** one frame per forecast day of where onset has happened, forecast next to observed.
- Rainy-season start dates per country are in `onset.SEASON_STARTS`, to be checked against
  Narayana's country/phenomenon table.
- Checked against the onset explorer's ICPAC code on 2,000 random series: identical, except that
  the explorer skips the last day whose 24-day window just fits.

## Several models at once

Step 2 takes up to 4 models. The date picker offers the start dates all of them can use (if
none is common, every date is offered and models that can't start are left out), the lead time
goes up to the shortest maximum among them, and the member count applies to the ensemble
models, each capped at its own limit. One button then loads saved runs (any member count),
downloads runs from the shared bucket, and runs the rest on the Spark one after another, never
two at once.

Results, in order: how each forecast was made (pick the model); event movies, made one at a
time and only for the models chosen (default: the first); for Onset, onset maps per chosen model
and onset movies; the time-series comparison of every loaded model (area-mean daily rainfall on
the CHIRPS grid with CHIRPS v3, or daily maximum 2 m temperature on the CHIRTS grid with CHIRTS,
plus ERA5 on request; for Onset,
each model's onset share next to the observed one); forecast maps and the NetCDF download (pick
the model).

## Saved runs from the bucket

Saved runs can be loaded from the program's bucket,
`gs://aim4scale_training_2026/demo3/<model>/` (same file names as `outputs/`). The Sparks can only
read it, so runs made on a Spark stay on it.

- **Start the container:** `bash docker/run.sh` uses that bucket by itself (add `DEV=1`,
  `DEMO3_CACHE`, `DEMO3_OUTPUTS` as usual). `DEMO3_BUCKET=gs://...` uses another one, and
  `DEMO3_BUCKET=` none.
- **Credentials:** the bucket is private. On a Spark there is nothing to set up: `run.sh` passes the
  machine's read-only key (`/etc/fleet-gcs-key.json`) into the container. Elsewhere:
  `gcloud auth application-default login --no-launch-browser` (tick every permission on Google's
  consent page), and `run.sh` passes the resulting
  `~/.config/gcloud/application_default_credentials.json` into the container. A service-account key
  instead: `DEMO3_GCP_KEY=<key.json>`.
- **On the page:** a run in the bucket but not on this Spark shows **Download saved run**; the
  results say whether the run is in the bucket.
- **In bulk:** `runners/sync_bucket.py pull [--models ...] [--dry-run]` (inside the container), or
  `gcloud storage rsync` on the Spark. `push` needs a key that can write to the bucket.

## FGN (WeatherNext 2): starting conditions from ECMWF open data

FGN is not in Earth2Studio, and Google publishes FGN-ready inputs for one date only
(2024-10-07 00Z). The full 0.25° model does not fit in a Spark's memory, so Demo 3 runs
Google's 1° Mini model (`WeatherNextCyclones_Mini`, the only Mini weights published).

**Converter** (`runners/fgn_convert.py`, runs in the `e2s018` environment)
- Downloads ECMWF's IFS analyses (step 0) for the start time and 6 hours earlier, at FGN's
  13 pressure levels, puts them on FGN's 1° grid (taking the 1° points) and writes Google's
  file layout. Works for any date ECMWF open data covers (from 2024-03-01).
- What open data lacks, and what stands in for it:
  - sea-surface temperature: IFS skin temperature over the ocean, floored at seawater
    freezing (271.46 K) under sea ice;
  - surface geopotential and land–sea mask: fixed fields, copied from Google's sample file.
- Checked against Google's own 2024-10-07 file, rebuilt from ECMWF data: every field matches
  to within rounding (correlation 1.000), so Google built theirs the same way. The one
  approximation is sea-surface temperature (correlation 0.997, 0.08 K too cold on average).
- From the same 2026-05-15 start, FGN and AIFS v2 agree closely at 24–48 h (z500 and 2 m
  temperature correlation ≥ 0.999).

**Runner** (`runners/runner_fgn.py`)
- Starting conditions: Google's sample file on 2024-10-07 (up to 7.5 days); the converter for
  every other date.
- First run: downloads the weights and the sample file (1.1 GB) from Google into
  `DEMO3_FGN_DIR` (default `~/.cache/fgn`), so each Spark sets itself up.
- Caches converted inputs (53 MB per start date) in `outputs/_fgn_inputs/`, so a rerun of the
  same date skips the conversion (~95 s).
- 10 days, 3 members: ~2 min on the GPU, 1.8 GB peak memory.

## Adding a model

1. Create `runners/runner_<model_key>.py` with `run(init, lead_hours, members, report) -> xr.Dataset`, returning `tp`, `t2m` and `z500` in the units above.
2. Make sure the model's environment exists at `/opt/envs/<env>/bin/python` inside the container.
3. Add its entry to `MODELS` in `demo3/catalog.py`: it runs live once its environment and runner exist.
4. Teach `runners/prefetch_weights.py` where its weights come from (`E2S` for an Earth2Studio model).
5. Add its measured runtime to `timings.json`.
