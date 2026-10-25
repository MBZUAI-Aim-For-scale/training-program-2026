# AI forecasting (Demo 3) — Running Your First AI Weather Forecast

The **AI Forecast** page of the shared training site (`ui/app.py`; the page is
`ui/pages/ai_forecasting.py`). Participants choose up to four models, a start date, lead time,
region and use case (heat, precipitation or rainy-season onset), run the forecasts on the DGX
Spark (or load saved runs), and compare them with each other and with observations.

The page lives in `ui/pages/ai_forecasting.py`; everything it uses (models, runners, onset,
event movies, the shared bucket) is in this folder, packaged as `apps.forecasting`.

## Run it

**Full step-by-step setup (new Spark, GPU check, copying weights, troubleshooting): [SETUP.md](SETUP.md).**

**On the Spark**, from the top of the repository (aarch64; build once, about an hour):

```bash
cd ~/training-program-2026
docker build -t ai_weather -f docker/Dockerfile .
bash docker/run.sh            # or DEV=1 bash docker/run.sh to use this folder's code live
```

`run.sh` keeps saved runs in `outputs/` and model weights (~90 GB, downloaded on each model's
first run) in `~/.cache/ai_weather`; set `DEMO_OUTPUTS` / `DEMO_CACHE` to use other folders.
It reads saved runs from `gs://aim4scale_training_2026/demo3` with the Spark's read-only key.

**From your laptop:** open a tunnel, browse to http://localhost:8501 and choose **Demo3**.

```bash
ssh -L 8501:localhost:8501 <you>@<spark>
```

**Environments in the image** (`/opt/envs/<env>`, pinned in `docker/locks/`):
`e2s018` Atlas CRPS, Aurora 1.5, AIFS v2 single · `e2s018ens` AIFS v2 ENS ·
`graphcast` GraphCast and NeuralGCM · `fgn` FGN · `ui` the site.

## Layout

```
catalog.py             the 7 models (type, environment, dates, lead-time limit, members)
regions.py             program countries → padded plotting boxes
contract.py            the output-file format every runner writes
store.py               where outputs live (<repo>/outputs); saved runs; the shared bucket
jobs.py                launches runs in the background and tracks progress
runs.py                gets the selected models' forecasts: saved, downloaded, or run one at a time
movie.py               runs the event movie in the background for a loaded run
results.py             pipeline stages, model requirements, forecast maps (with Play/Pause), NetCDF
compare.py             the time-series comparison of the loaded models
onset.py               onset definitions and detection (matches ROMP), observed + forecast series
onset_view.py          onset definition picker, maps, series and onset movie
obs.py                 CHIRPS daily rainfall (yearly files, downloaded once)
theme.py               styling helpers (hover hints)
data/mwset4x4.nc4      ROMP's MWset: local 5-day wet-spell amounts for Moron–Robertson
event_movie/           Panchali's forecast_event_movie.py (unchanged) + the Spark adapter
runners/run_model.py   runner entry point (+ synthetic output generator)
runners/runner_*.py    one per model
runners/wave_source.py AIFS v2 inputs for 2000–2025: ECMWF wave dataset (bucket) + ERA5
runners/fgn_convert.py FGN starting conditions from ECMWF open data (see the FGN section)
runners/sync_bucket.py pull saved runs from the shared bucket in bulk
timings.json           measured Spark runtimes shown in the app
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
| attrs | `model`, `model_name`, `init_source`, `members`, `synthetic` |

## Results: the event movie

The results are Panchali's `event_movie/forecast_event_movie.py`, kept unchanged
(drop in new versions as they come). `run_event_movie.py` adapts it to the Spark: it reads
Demo 3's output files, and fetches ERA5 observations from Google's ARCO copy (cached in
`outputs/_obs_cache/`) when the cluster's ERA5 folder isn't there. The use case picks the
movie: **Heat** = daily max 2 m temperature, **Precipitation** = daily rainfall. Each movie is
rendered once per request and saved under `outputs/_movies/`.

To get test data without a GPU, turn on **Use synthetic output** in Developer options and press Run. This writes a fake file in exactly this format.

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

- **Rainfall used:** observed CHIRPS (`obs.py`; one ~90 MB file per year, downloaded once
  into `outputs/_obs/chirps/`) from the search start to the day before the forecast, then the
  forecast's daily rainfall put on the CHIRPS grid. CHIRPS covers land, 50°S–50°N, and runs about
  a month behind; days missing before the forecast are flagged on the page.
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
the CHIRPS grid with CHIRPS, or daily maximum 2 m temperature with ERA5 on request; for Onset,
each model's onset share next to the observed one); forecast maps and the NetCDF download (pick
the model).

## Saved runs from the bucket

Saved runs can be loaded from the program's bucket,
`gs://aim4scale_training_2026/demo3/<model>/` (same file names as `outputs/`). The Sparks can only
read it, so runs made on a Spark stay on it.

- **Start the container:** `bash docker/run.sh` uses that bucket by itself (add `DEV=1`,
  `DEMO_CACHE`, `DEMO_OUTPUTS` as usual). `DEMO3_BUCKET=gs://...` uses another one, and
  `DEMO3_BUCKET=` none.
- **Credentials:** the bucket is private. On a Spark there is nothing to set up: `run.sh` passes the
  machine's read-only key (`/etc/fleet-gcs-key.json`) into the container. Elsewhere:
  `gcloud auth application-default login --no-launch-browser` (tick every permission on Google's
  consent page), and `run.sh` passes the resulting
  `~/.config/gcloud/application_default_credentials.json` into the container. A service-account key
  instead: `DEMO3_GCP_KEY=<key.json>`.
- **On the page:** a run in the bucket but not on this Spark shows **Download saved run**; the
  results say whether the run is in the bucket.
- **In bulk:** `python apps/forecasting/runners/sync_bucket.py pull [--models ...] [--dry-run]` (inside the container), or
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
2. Make sure the model's environment exists at `/opt/envs/<env>/bin/python` inside the container (`docker/Dockerfile`).
3. Add the model to `MODELS` in `catalog.py`; it runs live once its environment exists.
4. Add its measured runtime to `timings.json`.
