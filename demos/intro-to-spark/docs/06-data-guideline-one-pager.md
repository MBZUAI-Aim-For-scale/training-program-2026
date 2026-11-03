# One-pager: the data side of running AI weather models

The model is the easy part. A daily AI forecast service needs four kinds of data, on time and in the right form.
Numbers marked *measured* come from [Lab 12](../labs/5-data-and-climate/lab12_data_for_ai_weather.ipynb) on one DGX Spark (2026-11-02).

## The four kinds of data

| Data | What it is | Where from | Size | How often |
|---|---|---|---|---|
| **Initial conditions** | Today's state of the whole atmosphere, the model's input | GFS (NOAA), IFS open data (ECMWF) | 62 MB downloaded per SFNO run, 73 fields (*measured*) | Every run (up to 4 cycles a day) |
| **Model weights** | The trained network | Earth2Studio packages (NGC, Hugging Face) | 1–5 GB per model (*measured*: AIFS-ENS 0.9, SFNO 2.1, Aurora 4.7) | Once, then on upgrade |
| **History** | Reanalysis (ERA5) and past forecasts, for verification, climatology, bias correction | ARCO ERA5, WeatherBench 2, your archive | 10s of GB, grows with every hindcast (18 GB on our box) | Pulled as needed, cached |
| **Observations** | Station, rain-gauge, satellite data | GHCN-Daily, national met service, GOES/Meteosat | Small, but scarce: **10 stations per M km² in Africa vs 240 in Europe** (*measured*) | Daily |

## Guidelines

**Before you build**
1. **Match model to data source.** Compare `model.input_coords()["variable"]` with the source's lexicon. *Measured:*
   GFS cannot feed AIFS-ENS (missing soil and skin temperature, total column water); IFS can feed all three
   models we tested. A mismatch otherwise fails minutes into a run.
2. **Pick the cycle from your deadline.** *Measured:* the newest GFS cycle was 4.7 h old, and a cycle's folder appears
   before all its files. Rule: start from the cycle before the newest. For a 07:00 EAT bulletin, the 18 UTC cycle
   of the day before, plus the run time (1–12 min on Spark), leaves margin.
3. **Have a fallback source** (GFS ↔ IFS; IFS mirrors `google`/`azure`/`ecmwf`; AWS throttled us), and keep the last
   good forecast to issue if both fail.
4. **Check the licence** of each data source and each set of model weights before you publish a product; they differ.

**Daily operations**
5. **Cache on the data disk**, shared by all users: set `EARTH2STUDIO_CACHE` (and `HF_HOME` for LLMs) to a mounted
   folder. Once weights are cached, a run needs only the initial conditions from the internet.
6. **Save only your region and variables** (`output_coords`). *Measured:* the Lab 04 Africa ensemble is 362 MB/run
   (129 GB/year); the same for the globe would be ~4.5 GB/run (1.6 TB/year).
7. **Clean the reanalysis cache** or use `cache=False` for one-off hindcasts; it is what grows.

**Formats and conventions**
8. **Store forecasts as compressed, chunked Zarr; chunk for your most common read.** *Measured:* the matching chunk
   layout read 10–30x faster than the mismatched one; compression saved 3x disk.
9. **Wrap the conventions once in a helper and reuse it:** longitude 0–360 → −180–180; latitude north→south;
   K, Pa, m, J/m² → °C, hPa, mm, kWh/m²; accumulations (`tp06`, `ssrd06`) cover the 6 h *before* each step, so skip
   lead 0; model times are UTC.

**Observations**
10. **Your station archive is the asset only you have.** Use it to verify (Lab 03) and bias-correct (Lab 11: per-station
    correction cut daily-max error from 2.2 to 0.8 °C). On a Spark it can be used without leaving the building.
11. **Record what you ran:** model version, data source and cycle, container tag. A forecast you cannot reproduce cannot
    be verified.

## Sizing rule of thumb (one country, one Spark)

| Item | Per day | Per year |
|---|---|---|
| Initial conditions (1–4 cycles) | under 1 GB, can be deleted after the run | — |
| Regional forecast output (ensemble, a few variables) | ~0.4 GB | ~130 GB |
| Model weights (3–4 models) | — | ~10 GB, one-off |
| Reanalysis / hindcast cache | — | 20–100 GB |
| **Disk to plan for** | | **~0.5 TB, with room to grow** |
