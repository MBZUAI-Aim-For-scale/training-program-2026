# One-pager: Spark vs cloud for AI weather models

Curriculum item (Day 2): "hardware considerations (Spark vs cloud)".

| Consideration | DGX Spark (on-prem desktop) | Cloud GPU (H100/B200-class) |
|---|---|---|
| Up-front cost | One purchase | None |
| Running cost | Power only | Per-hour; adds up for daily operational runs |
| Memory for models | 128 GB unified | 80–192 GB HBM per GPU |
| Memory bandwidth | ~273 GB/s spec; **230–232 GB/s measured** (Lab 01 device copy) | ~3–8 TB/s |
| Throughput | Moderate: **35–36 TFLOPS bf16, 18–19 tf32, 7 fp32** measured (Lab 01 matmul) | High; scales to many GPUs |
| Data sovereignty | Data never leaves site | Depends on region and agreements |
| Internet dependence | Needed to fetch initial conditions/weights, then runs offline | Constant |
| Ops burden | You maintain it | Provider maintains hardware |
| Scale-out | Link two units | Elastic |
| Best for | Learning, prototyping, small daily operational runs | Large ensembles, training, burst capacity |

## Decision rule of thumb

- One deterministic forecast per day, or a small ensemble, for one country: **Spark is enough** (confirmed: a
  5-day SFNO forecast takes under a minute, an 8-member AIFS-ENS ensemble for all of Africa about 11 min).
- Large ensembles, high resolution downscaling at scale, or any training: **cloud or data-center GPUs**.
- Km-scale generative nowcasting (StormScope) is **not practical on Spark**: about 9–13 min of compute per
  10-minute forecast step. Use it pre-run or on data-center GPUs.
- One heavy job at a time: the heaviest Earth2Studio workflows (large ensembles, CorrDiff) peak at 70–77 GiB, and `docker --memory` does not cap GPU use.
- Start on Spark, containerize, and move the same container to cloud when you outgrow it.

## Measured results (one DGX Spark, GB10, 2026-10-27/30)

Peak memory is system-wide unified memory, including ~10 GiB baseline. Times for the Earth2Studio workflows below
the labs include first-time model downloads. Cloud reference times were not measured in this project.

| Workload | Spark wall time | Peak memory | Cloud reference |
|---|---|---|---|
| 5-day deterministic SFNO forecast from GFS (Lab 02) | 42 s forecast incl. GFS fetch (1.5 min notebook) | 35 GiB (15 GiB GPU) | not measured |
| 8-member, 5-day AIFS-ENS ensemble, all of Africa (Lab 04) | 10.8 min ensemble (11.8 min notebook) | 63 GiB (36 GiB GPU) | not measured |
| English advisories for 5 cities from a local 8B LLM (Lab 05) | 2.8 min notebook (12 s generation, batched) | 38 GiB (16 GiB GPU) | not measured |
| Cyclone hindcast, 3 models x 6 days (Lab 07) | SFNO 56 s, AIFS 90 s, Aurora 5.6 min | 64 GiB (35 GiB GPU) | not measured |
| Model scorecard, 3 models x 4 dates x 5 days (Lab 09) | 26 min | 66 GiB | not measured |
| Rainfall onset, 41 years x 2,300 points (Lab 10) | CPU 445 ms, GPU 37 ms; 24 definitions 1.5 s | 8 GiB | not measured |
| Earth2Studio workflow: AIFS-ENS 4-day forecast | 11.4 min | 74 GiB | not measured |
| Earth2Studio workflow: AIFS-ENS 16-member + Aurora ensembles | 49.2 min | 77 GiB | not measured |
| Earth2Studio workflow: CorrDiff downscaling | 11.3 min | 69 GiB | not measured |
| StormScope GOES 3 km, one 10-min step (100 sampler steps) | ~13 min | ~15 GiB GPU | not measured |
| StormScope Meteosat EU, one 10-min frame (48 steps) | 9.2 min | 12 GiB GPU | not measured |
