# One-pager: Spark vs cloud for AI weather models

Curriculum item (Day 2): "hardware considerations (Spark vs cloud)".

| Consideration | DGX Spark (on-prem desktop) | Cloud GPU (H100/B200-class) |
|---|---|---|
| Up-front cost | One purchase | None |
| Running cost | Power only | Per-hour; adds up for daily operational runs |
| Memory for models | 128 GB unified | 80–192 GB HBM per GPU |
| Memory bandwidth | ~273 GB/s | ~3–8 TB/s |
| Throughput | Moderate | High; scales to many GPUs |
| Data sovereignty | Data never leaves site | Depends on region and agreements |
| Internet dependence | Needed to fetch initial conditions/weights, then runs offline | Constant |
| Ops burden | You maintain it | Provider maintains hardware |
| Scale-out | Link two units | Elastic |
| Best for | Learning, prototyping, small daily operational runs | Large ensembles, training, burst capacity |

## Decision rule of thumb

- One deterministic forecast per day, or a small ensemble, for one country: **Spark is enough** (confirm with your own timings: Lab 03, Demo 3).
- Large ensembles, high resolution downscaling at scale, or any training: **cloud or data-center GPUs**.
- Start on Spark, containerize, and move the same container to cloud when you outgrow it.

## Measured results

Measured on GB10 (driver 580.95, NGC PyTorch 26.06), from these labs (throughput read off the Lab 02 chart):

| Metric | Spark measured | Note |
|---|---|---|
| Matmul bf16 / fp16 (8192²) | ~35 TFLOPS | Lab 02 |
| Matmul tf32 / fp32 | ~18 / ~7 TFLOPS | Lab 02 |
| Copy bandwidth | ~230 GB/s | ~85% of the 273 GB/s spec |
| `docker --memory` caps GPU memory? | No | Lab 04 section 5 |
| SFNO 5-day t2m RMSE over UAE (init 2023-07-01) | ~1.0–1.3 °C | Beats "yesterday again" baseline at most leads; Lab 06 |

Still to fill in:

| Workload | Spark wall time | Peak memory | Cloud reference |
|---|---|---|---|
| 5-day SFNO forecast (Lab 03) | _TBD_ | _TBD_ | _TBD_ |
| Larger models, e.g. Aurora / AIFS (Demo 3) | _TBD_ | _TBD_ | _TBD_ |
| Downscaling (diffusion) | _TBD_ | _TBD_ | _TBD_ |
