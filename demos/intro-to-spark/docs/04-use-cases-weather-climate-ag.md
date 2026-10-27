# Use cases: weather, climate, agriculture

Each row maps a decision to the forecast variable, a model family, and what Spark contributes.
Variable names follow Earth2Studio conventions (`t2m`, `u10m`, `v10m`, `tp`, ...).

## Weather services

| Use case | Variables | Model families | Spark role |
|---|---|---|---|
| Medium-range deterministic forecast (1–10 d) | t2m, u10m, v10m, z500, msl | SFNO, FourCastNet-class, Pangu, GraphCast-class, Aurora, AIFS | Run locally on national data |
| Ensembles for uncertainty | same | AIFS-ENS, perturbed-IC SFNO | Run N members, compute probabilities |
| Benchmarking vs ERA5 / NWP | RMSE, ACC, CRPS | Earth2Studio statistics | Reproducible local scorecards |
| Nowcasting convective storms (0–6 h) | radar/satellite fields | StormScope, StormCast | Test feasibility on a regional domain |
| Regional downscaling | t2m, winds, precip | CorrDiff | Test for Gulf/Arabian Peninsula domain |

## Climate

| Use case | Notes |
|---|---|
| Seasonal outlooks | Coupled/seasonal models (notebook 12). Mostly a data and verification workflow |
| Extremes statistics (heat days, dust) | Post-process ensembles; counts and return-period thresholds |

## Agriculture (the training's Ag track)

| Decision | Forecast information | Threshold example (set locally) |
|---|---|---|
| Irrigation scheduling | t2m, wind, humidity -> reference ET | ET0 above crop-specific demand |
| Heat stress on crops/livestock | Daily max t2m | e.g. > 40–45 °C (crop dependent) |
| Spraying windows | 10 m wind speed | e.g. wind < 3–5 m/s |
| Frost protection | Daily min t2m | e.g. < 2 °C |
| Planting/harvest timing | Rain probability, soil moisture proxies | Locally calibrated |
| Dust and wind damage | Wind gusts | Locally calibrated |

Thresholds above are placeholders for discussion, not agronomic advice. Country teams set their own.

## The forecast-to-decision chain (matches the curriculum)

```
AI model forecast -> variables & thresholds -> impact-based info -> advisory message -> channel -> adoption
   (Lab 03)              (Lab 05)                (Lab 05)           (Ag track)
```

Labs 03 and 05 walk the first three links on Spark. Message design and dissemination are the Ag track's job.
