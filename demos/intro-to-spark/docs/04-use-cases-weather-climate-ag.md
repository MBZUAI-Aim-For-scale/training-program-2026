# Use cases for Africa: weather, agriculture, energy, climate

Each row maps a decision to the Earth-2 pieces that serve it, and to the lab that shows it on a DGX Spark.
"Verified" means it ran on a Spark in this project (see the [one-pager](05-hardware-one-pager-spark-vs-cloud.md)).
Variable names follow Earth2Studio (`t2m`, `tp06`, `ssrd06`, ...). All thresholds in the labs are placeholders
for country teams to set.

## Agriculture

| Decision | Forecast information | Earth-2 pieces | Lab |
|---|---|---|---|
| When to plant | Probability of a wet spell; onset of the rains vs the 40-year normal | AIFS-ENS `tp06`; ERA5 climatology | 04, 10 |
| Heat damage at flowering (maize, wheat) | Days above 35 °C, with probability | AIFS-ENS `t2m` | 06 |
| Crop water stress, irrigation need | Vapour-pressure deficit, dry spells | AIFS-ENS `t2m`, `d2m`, `tp06`; `DerivedVPD` | 06 |
| Livestock heat stress | Temperature-humidity index | AIFS-ENS `t2m`, `d2m` | 06 |
| Spraying windows, drying harvest | Calm and dry days | AIFS-ENS winds and rain | 05 |
| Farmer SMS advisories | All of the above, in plain words, checked against the facts | Local LLM on the Spark | 05 |
| Correct forecasts for a specific farm or district | Station-specific correction | GHCN-Daily (or national stations) + ML | 11 |

## Disasters and early warning

| Decision | Forecast information | Earth-2 pieces | Lab |
|---|---|---|---|
| Cyclone landfall (Mozambique, Madagascar, Malawi) | Track 3–5 days ahead | SFNO, FourCastNet, Aurora from ERA5; IBTrACS truth | 07 |
| Heavy-rain / flood risk | Probability of large 5-day totals | AIFS-ENS `tp06`, runoff `ro06` | 04 |
| Heatwaves in cities | Daily max temperature, multi-model | SFNO, AIFS-ENS | 02, 06 |

## Energy

| Decision | Forecast information | Earth-2 pieces | Lab |
|---|---|---|---|
| Mini-grid battery / diesel planning | kWh per kWp per day, with a range | AIFS-ENS `ssrd06`; AIFS + `SolarRadiationAFNO6H` | 08 |
| Solar irrigation pumps | m³ of water per day | same | 08 |
| Wind power | 100 m wind | AIFS-ENS `u100m`, `v100m` | 08 (exercise) |

## Weather services: choosing and trusting models

| Task | Earth-2 pieces | Lab |
|---|---|---|
| Is the forecast better than "no change"? | ERA5 (ARCO), persistence baseline | 03 |
| Check against real stations | GHCN-Daily via Earth2Studio | 03, 11 |
| Pick a model for your region | Multi-model, multi-date scorecard | 09 |
| Build climate normals and anomalies fast | ERA5 (WeatherBench 2 copy), CuPy on the GPU | 10 |

## What does not fit (today)

- **StormScope / StormCast nowcasting:** trained only for the US (and Europe for StormScope). No African domain.
- **CorrDiff downscaling:** checkpoints exist for Taiwan, the US and Europe, not Africa.
- **GraphCast, GenCast** (JAX), **FuXi, Pangu** (ONNX Runtime GPU): not in the Spark container yet.
- **Meteosat (covers Africa):** needs EUMETSAT Data Store credentials.
- **Training a weather model from scratch:** data-centre scale. Fine-tuning small models and corrections (Lab 11)
  fits.

## The forecast-to-decision chain

```
AI forecast -> facts & thresholds -> impact (stress, energy, onset) -> checked message -> channel -> adoption
 (02, 04, 07)      (04, 06, 08)              (06, 08, 10)               (05)         (Ag track)
                          ^ trust: verification and station correction (03, 09, 11)
```
