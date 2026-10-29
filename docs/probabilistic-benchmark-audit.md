# Probabilistic benchmark configuration audit

## Confirmed problems and fixes

| Problem | Evidence and correction |
| --- | --- |
| Precipitation converted twice | Local `aifs_ens/2000.nc` declares `tp.units=mm`; the model catalog previously requested a multiplier of 1000. Its processed-data catalog multiplier is now 1. The shared converter automatically uses declared units: m ×1000, cm ×10, mm and kg m-2 ×1. Metadata takes precedence without override warnings. Only unlabeled files use an explicit catalog fallback. A fallback of 1 logs the assumed mm units at INFO; scaling fallbacks still warn. Unsupported units and rates raise an error. A real sample remains 26.2954 mm instead of becoming 26,295.4 mm. |
| Selected window overwritten | Dashboard advanced fields always sent both windows, even after choosing just 1–15 or 16–30. Advanced defaults now follow the selected window. Editing the list to include 16–30 automatically raises the minimum forecast horizon to 30, including stale session values. Tolerance defaults follow the edited windows, and malformed or inconsistent lists disable launching with an inline explanation. Narrowing a window preserves the configured bins for subsequent overrides; final runner validation still checks horizon, tolerances and complete, nonoverlapping bins. |
| Parameter sources disagree | `params.py` disabled probabilistic flags by default, while `params/config.in` enabled them even with `probabilistic=False`. Their probability metric/plot defaults now follow the configured mode, with explicit overrides preserved. The template's default model list now matches its single directory. |
| Reliability flag lost | `Reliability` was not a `Setting` dataclass field, so it disappeared before case configuration. It is now retained and respected alongside `plot_reliability`. |
| Reliability event definition | Five-day pairs were pooled under 15-day titles. Pair generation now also computes the full requested event interval; reliability accepts only exact 1–15 or 16–30 pairs. Added aggregate bins are excluded from ordinary binned scores. |
| Incomplete RPS distribution | RPS used only in-window bins, discarding the before/after categories. Both forecast and reference RPS now use the complete disjoint partition, excluding the overlapping reliability aggregate. |
| Inconsistent climatology member count | Climatology probabilities use valid onset years, but Fair RPS used nominal years. Its correction now uses the effective count, matching the probability denominator. |
| Missing members counted as non-events | Missing rainfall and selection exceptions became `onset_day=None`, contributing false no-onset outcomes. Members with nonfinite values in the required detection horizon are excluded; extraction errors surface explicitly. Cases need at least two usable members. Empty output fails with a specific data error. |
| Undefined skill scores | Zero baseline Brier/RPS scores caused division errors or infinities. Undefined overall and per-bin skill now returns NaN. |
| Undefined skill crashes bar plots | The bar plot included NaN RPSS and dimensionless AUC in percentage-axis limits. Limits now use finite BSS/RPSS only, with a finite fallback when all scores are missing. Undefined values are labeled N/A. AUC uses its full 0–1 range; reference markers are drawn only for valid values. Axis bounds include every model row and positive/negative skill without fixed clipping. |
| CSV toggle breaks plots | Panel summary variables were defined only when CSV saving was enabled. Summaries are now built when panels need them without forcing CSV writes. Reliability-only runs can bypass unrelated baseline calculations. |
| Concurrency toggle ignored | The runner stored `run_years_concurrently`, while the pipeline reads `parallel`. Final configuration maps the override to the consumed setting. |
| Broken scientific environment inherited | Streamlit launched workers with its own Python, which could be Anaconda's broken SciPy 1.16.3 (`_spropack` import failure). Dashboard workers now prefer the project's `.venv/bin/python`, retaining its symlink path so its site-packages are used, and check scientific imports before starting. An explicit `ROMP_PYTHON` path takes precedence; invalid overrides fail clearly. |

## Verification

- All 88 regression tests pass; the audit, unit-conversion, dashboard verification, skill-bar, runtime, clipping, portrait, contingency, figure-loading, map-configuration and all-model map regression files pass isolated Ruff checks.
- Final configuration now removes unused day bins only after all overrides are merged. Both the runner validator and `ROMPValidator` pass for GenCast and AIFS ENS with days 1–15, days 16–30, and both windows (six real configuration smoke checks). This resolves the older validator rejecting retained day-30 bins in a day-15 run.
- Real subprocess checks reproduce the Anaconda SciPy failure, pass with project SciPy 1.18.1, and successfully import the complete MOMP driver in the project environment.
- The exact reported NaN RPSS case renders successfully. Four saved skill-bar figures were regenerated from existing CSVs; no metrics were recalculated.
- Regression coverage uses synthetic probabilities, genuine onset detection and plot rendering. It checks disjoint RPS categories, reliability windows, missing members, zero baselines, unit conversion, configuration parity, and CSV-disabled summaries.
- Actual loader/case-building smoke checks pass for AIFS ENS and GenCast, each at 1–15 and 16–30. They verify member normalization, three matching five-day bins, probabilistic output flags and serial execution overrides.
- A sample read from the real AIFS ensemble NetCDF confirms declared-mm data is no longer multiplied by 1000.
- Changed Python modules compile. New regression files pass isolated Ruff checks. Repository-wide Ruff configuration has a pre-existing invalid `force_sort_within_sections` option.

## Changed files

- `run_momp.py`: final configuration validation; preserve bins across window overrides; map concurrency to `parallel`.
- `ai_weather/config.py`: GenCast's processed, unlabeled rainfall uses multiplier 1 instead of 1000. The supplied mean 2.7168 and maximum 267.7002 are consistent with mm; the previous conversion produced implausible daily rainfall. Explicit source-unit metadata still takes precedence.
- `ui/pages/short_medium_benchmarking.py`: advanced window, tolerance and horizon defaults follow the selected lead window.
- `apps/benchmarking/ROMP/momp/params.py` and `params/config.in`: consistent probability defaults.
- `apps/benchmarking/ROMP/momp/lib/convention.py`: retain the reliability metric flag in stage settings.
- `ai_weather/precipitation.py`: shared, repeat-safe unit conversion without magnitude heuristics.
- `apps/benchmarking/ROMP/momp/io/input.py`: both forecast loaders use shared unit conversion and close datasets on conversion errors; correct no-initialization exception type.
- `apps/benchmarking/ROMP/momp/io/output.py` and `app/bin_skill_score.py`: separate panel summaries from CSV writes; honor reliability flags.
- `apps/benchmarking/ROMP/momp/metrics/skill.py`: disjoint RPS partitions and full-window reliability pairs; bypass irrelevant baselines for reliability-only runs.
- `apps/benchmarking/ROMP/momp/stats/detect.py` and `stats/bins.py`: distinguish unavailable members from valid non-events and reject insufficient ensembles.
- `apps/benchmarking/ROMP/momp/stats/score.py`: effective climatology count, zero-baseline guards and window-neutral diagnostic labels.
- `apps/benchmarking/ROMP/momp/graphics/reliability.py`: close noninteractive figures after plotting.
- `tests/test_probabilistic_config.py` and `tests/test_probabilistic_scoring.py`: configuration and numerical regressions.
- `tests/test_precipitation_units.py`: unit aliases, mixed metadata/config multipliers, repeat conversion, unsupported rates, explicit fallbacks, and UI rainfall below 1 mm.
- `ui/utils/verification_settings.py` and `tests/test_dashboard_verification.py`: validate editable window/tolerance lists, normalize single pairs, derive the required horizon and test actual widget-rendering logic against stale values.
- `apps/benchmarking/ROMP/momp/graphics/panel_bar_skill.py` and `tests/test_skill_bar_plot.py`: finite axis bounds, missing-score annotations, unclipped AUC and multi-model rows, plus noninteractive figure cleanup.
- `ai_weather/runtime.py`, the dashboard worker launch sites and `tests/test_scientific_runtime.py`: explicit/project/current interpreter selection, scientific dependency checks and clear binary-import diagnostics.
- `ai_weather/map_extent.py`, `graphics/onset_map.py`, `utils/visual.py` and `tests/test_climatology_framing.py`: spatial maps center the displayed boundary rather than the wider rainfall grid. Ethiopia uses `ETH_Zones.geojson` derived from canonical `config.shpfile_dir` for overlay and margins. `CLIMATOLOGY_SHAPEFILE` is a derived alias, not a required renderer attribute, so an older loaded config module does not fail. A missing or unreadable Ethiopia source fails instead of substituting an unrelated boundary. Country margins are a fixed 8% of each geographic span (at least 0.25 degrees per side), independent of grid resolution or finite values. Explicit bounds take precedence; other regions retain data-extent fallback.
- `utils/land_mask.py` and `tests/test_geojson_boundary.py`: explicit GeoJSON/JSON paths now take precedence over cached shapefiles, and outlines are reprojected to EPSG:4326. The NetCDF explorer uses the same zones source for its overlay and margins. Verification against the real file confirms all 101 zones (111 exterior rings), with identical padded country bounds: longitude 31.792085–49.187955 and latitude 2.491565–15.760581.

The changes reuse the existing pair extractor and member/case pipeline. One shared conversion function replaces duplicated, unconditional multiplications; CSV serialization no longer controls whether panel data exists. Earlier map-framing changes in the working tree are separate from this audit.

The UI's AIFS normalization now uses the same converter. It no longer infers metres from a maximum below 1, which could inflate correctly labeled light rainfall. Conversion preserves the source array and returns mm metadata so a second call applies a multiplier of 1.

## Clipping, late-window metrics and figure loading

- `ai_weather/spatial.py` provides cached EPSG:4326 boundary frames/unions, masks grid points outside polygons, and clips rendered raster cell edges exactly to polygon boundaries and holes. Boundary caches invalidate when the source file changes.
- Climatology onset/metric renderers (`graphics/onset_map.py`, `graphics/maps.py`, `graphics/func_map.py`) and both spatial pipeline entry points now use these masks. Four cached climatology NetCDF files were atomically rewritten and verified to contain no finite values outside ETH_Zones.geojson. Seven saved climatology/portrait figures were regenerated from the cached datasets.
- `stats/benchmark.py` now checks observed onset against both bounds of the selected verification window. Forecasts before day 16 no longer disappear from 16–30-day contingency counts; they enter the appropriate false-negative or true-negative count. Per-row observation dates and existing timing tolerance semantics are preserved.
- `io/output.py` and `graphics/panel_portrait_error.py` retain all model/window columns and reduce only finite values. Undefined portrait cells are explicitly N/A, including wholly missing panels, with a footnote. Cached AIFS 16–30 MAE/FAR contain no finite values, while MR is available. Those missing historical numbers cannot be recovered from saved maps: rerun AIFS with the corrected event counting. MAE/FAR remain legitimately undefined if no in-window predictions exist.
- The dashboard's NetCDF Explorer frames every lat/lon map using the zones shapefile and clips climatology fields to it. The explorer scans/opens files only after the user enables it.
- Result tabs and map subtabs now use Streamlit state tracking and `.open` guards; inactive tabs do not run. The figure tab skips NetCDF metadata discovery, defaults to one figure, paginates galleries six at a time, caches display previews (up to 1600×1200 pixels, 24 entries), and defers original PNG reads until Download. Two redundant swipe-map renders were removed.
- New tests: `test_spatial_clipping.py`, `test_lazy_result_tabs.py`, `test_figure_loading.py`, `test_portrait_metrics.py`, and `test_onset_contingency.py`. Existing boundary/explorer tests were updated for shared clipping.
- Streamlit AppTest checks passed for initial page rendering, figure browsing, opt-in NetCDF selection, climatology DOY rendering, and the lazily selected climatology Skill view, without exceptions. A local switch to Benchmark Outputs took 0.17 seconds; enabling and selecting a climatology NetCDF map took 1.18 seconds across two reruns. These are local server smoke measurements, not a guarantee of browser latency. AppTest does not synthesize native tab clicks, so selected tabs were pinned in the rendering checks.
- Visual previews confirm blank exteriors/polygon holes and readable N/A portrait cells. Full historical AIFS metrics were not recomputed during this fix.

## Canonical map configuration

- `ai_weather/config.py` defines `CLIMATOLOGY_SHAPEFILE = join(shpfile_dir, "ETH_Zones.geojson")` immediately after `shpfile_dir`; renderers use `ai_weather.spatial.climatology_boundary_path(config)` to derive that path even when a cached config lacks the uppercase alias. An obsolete alias cannot override the configured directory.
- Map renderers and the NetCDF Explorer read the JJAS mask from `config.nc_mask`. Mask reads are cached by file modification time, standardize latitude/longitude coordinate names, align to the plotted grid, and do not extrapolate beyond mask coverage. An explicit runtime `nc_mask=None` still disables the mask for that run. The dashboard defaults its mask selector to the configured path, including custom paths, while retaining the explicit No mask option.
- The provided file is `/Users/mesfind/data/external/jjas_100mm_rainfall_mask_0p25.nc`, and the overlay resolves to `/Users/mesfind/data/external/shapefile/ETH_Zones.geojson`. Four saved climatology maps were regenerated using these configured resources. Full-country margins remain based on zones, even where JJAS cells are masked out.
- Mask values retain the existing **True/1 = keep** convention. The file's `low_rainfall_mask` name/description conflicts with its actual wet-area pattern: sampled CHIRPS 2003 JJAS rainfall is about 897.5 mm at 9°N, 39°E (mask=True) and 24.7 mm at 6°N, 45°E (mask=False). The mask file itself was not modified or inverted based on its descriptive metadata. Explicit keep-value overrides in the original mask API are respected for boolean inputs.
- `tests/test_map_configuration.py` covers a config without the new attribute, stale aliases, configured dashboard defaults, coordinate alignment, mask coverage, and explicit keep-value overrides. A real Streamlit AppTest removes the uppercase attribute, enables the NetCDF Explorer and successfully renders the climatology DOY map with the configured mask and overlay.

## Remaining limits

- GenCast source files still lack rainfall units metadata. The catalog now treats these processed files as mm, based on the supplied data statistics; this is an explicit dataset assumption, not universal unit detection. The loader records this non-scaling assumption as an INFO message, while unlabeled scaling fallbacks still warn. Raw files explicitly declaring m are still converted to mm. Negative values are preserved rather than silently clipped.
- No complete multi-year ensemble benchmark was run during this audit. Existing probabilistic CSV/PNG outputs predate these corrections and must be regenerated; they do not establish correctness of the new calculations.
- Excluding members with missing required-horizon rainfall changes effective ensemble sizes. Fair scores require at least two usable members. Missing forecasts are distinct from valid dry forecasts with no detected onset.
- RPS now includes outside-window categories, so regenerated values differ from historical outputs by design. Do not mix old and new scores in comparisons.
- Anaconda's global SciPy installation remains broken; no global packages were modified. Dashboard workers use the verified project environment. Start the dashboard from the repository root with `.venv/bin/python -m streamlit run ui/app.py --server.port 8502`. Set `ROMP_PYTHON` to an executable Python path only when intentionally selecting another scientific environment.

## Consistent boundaries for all model maps

- `graphics/maps.py`, `graphics/func_map.py` and dashboard map renderers now use configured ETH_Zones boundaries for every Ethiopian model, not only climatology. Model fields receive configured JJAS masking and exact polygon clipping. The default country frame is identical across models, resolutions, sparse results and missing metrics.
- `utils/visual.py` draws zone lines at z-order 10 with an explicit PlateCarree transform; dashboard overlays use the same layer priority. Overlays remain above rasters rather than being obscured by them. Selected administrative boundaries can be emphasized on top.
- `ai_weather/map_extent.py` now applies fixed 8% geographic padding with a minimum 0.25 degrees, replacing data-dependent/cell-size-dependent region padding. Unused extent calculations were removed from the model-panel renderer.
- Interactive all-missing maps retain the country boundary and show “No valid metric values” without inventing a color scale. Explicit extent overrides, deliberate No mask runs and selected subregion zooms remain supported.
- Added `tests/test_fixed_spatial_panels.py`, `tests/test_basemap_overlay.py` and `tests/test_dashboard_map_boundaries.py`; updated extent tests. Coverage includes differing grids, sparse and all-missing fields, visible overlays above raster layers, correct CRS transforms, JJAS exclusion and exact clipping.
- Real AIFS, FuXi, GraphCast and climatology map previews have identical bounds and visible zone overlays. Four real Streamlit AppTest views passed without exceptions, including the all-missing AIFS 16–30-day MAE case. Ten saved spatial figures were regenerated from existing datasets; no source forecasts or numerical metrics were recomputed.

## AIFS display names

- `ui/pages/short_medium_benchmarking.py` now displays AIFS Single and AIFS Ensemble without v2 in model overrides, map inference, tables, ordering and figure/map labels.
- `ai_weather/utils.py` legacy versioned model-key aliases now produce unversioned plotting labels. No active benchmark figure filenames contain v2.
- All 88 tests pass; direct model-label checks and a real dashboard render confirm the unversioned model options. Restarting Streamlit clears previously cached display labels.
