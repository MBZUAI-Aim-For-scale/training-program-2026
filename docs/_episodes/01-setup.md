---
title: "Demo 1"
teaching: 30
exercises: 15
questions:
- "What are the essential components of an AI weather forecasting environment?"
- "How do we verify GPU availability and configure containerized workflows?"
- "What runtime configuration options are available for different deployment scenarios?"
- "How do we configure model-specific environments and data sources?"
objectives:
- "Understand the environmental requirements for AI weather forecasting on DGX Spark infrastructure."
- "Verify GPU visibility and configure Docker with NVIDIA Container Device Interface (CDI)."
- "Configure runtime options for development, caching, and data source selection."
- "Set up model-specific environments and understand their dependencies."
- "Troubleshoot common setup issues related to GPU access and container initialization."
keypoints:
- "GPU visibility must be verified before running AI weather models in containers."
- "Docker rootless mode with NVIDIA CDI provides secure GPU passthrough for AI workloads."
- "Runtime configuration flags control development mode, cache locations, and data sources."
- "Model weights are downloaded once and cached; configuration determines storage location."
- "Essential libraries include earth2grid, NATTEN, and model-specific dependencies."
- "Pre-built container images can significantly reduce setup time for new deployments."
---

<!-- MathJax -->
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Setting Up the AI Weather Forecasting Lab

This lesson documents the complete environmental setup and configuration for AI weather forecasting on DGX Spark infrastructure. It is the single source of truth for all configuration — covering GPU verification, container configuration, runtime options, model-specific environments, FGN integration, and the ROMP/MOMP benchmarking pipeline. It is required reading before [Demo 3: Running AI Weather Forecasts](/training-program-2026/03-forecast) and [Demo 4: AI Weather Model Scorecard](/training-program-2026/04-weather-model).

> #### Prerequisites
>
> - Access to a DGX Spark node (e.g., `country-spark`) via Tailscale.
> - A user account on the Spark with sudo privileges for Docker configuration.
> - Familiarity with the Linux command line and basic Docker concepts.
{: .prereq}

## Learning Objectives

By the end of this lesson, you will be able to:

- Verify GPU availability using NVIDIA Container Device Interface (CDI).
- Configure Docker rootless mode with proper GPU passthrough.
- Build and deploy containerized AI weather forecasting environments.
- Configure runtime options for development, caching, and data sources.
- Understand model-specific environment requirements and dependencies.
- Identify and resolve common setup issues related to GPU access.

## Lesson Roadmap

| Section | Purpose |
| --- | --- |
| Why Environment Setup Matters | The importance of reproducible AI forecasting environments |
| GPU Verification | Confirming GPU visibility via CDI |
| Docker Configuration | Setting up rootless Docker with GPU support |
| Container Image Management | Building, caching, and transferring container images |
| Runtime Configuration | Environment variables and deployment options |
| Model-Specific Environments | Understanding dependencies for different AI models |
| FGN Configuration | Special setup for WeatherNext 2 |
| ROMP/MOMP Configuration | Benchmarking pipeline setup |
| Essential Libraries | Understanding dependencies for AI weather models |
| Troubleshooting | Resolving common setup issues |
| Best Practices | Checklists and performance optimization |
| Exercises | Hands-on practice with environment verification |

## Why Environment Setup Matters

AI weather forecasting models require specialized hardware (GPUs) and software environments to run efficiently. Unlike traditional weather models that run on CPUs, modern AI models like GraphCast, Aurora, and AIFS leverage GPU acceleration for:

- **Fast inference**: 10-day global forecasts in minutes instead of hours.
- **Ensemble generation**: Running multiple forecast scenarios simultaneously.
- **High-resolution predictions**: 0.25° grid resolution with complex neural architectures.

A reproducible environment ensures that:

- Results can be replicated across different DGX Spark nodes.
- Model weights and dependencies are consistent.
- GPU resources are properly allocated and monitored.

## 1. GPU Verification

The DGX Spark systems run Docker in rootless mode with NVIDIA's Container Device Interface (CDI). This provides secure, user-level GPU access without requiring root privileges.

### 1.1 Verify GPU Visibility

Confirm that Docker can access the GPU by running:

```bash
docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi -L
```

**Expected output:**

```
GPU 0: NVIDIA GB10 (UUID: GPU-xxxxx)
```

> #### Common Issue: Unresolvable CDI Devices
>
> If the command fails with `unresolvable CDI devices nvidia.com/gpu=all`, Docker started before the GPU registry was populated. This commonly occurs immediately after a system reboot.
{: .warning}

**Solution:** Restart the Docker service:

```bash
systemctl --user restart docker
```

Then retry the GPU verification command.

If it instead fails with `failed to fulfil mount request: open /usr/lib/aarch64-linux-gnu/libnvidia-...: no such file or directory`, the NVIDIA libraries were updated since the Spark booted. Restarting Docker doesn't help; an admin has to refresh the GPU registry (`sudo systemctl start nvidia-cdi-refresh.service`, then `systemctl --user restart docker`) or reboot the Spark.

### 1.2 Preventative Configuration

To prevent this issue from recurring after every reboot, configure Docker to wait for the GPU registry:

```bash
mkdir -p ~/.config/systemd/user/docker.service.d
cat > ~/.config/systemd/user/docker.service.d/wait-for-gpu.conf <<'EOF'
[Service]
ExecStartPre=/bin/sh -c "for i in $(seq 120); do [ -s /var/run/cdi/nvidia.yaml ] && exit 0; sleep 1; done; exit 0"
EOF
systemctl --user daemon-reload
```

> #### Why This Works
>
> The configuration adds a pre-start hook that waits up to 120 seconds for the CDI registry file (`/var/run/cdi/nvidia.yaml`) to appear before starting Docker. This ensures GPU devices are properly registered before Docker attempts to use them.
{: .tip}

## 2. Docker Configuration

### 2.1 Understanding Rootless Docker

Rootless Docker runs the Docker daemon as a non-root user, providing:

- **Enhanced security**: No root privileges required.
- **User isolation**: Each user manages their own containers.
- **Simplified permissions**: No need for sudo or group membership.

### 2.2 Container Storage Management

When running AI weather models, Docker manages two primary storage locations:

| Asset Type | Location | Size | Purpose |
| --- | --- | --- | --- |
| Saved forecasts, event movies, job logs | `training-program-2026/outputs/` | Grows with use | Persistent results |
| Model weights (downloaded on first run) | `~/.cache/ai_weather/` | Up to ~75 GB | Model parameters |

> #### Storage Best Practices
>
> - Model weights are downloaded once per model and cached locally.
> - Saved forecasts accumulate over time; consider periodic cleanup.
> - Use `DEMO_CACHE=/path` to redirect weights (and `DEMO_OUTPUTS=/path` for saved forecasts) to other storage.
{: .keypoints}

## 3. Container Image Management

### 3.1 Building the Container Image

The container image includes all dependencies for running AI weather models:

```bash
docker build -t ai_weather -f docker/Dockerfile .
```

**Build characteristics:**

- **First build**: ~1 hour (compiling earth2grid, NATTEN libraries).
- **Subsequent builds**: Seconds (using Docker layer cache), unless `docker/requirements.txt` or `docker/locks/` changed, which rebuilds from that step.
- **Image contents**: Python environments, dependencies, application code.
- **Excluded**: Model weights (downloaded on first run).

> #### Build Optimization
>
> Most of the build time is spent compiling two specialized libraries:
>
> - **earth2grid**: Grid interpolation for weather data.
> - **NATTEN**: Neighborhood attention for transformer architectures.
>
> These are compiled from source to ensure compatibility with the target GPU architecture.
{: .tip}

### 3.2 Transferring Pre-built Images

To avoid rebuilding on every Spark node, transfer a pre-built image:

```bash
# On the source Spark:
docker save ai_weather | gzip > ai_weather_image.tar.gz

# Transfer via scp/rsync over Tailscale:
scp ai_weather_image.tar.gz <user>@<target-spark>:~/

# On the target Spark:
docker load < ai_weather_image.tar.gz
```

> #### Exercise 1: Image Transfer (5 min)
>
> You have a pre-built image on Spark `country-spark` and need to deploy it to a new Spark `country-spark`. Write the complete sequence of commands to:
>
> 1. Export the image from the source Spark.
> 2. Transfer it to the target Spark.
> 3. Load it on the target Spark.
{: .exercise}



> #### Solution
>
> ```bash
> # On your country-spark:
> docker save ai_weather | gzip > ai_weather_image.tar.gz
> scp ai_weather_image.tar.gz <user>@country-spark:~/
>
> # On : your country-spark
> docker load < ai_weather_image.tar.gz
> ```
{: .solution}


## 4. Runtime Configuration

### 4.1 Core Setup Commands

```bash
# 1. Clone the repository
git clone https://github.com/MBZUAI-Aim-For-scale/training-program-2026.git
cd training-program-2026

# 2. Build the container image (~1 hour first time) and start it
bash docker/run.sh rebuild

# 3. Access the UI from your laptop (on Tailscale)
ssh -L 8501:localhost:8501 <user>@<spark>
# Then browse to http://localhost:8501 and choose Demo3 in the menu
```

On a Spark, `run.sh` also reads saved runs from the program's bucket (`gs://aim4scale_training_2026/demo3`) with the machine's read-only key, `/etc/fleet-gcs-key.json`, which it mounts into the container. Never copy that key into the image or the repository.

### 4.2 Runtime Configuration Flags

Prepend these flags to `bash docker/run.sh` to customize behavior:

| Flag | Purpose | Example | Default |
| --- | --- | --- | --- |
| `DEV=1` | Mount local code for live development | `DEV=1 bash docker/run.sh` | Disabled |
| `DEMO_CACHE=/path` | Redirect model weight cache location | `DEMO_CACHE=/shared/cache bash docker/run.sh` | `~/.cache/ai_weather/` |
| `DEMO_OUTPUTS=/path` | Redirect saved forecasts, movies and logs | `DEMO_OUTPUTS=/data/outputs bash docker/run.sh` | `training-program-2026/outputs/` |
| `DEMO3_BUCKET=gs://...` | Bucket to load saved runs from (empty: none) | `DEMO3_BUCKET= bash docker/run.sh` | `gs://aim4scale_training_2026/demo3` |
| `DEMO3_IFS_SOURCE=azure` | Use Azure mirror for ECMWF data | `DEMO3_IFS_SOURCE=azure bash docker/run.sh` | AWS |

> #### Development Workflow
>
> Use `DEV=1` during active development to avoid rebuilding the container after every code change. Changes to a page file show after a page refresh; changes to the modules it imports (e.g. `apps/forecasting/`) need `bash docker/run.sh restart`.
{: .tip}

> #### First-Run Downloads
>
> The first execution of each model downloads weights (~75 GB total). This can take 30+ minutes for large models like Atlas CRPS. Subsequent runs start in seconds.
{: .warning}

### 4.3 Pre-populating Cache and Outputs

Skip first-run downloads by copying weights and outputs from an existing Spark:

```bash
# Copy model weights
rsync -a --info=progress2 <user>@<spark>:<its weights cache>/ ~/.cache/ai_weather/

# Copy saved runs
rsync -a --info=progress2 <user>@<spark>:<its outputs folder>/ training-program-2026/outputs/
```

**Known cache locations:**

- On `hcwfpgx` and `rwanda-spark`: `~/e2s-spark/root_cache`

## 5. Model-Specific Environments

### 5.1 Environment Isolation

Different models require different Python environments, provisioned inside the container at `/opt/envs/<env>/bin/python`:

| Environment | Models | Key Dependencies | Peak GPU Memory |
| --- | --- | --- | --- |
| `e2s018` | Atlas CRPS, Aurora 1.5, AIFS v2 single | PyTorch, Earth2Studio | 14–33 GB |
| `e2s018ens` | AIFS v2 ENS | PyTorch, ensemble utilities | 25 GB |
| `graphcast` | GraphCast, NeuralGCM | JAX, Graph Neural Networks | 16–18 GB |
| `fgn` | FGN (WeatherNext 2) | JAX, Functional Generative Networks | 1.8 GB |
| `ui` | Streamlit interface | Web framework | N/A |

> #### Environment Isolation
>
> Each model runs in its own isolated Python environment. Do not attempt to mix dependencies across environments, as this can cause version conflicts.
{: .warning}

### 5.2 Model Weight Downloads

First-time executions download model weights:

| Model | Weight Size | Download Time | Notes |
| --- | --- | --- | --- |
| Atlas CRPS | ~30 GB | ~31 min | Largest model |
| Aurora 1.5 | ~7 GB | ~7 min | Multi-variable |
| AIFS v2 ENS | ~5 GB | ~5 min | Ensemble model |
| FGN Mini | 1.1 GB | ~2 min | Includes sample file |
| GraphCast | ~15 GB | ~10 min | Google DeepMind |
| NeuralGCM | ~12 GB | ~8 min | Hybrid model |
| AIFS v2 single | ~5 GB | ~5 min | Deterministic |

## 6. FGN (WeatherNext 2) Configuration

FGN requires special configuration due to its unique data requirements.

### 6.1 Starting Conditions

Google publishes FGN-ready inputs for only one date (2024-10-07 00Z). For other dates, Demo 3 uses a converter:

```bash
# Converter runs in e2s018 environment
python runners/fgn_convert.py --date 2024-10-08
```

**Converter workflow:**

1. Downloads ECMWF IFS analyses at 13 pressure levels.
2. Regrids to FGN's 1° resolution.
3. Constructs Google's expected file layout.

> #### Sea Surface Temperature Approximation
>
> The converter uses IFS skin temperature over oceans, floored at 271.46 K under sea ice. This introduces a small bias (correlation 0.997, mean error −0.08 K) compared to true SST.
{: .warning}

### 6.2 FGN-Specific Configuration

| Parameter | Default | Purpose |
| --- | --- | --- |
| FGN weights and sample file | `fgn/` in the weights cache (`~/.cache/ai_weather/fgn`) | Downloaded from Google on FGN's first run |
| Sample file date | 2024-10-07 | Google's published input (up to 7.5 days) |
| Converted input cache | `outputs/_fgn_inputs/` | 53 MB per start date, reused for subsequent runs |

### 6.3 FGN Performance

- **10-day, 3-member forecast**: ~2 minutes, 1.8 GB peak memory.
- **Model variant**: `WeatherNextCyclones_Mini` (1° resolution, the only Mini weights published).
- **Full 0.25° model**: Does not fit in Spark memory.

## 7. ROMP/MOMP Benchmarking Configuration

The ROMP (Rainy season Onset Metrics Package) pipeline has its own configuration system.

### 7.1 Installation

Install ROMP and its dependencies:

```bash
# From the ai_weather root directory
uv pip install pyproject.toml
```

This installs the `momp-run` command-line tool and all required dependencies for running onset benchmarks.

> #### Installation Verification
>
> After installation, verify that `momp-run` is available:
>
> ```bash
> momp-run --help
> ```
>
> This should display the command-line options for the ROMP runner.
{: .tip}

### 7.2 Configuration File Structure

ROMP uses configuration files (e.g., `notebooks/config_et.in`) to define:

| Parameter | Example Value | Purpose |
| --- | --- | --- |
| `wet_init` | 20 mm | Minimum initial rainfall for onset candidate |
| `wet_spell` | 3 days | Consecutive wet days required |
| `wet_threshold` | 1 mm/day | Daily rainfall threshold for "wet" |
| `dry_spell` | 7 days | Dry spell length that vetoes onset |
| `dry_threshold` | 1 mm/day | Daily rainfall threshold for "dry" |
| `dry_extent` | 20 days | Window after candidate for veto |
| `start_date` | June 1 | Season search window start |
| `end_date` | September 30 | Season search window end |
| `verification_window` | Days 1–15, Days 16–30 | Lead time windows to evaluate |
| `matching_tolerance` | 3 days (1–15), 5 days (16–30) | How close forecast must match observation |
| `run_mode` | `DET` or `PROB` | Deterministic or probabilistic track |

### 7.3 Model Registration

Models are registered centrally in `BENCHMARK_MODEL_CATALOG` within `config.py`:

| Model | Type | Resolution | Track |
| --- | --- | --- | --- |
| AIFS | Deterministic | 0.25° | DET |
| FuXi | Deterministic | 0.25° | DET |
| GraphCast | Deterministic | 0.25° | DET |
| AIFS-ENS | Probabilistic (50 members) | 0.25° | PROB |
| GenCast | Probabilistic (52 members) | 0.25° | PROB |

### 7.4 Data Sources

| Data Type | Sources | Resolution |
| --- | --- | --- |
| Observations | CHIRPS, ENACTS | 0.05°, 0.1° |
| Forecasts | Model reforecasts | 0.25° (common grid) |
| Seasonal mask | `jjas_seasonal_mask_0p25.nc` | 0.25° |

> #### Next Steps
>
> ROMP installation is complete. For detailed instructions on running benchmarks, configuring experiments, and interpreting results, see [Demo 4: AI Weather Model Scorecard](/training-program-2026/04-weather-model).
{: .tip}

## 8. Essential Libraries

### 8.1 Core Dependencies

AI weather models require specialized libraries for:

| Library | Purpose | Models Using It |
| --- | --- | --- |
| **earth2grid** | Grid interpolation and coordinate transformations | All models |
| **NATTEN** | Neighborhood attention for efficient transformers | GraphCast, NeuralGCM |
| **xarray** | Multi-dimensional array handling | All models |
| **PyTorch** | Deep learning framework | Aurora, AIFS, FuXi |
| **JAX** | High-performance numerical computing | GraphCast, NeuralGCM, FGN |
| **NetCDF4** | Weather data I/O | All models |
| **Streamlit** | Web interface framework | Demo 3 UI |

### 8.2 Library Versions

Pinned in `docker/locks/` for reproducibility. Key versions:

- PyTorch: 2.1+ (CUDA 12.x)
- JAX: 0.4+ (with CUDA support)
- xarray: 2023.x+
- earth2grid: Custom build
- NATTEN: Custom build

## 9. Troubleshooting

### 9.1 Common Issues

| Symptom | Likely Cause | Resolution |
| --- | --- | --- |
| `docker run` fails: `unresolvable CDI devices` | Docker started before GPU registry populated | `systemctl --user restart docker`, then `bash docker/run.sh restart` |
| `docker run` fails: `failed to fulfil mount request ... libnvidia-...: no such file` | NVIDIA libraries updated since boot | An admin runs `sudo systemctl start nvidia-cdi-refresh.service`, or reboot (section 1.1) |
| Port 8501 already in use when starting | Another container (e.g. the old stand-alone `demo3`) holds it | `docker ps`, then `docker stop <name>` and `bash docker/run.sh` |
| A code change doesn't show (with `DEV=1`) | The page keeps imported modules loaded | `bash docker/run.sh restart` |
| Browser can't reach `localhost:8501` | SSH tunnel not open or port conflict | Use `-L 8502:localhost:8501` and navigate to `:8502` |
| Run fails during starting condition download | ECMWF/Google mirrors throttled | Retry or use `DEMO3_IFS_SOURCE=azure` |
| First model run is exceptionally slow | Downloading model weights (~30 min for Atlas CRPS) | Expected behavior; subsequent runs are fast |
| ROMP run stops after switching tracks | DET and PROB mixed in one run | Set mode in config or with `--mode`, run separately |
| Model missing from ROMP output | Not registered for that track | Check `BENCHMARK_MODEL_CATALOG` in `config.py` |
| Many blank grid cells in ROMP maps | No onset detected or no matched events | Check onset thresholds, search window, mask, data coverage |
| Grid or shape errors when loading data | Forecast and observation grids differ | Ensure common grid, mask file, and resolution |

### 9.2 Diagnostic Commands

```bash
# Check container status
docker ps -a | grep ai_weather

# View container logs
docker logs -f ai_weather

# Check GPU allocation
nvidia-smi

# Verify Docker service
systemctl --user status docker

# Check disk space
df -h ~/.cache/ai_weather/
```

> #### Proactive Monitoring
>
> Run `docker logs -f ai_weather` in a separate terminal while testing to catch errors in real-time. This is especially useful during first-time model runs when weights are being downloaded.
{: .tip}

## 10. Everyday Maintenance Commands

```bash
# Monitor real-time application logs
docker logs -f ai_weather

# Gracefully restart the application container
docker restart ai_weather

# Fetch latest code: restarts with DEV=1, otherwise rebuilds and restarts
bash docker/run.sh update

# Full rebuild and restart cycle
bash docker/run.sh rebuild

# Clean up old containers
docker system prune -f

# Check container resource usage
docker stats ai_weather
```

## 11. Best Practices

### 11.1 Environment Setup Checklist

Before running AI weather models, verify:

- [ ] GPU is visible to Docker (`nvidia-smi -L` works in container)
- [ ] Docker service is running (`systemctl --user status docker`)
- [ ] Container image is built or loaded (`docker images | grep ai_weather`)
- [ ] Sufficient disk space for model weights (~75 GB)
- [ ] SSH tunnel is active (if accessing remotely)
- [ ] Network connectivity to ECMWF/Google data sources
- [ ] ROMP config file is properly configured (if running benchmarks)

### 11.2 Performance Optimization

- **Use pre-built images**: Transfer images between Sparks instead of rebuilding.
- **Cache model weights**: Use `DEMO3_CACHE` to point to shared storage.
- **Pre-run heavy models**: Aurora 1.5 (~26 min) and Atlas CRPS (~19 min) should be pre-run for classroom demonstrations.
- **Monitor GPU utilization**: Use `nvidia-smi` to ensure models are using GPU acceleration.
- **Separate DET and PROB tracks**: Never mix deterministic and probabilistic evaluations.

### 11.3 Reporting Checklist

Every benchmark result should document:

- [ ] Onset definition parameters (for ROMP)
- [ ] Verification dataset and resolution
- [ ] Model, reforecast period, initialization dates, number of members
- [ ] Lead-time windows and matching tolerance
- [ ] Reference used for skill scores
- [ ] Track (DET or PROB), and whether forecasts were calibrated
- [ ] Package version and config file used

## Summary

> #### Key Points
>
> - GPU visibility must be verified before running AI weather models in containers.
> - Docker rootless mode with NVIDIA CDI provides secure GPU passthrough.
> - Runtime configuration flags control development mode, caching, and data sources.
> - Model weights are downloaded once and cached locally.
> - Model-specific environments ensure dependency isolation.
> - FGN requires special configuration for starting conditions.
> - ROMP/MOMP has its own configuration system for benchmarking.
> - Essential libraries include earth2grid, NATTEN, and model-specific frameworks.
> - Troubleshooting common issues requires understanding Docker, GPU, and network interactions.
> - Always document configuration choices with benchmark results.
{: .keypoints}

## Exercises

> #### Exercise 2: Diagnose GPU Access (10 min)
>
> You SSH into a DGX Spark after a reboot and try to run a container, but get the error:
>
> ```
> docker: Error response from daemon: unresolvable CDI devices nvidia.com/gpu=all
> ```
>
> 1. What caused this error?
> 2. What command do you run to fix it immediately?
> 3. What configuration change prevents this from happening after future reboots?
{: .exercise}

<details>
<summary><strong>Check your answer</strong></summary>
<br>

> #### Solution
>
> 1. **Cause**: Docker started before the NVIDIA CDI registry was populated after the reboot.
> 2. **Immediate fix**: `systemctl --user restart docker`
> 3. **Prevention**: Create `~/.config/systemd/user/docker.service.d/wait-for-gpu.conf` with the ExecStartPre hook that waits for `/var/run/cdi/nvidia.yaml`.
{: .solution}

</details>

> #### Exercise 3: Container Management (5 min)
>
> You need to update the code in your running container without rebuilding the entire image. What environment variable do you set, and what command do you run?
{: .exercise}

<details>
<summary><strong>Check your answer</strong></summary>
<br>

> #### Solution
>
> Set `DEV=1` to mount the local directory into the container:
>
> ```bash
> DEV=1 bash docker/run.sh
> ```
>
> This allows you to edit code locally and see changes immediately without rebuilding.
{: .solution}

</details>

> #### Exercise 4: Configure ROMP Benchmark (15 min)
>
> You want to run a deterministic benchmark for Ethiopia's Kiremt season with the following requirements:
>
> - Wet spell: 3 consecutive days ≥ 2 mm/day
> - Dry spell veto: 7 consecutive days < 1 mm within 20 days
> - Verification windows: Days 1–15 and Days 16–30
> - Models: AIFS, GraphCast
>
> Write the configuration parameters you would set in `config_et.in`.
{: .exercise}

<details>
<summary><strong>Check your answer</strong></summary>
<br>

> #### Solution
>
> ```
> wet_init = 20
> wet_spell = 3
> wet_threshold = 2
> dry_spell = 7
> dry_threshold = 1
> dry_extent = 20
> start_date = 2015-06-01
> end_date = 2022-09-30
> verification_windows = [1-15, 16-30]
> matching_tolerance = {1-15: 3, 16-30: 5}
> run_mode = DET
> models = [AIFS, GraphCast]
> ```
{: .solution}

</details>

## Next Steps

After completing this setup, proceed to:

- **[Demo 3: Running AI Weather Forecasts](/training-program-2026/03-forecast)** — Learn how to run AI weather models and generate forecasts.
- **[Demo 4: AI Weather Model Scorecard](/training-program-2026/04-weather-model)** — Learn systematic benchmarking with ROMP/MOMP.

