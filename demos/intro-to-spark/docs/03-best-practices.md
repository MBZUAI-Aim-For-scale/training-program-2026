# Best practices on DGX Spark

## Environment

- **Use NGC containers** (`nvcr.io/nvidia/pytorch:<tag>-py3`, arm64) instead of pip-installing torch on the host.
  This avoids Blackwell/aarch64 wheel mismatches and keeps the host clean.
- **Pin versions** (container tag, earth2studio commit, natten wheel). Record them in the lab report.
- **Never `pip install torch`** over the container's build; it can replace a Blackwell-capable build with one that is not.
- Keep the model/data cache on the NVMe, not in the container layer: set `EARTH2STUDIO_CACHE` to a mounted dir.

## Memory (unified: CPU and GPU share 128 GB)

- `nvidia-smi` may not report a GPU memory total. Use `torch.cuda.mem_get_info()` and `free -h`.
- A big dataset in page cache counts against what the GPU can allocate. If a job OOMs with "free" memory
  showing as cache, drop caches: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`.
- Leave **10–15 GB** for the OS and Jupyter. Treat ~110 GB as your budget.
- Load models in bf16/fp16 where the model supports it; bandwidth, not capacity, is the bottleneck.
- Stream large inputs with xarray/dask chunks instead of loading whole years.

## Performance

- Bandwidth-bound work (large convolutions/attention over big grids) is where Spark is slowest vs HBM GPUs.
  Use mixed precision and avoid unnecessary copies.
- Warm up before timing (first call includes CUDA init and kernel compilation).
- Use `torch.cuda.synchronize()` around timers.
- Prefer one big batch over many small launches.

## Sharing one Spark between teams

- One container per team: `docker run --device nvidia.com/gpu=all --name team-uae --memory 30g --cpus 4 ...`. The `--memory` limit caps
  CPU-side RAM only. **It does not cap CUDA allocations**: a container limited to 8 GiB filled 16 GiB on the GPU
  (verified on GB10, driver 580.95, NGC PyTorch 26.06; Lab 04 section 5). One team's GPU job can therefore use
  memory the others need, so schedule heavy GPU jobs so teams don't overlap.
- Run Jupyter bound to `127.0.0.1` and reach it by SSH tunnel: `ssh -L 8888:localhost:8888 user@spark`.
  Do **not** run with an empty token on a shared network.
- Agree who runs heavy jobs when; use `nvidia-smi` and `docker stats` to see load.
- Heavy runs: use `tmux` or `nohup`, not a notebook tab that dies with your Wi-Fi.

## Data and reproducibility

- Keep forecasts as Zarr/NetCDF with the model name, version, init time and data source in the attributes.
- Write outputs to `outputs/`, never inside the container's writable layer.
- Commit the lab report (versions, timings, errors) next to the results.

## Security basics

- SSH keys only; per-user accounts; no shared passwords.
- Do not put API keys (NGC, CDS) in notebooks; use env vars or `.env` (git-ignored).

Next: [use cases](04-use-cases-weather-climate-ag.md).
