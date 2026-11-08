# Running the AI Forecast page of Demo 4's site on a DGX Spark

Step-by-step setup for Demo 4's site, which has an **AI Forecast** page
(`demos/demo4/ui/pages/ai_forecasting.py`, with the code in `demos/demo4/apps/forecasting/`).
Everything runs in one Docker container, `ai_weather`, on the Spark; you open the site in the
Spark's web browser.

> **This is not the course's Demo 3.** In the course, Demo 3 is the standalone app in
> [`demos/demo3`](../../../demo3/SETUP.md): its own container (`demo3`), on port 8502, started
> from the Demo 1 dashboard.

There are two ways in:

- **A. Use a Spark set up with Demo 1:** launch **Demo 4** from the dashboard
  (http://localhost:8500), then go to step 5.
- **B. Set it up by hand:** steps 1–5. Budget about an hour for the first build,
  plus model downloads on first use (or copy them from a Spark that has them; step 4).

---

## 1. Check Docker can see the GPU

The Sparks run Docker rootless with NVIDIA's CDI device list. This should print the GPU:

```bash
docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi -L
```

- `unresolvable CDI devices nvidia.com/gpu=all`: Docker started before the GPU list existed
  (common right after a reboot). Run `systemctl --user restart docker` and try again.
  To stop this happening after every reboot, make Docker wait for the GPU list:

  ```bash
  mkdir -p ~/.config/systemd/user/docker.service.d
  cat > ~/.config/systemd/user/docker.service.d/wait-for-gpu.conf <<'EOF'
  [Service]
  ExecStartPre=/bin/sh -c "for i in $(seq 120); do [ -s /var/run/cdi/nvidia.yaml ] && exit 0; sleep 1; done; exit 0"
  EOF
  systemctl --user daemon-reload
  ```
- `failed to fulfil mount request: open /usr/lib/aarch64-linux-gnu/libnvidia-...: no such file or
  directory`: the NVIDIA libraries were updated since the Spark booted. Restarting Docker doesn't
  help; an admin has to run `sudo systemctl start nvidia-cdi-refresh.service` (then
  `systemctl --user restart docker`), or reboot the Spark.

## 2. Get the code

The code is already on a course Spark, in `~/training-program-2026`. Elsewhere:

```bash
git clone https://github.com/MBZUAI-Aim-For-scale/training-program-2026.git ~/training-program-2026
cd ~/training-program-2026
```

Every command below runs from this top-level folder.

## 3. Build the image (once, ~1 hour)

```bash
docker build -t ai_weather -f demos/demo4/docker/Dockerfile .
```

Most of the hour is compiling two libraries (earth2grid, NATTEN). Later rebuilds reuse
Docker's cache. The image holds the site and every model's Python environment; model weights
are not inside it.

**Faster alternative:** copy the image from a Spark that already built it.

```bash
# on the Spark that has it
docker save ai_weather | gzip > ai_weather-image.tar.gz
# copy the file over (scp or rsync), then on the new Spark
docker load < ai_weather-image.tar.gz
```

## 4. Start it

```bash
bash demos/demo4/docker/run.sh start
```

This starts a container named `ai_weather` (it replaces an existing `ai_weather` container),
and keeps:

| What | Where on the Spark | Size |
|---|---|---|
| Saved forecasts, event movies, job logs | `demos/demo4/outputs/` | grows with use |
| Model weights (downloaded on each model's first run) | `~/.cache/ai_weather/` | up to ~75 GB |

Options (put them before `bash demos/demo4/docker/run.sh start`):

- `DEV=1`: use the code in this folder instead of the copy inside the image, so `git pull`
  (or editing a file) changes the site without a rebuild. Pages reload on the next browser
  refresh; edits to `apps/forecasting/` need `bash demos/demo4/docker/run.sh restart`.
- `DEMO_OUTPUTS=/path`, `DEMO_CACHE=/path`: keep saved runs or model weights somewhere else
  (e.g. an existing cache).
- `DEMO3_IFS_SOURCE=azure`: download ECMWF data from the Azure mirror if AWS is slow.
- `DEMO3_BUCKET=gs://...`: where saved runs are downloaded from. By default it is
  `gs://aim4scale_training_2026/demo3`, the program's bucket, which also holds the ocean-wave
  inputs AIFS v2 needs for 2000–2025 starts; `DEMO3_BUCKET=` turns it off. On a Spark nothing
  else is needed: `run.sh` mounts the machine's read-only key, `/etc/fleet-gcs-key.json`, into
  the container. The key can't write, so runs made on a Spark stay on it. Never copy the key
  into the image or the repository. Elsewhere, see "Saved runs from the bucket" in [README.md](README.md#saved-runs-from-the-bucket).

Only one demo runs at a time on a Spark. Starting it this way doesn't stop the others for you:
stop them first (from the dashboard, or `docker stop <name>`).

**Skip the first-run downloads and get the saved runs** by copying them from a Spark that
has them (replace `<user>@<spark>` and the source paths with the right ones):

```bash
rsync -a --info=progress2 <user>@<spark>:<its weights cache>/ ~/.cache/ai_weather/
rsync -a --info=progress2 <user>@<spark>:<its outputs folder>/ demos/demo4/outputs/
```


## 5. Open the site

Open **http://localhost:8503** in the Spark's web browser and choose **AI Forecast** in the
menu.

### Trying it

1. **Model(s):** pick up to four models. Hover over a card for its run time on the Spark, what
   it starts from and what it can't do; the dropdown below lists what each needs on the Spark.
2. **Forecast setup:** start date (hover the ⓘ for each model's allowed range), lead time and
   members; then a country (or a custom box) and a use case: **Heat**, **Precipitation** or
   **Onset**.
3. **Run:** one button loads the saved runs that match (instant), downloads any that are in the
   bucket, and runs the rest on the Spark one after another, with progress.
4. **Results:** how each forecast was made; event movies (forecast next to observations, plus the
   event's track) for the models you choose, or onset maps and movies for Onset; a time-series
   comparison of all the models; and forecast maps with Play/Pause to step through the days.

Slow models (Aurora 1.5 ~26 min, Atlas CRPS ~19 min for 10 days) are best pre-run and
loaded in class.

---

## Everyday commands

```bash
bash demos/demo4/docker/run.sh logs              # site log
bash demos/demo4/docker/run.sh restart           # restart the site (also picks up code edits with DEV=1)
bash demos/demo4/docker/run.sh update            # git pull, then restart (DEV=1) or rebuild and restart
bash demos/demo4/docker/run.sh rebuild           # rebuild the image and start a new container
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker run` fails: `unresolvable CDI devices` | `systemctl --user restart docker`, then `bash demos/demo4/docker/run.sh restart` (step 1) |
| `docker run` fails: `failed to fulfil mount request ... libnvidia-...: no such file` | NVIDIA libraries updated since boot: ask an admin to refresh the GPU list or reboot (step 1) |
| Browser can't reach localhost:8503 | the `ai_weather` container isn't running: `docker ps -a \| grep ai_weather`, then launch Demo 4 from the dashboard or `bash demos/demo4/docker/run.sh start` |
| A code change doesn't show | `bash demos/demo4/docker/run.sh restart` (needed for `apps/forecasting/`, even with `DEV=1`) |
| A run fails while downloading starting conditions | ECMWF/Google mirror busy: try again, or start with `DEMO3_IFS_SOURCE=azure` |
| First run of a model is slow | it is downloading the weights once (Atlas CRPS ~30 min); later runs start in seconds |
