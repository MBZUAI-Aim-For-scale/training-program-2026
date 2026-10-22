# Running the AI forecasting page (Demo 3) on a DGX Spark

Step-by-step setup for the shared training site, where Demo 3 is the **AI Forecast** page
(`ui/pages/ai_forecasting.py`, with the code in `apps/forecasting/`). Everything runs in one
Docker container, `ai_weather`, on the Spark; you open the site in your laptop's browser
through an SSH tunnel.

For Demo 3 on its own (its own image and container, `demo3`, on port 8502), see
[demos/demo3/SETUP.md](../../demos/demo3/SETUP.md). The two share the same models and code.

There are two ways in:

- **A. Use a Spark where the site already runs:** only step 5 below. You need Tailscale
  access to the Spark and an account on it.
- **B. Set up your own Spark:** steps 1–5. Budget about an hour for the first build,
  plus model downloads on first use (or copy them from a Spark that has them; step 4).

---

## 1. Check Docker can see the GPU

The Sparks run Docker rootless with NVIDIA's CDI device list. This should print the GPU:

```bash
docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi -L
```

- `unresolvable CDI devices nvidia.com/gpu=all`: Docker started before the GPU list existed
  (common right after a reboot). Run `systemctl --user restart docker` and try again.
  [demos/demo3/SETUP.md](../../demos/demo3/SETUP.md#1-check-docker-can-see-the-gpu) shows how
  to stop this happening after every reboot.
- `failed to fulfil mount request: open /usr/lib/aarch64-linux-gnu/libnvidia-...: no such file or
  directory`: the NVIDIA libraries were updated since the Spark booted. Restarting Docker doesn't
  help; an admin has to run `sudo systemctl start nvidia-cdi-refresh.service` (then
  `systemctl --user restart docker`), or reboot the Spark.

## 2. Get the code

```bash
git clone https://github.com/MBZUAI-Aim-For-scale/training-program-2026.git
cd training-program-2026
```

Every command below runs from this top-level folder.

## 3. Build the image (once, ~1 hour)

```bash
docker build -t ai_weather -f docker/Dockerfile .
```

Most of the hour is compiling two libraries (earth2grid, NATTEN). Later rebuilds reuse
Docker's cache. The image holds the site and every model's Python environment; model weights
are not inside it.

**Faster alternative:** copy the image from a Spark that already built it.

```bash
# on the Spark that has it
docker save ai_weather | gzip > ai_weather-image.tar.gz
# copy the file over (scp/rsync over Tailscale), then on the new Spark
docker load < ai_weather-image.tar.gz
```

## 4. Start it

```bash
bash docker/run.sh
```

This starts a container named `ai_weather` that restarts by itself after reboots (it replaces
an existing `ai_weather` container), and keeps:

| What | Where on the Spark | Size |
|---|---|---|
| Saved forecasts, event movies, job logs | `training-program-2026/outputs/` | grows with use |
| Model weights (downloaded on each model's first run) | `~/.cache/ai_weather/` | up to ~75 GB |

Options (put them before `bash docker/run.sh`):

- `DEV=1`: use the code in this folder instead of the copy inside the image, so `git pull`
  (or editing a file) changes the site without a rebuild. Pages reload on the next browser
  refresh; edits to `apps/forecasting/` need `bash docker/run.sh restart`.
- `DEMO_OUTPUTS=/path`, `DEMO_CACHE=/path`: keep saved runs or model weights somewhere else
  (e.g. an existing cache).
- `DEMO3_IFS_SOURCE=azure`: download ECMWF data from the Azure mirror if AWS is slow.
- `DEMO3_BUCKET=gs://...`: where saved runs are downloaded from. By default it is
  `gs://aim4scale_training_2026/demo3`, the program's bucket, which also holds the ocean-wave
  inputs AIFS v2 needs for 2000–2025 starts; `DEMO3_BUCKET=` turns it off. On a Spark nothing
  else is needed: `run.sh` mounts the machine's read-only key, `/etc/fleet-gcs-key.json`, into
  the container. The key can't write, so runs made on a Spark stay on it. Never copy the key
  into the image or the repository. Elsewhere, see "Saved runs from the bucket" in
  [demos/demo3/README.md](../../demos/demo3/README.md#saved-runs-from-the-bucket).

**Skip the first-run downloads and get the saved runs** by copying them from a Spark that
has them (replace `<user>@<spark>` and the source paths with the right ones):

```bash
rsync -a --info=progress2 <user>@<spark>:<its weights cache>/ ~/.cache/ai_weather/
rsync -a --info=progress2 <user>@<spark>:<its outputs folder>/ outputs/
```

The saved-run files are the same for this site and for Demo 3 on its own, so either's
outputs folder works.

## 5. Open the site

From your laptop (on Tailscale):

```bash
ssh -L 8501:localhost:8501 <user>@<spark>
```

Leave that terminal open, browse to **http://localhost:8501** and choose **Demo3** (AI Forecast)
in the menu.

### Trying it

1. Open **Developer options** and turn **Use synthetic output** off; it is on by default,
   which makes fake forecasts for testing the page without a GPU.
2. **Model(s):** pick up to four models. Hover over a card for its run time on the Spark, what
   it starts from and what it can't do; the dropdown below lists what each needs on the Spark.
3. **Forecast setup:** start date (hover the ⓘ for each model's allowed range), lead time and
   members; then a country (or a custom box) and a use case: **Heat**, **Precipitation** or
   **Onset**.
4. **Run:** one button loads the saved runs that match (instant), downloads any that are in the
   bucket, and runs the rest on the Spark one after another, with progress.
5. **Results:** how each forecast was made; event movies (forecast next to observations, plus the
   event's track) for the models you choose, or onset maps and movies for Onset; a time-series
   comparison of all the models; and forecast maps with Play/Pause to step through the days.

Slow models (Aurora 1.5 ~26 min, Atlas CRPS ~19 min for 10 days) are best pre-run and
loaded in class.

---

## Everyday commands

```bash
bash docker/run.sh logs              # site log
bash docker/run.sh restart           # restart the site (also picks up code edits with DEV=1)
bash docker/run.sh update            # git pull, then restart (DEV=1) or rebuild and restart
bash docker/run.sh rebuild           # rebuild the image and start a new container
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker run` fails: `unresolvable CDI devices` | `systemctl --user restart docker`, then `bash docker/run.sh restart` (step 1) |
| `docker run` fails: `failed to fulfil mount request ... libnvidia-...: no such file` | NVIDIA libraries updated since boot: ask an admin to refresh the GPU list or reboot (step 1) |
| Browser can't reach localhost:8501 | the SSH tunnel isn't open, or port 8501 is busy on your laptop: use `-L 8502:localhost:8501` and open :8502 |
| The page shows fake forecasts | turn off **Use synthetic output** in Developer options |
| A code change doesn't show | `bash docker/run.sh restart` (needed for `apps/forecasting/`, even with `DEV=1`) |
| A run fails while downloading starting conditions | ECMWF/Google mirror busy: try again, or start with `DEMO3_IFS_SOURCE=azure` |
| First run of a model is slow | it is downloading the weights once (Atlas CRPS ~30 min); later runs start in seconds |
