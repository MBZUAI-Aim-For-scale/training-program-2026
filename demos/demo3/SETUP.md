# Running Demo 3 on a DGX Spark

Step-by-step setup for the Demo 3 Streamlit platform (AI forecast lab). Everything runs in
one Docker container on the Spark; you open the page in your laptop's browser through an
SSH tunnel.

There are two ways in:

- **A. Use a Spark where Demo 3 already runs** (e.g. `rwanda-spark`): only step 5 below.
  You need Tailscale access to the Spark and an account on it.
- **B. Set up your own Spark:** steps 1–5. Budget about an hour for the first build,
  plus model downloads on first use (or copy them from a Spark that has them; step 4).

---

## 1. Check Docker can see the GPU

The Sparks run Docker rootless with NVIDIA's CDI device list. This should print the GPU:

```bash
docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi -L
```

If it fails with `unresolvable CDI devices nvidia.com/gpu=all`, Docker started before the
GPU list existed (common right after a reboot). Restart Docker and try again:

```bash
systemctl --user restart docker
```

To stop this happening after every reboot, make Docker wait for the GPU list:

```bash
mkdir -p ~/.config/systemd/user/docker.service.d
cat > ~/.config/systemd/user/docker.service.d/wait-for-gpu.conf <<'EOF'
[Service]
ExecStartPre=/bin/sh -c "for i in $(seq 120); do [ -s /var/run/cdi/nvidia.yaml ] && exit 0; sleep 1; done; exit 0"
EOF
systemctl --user daemon-reload
```

## 2. Get the code

```bash
git clone https://github.com/MBZUAI-Aim-For-scale/training-program-2026.git
cd training-program-2026/demos/demo3
```

## 3. Build the image (once, ~1 hour)

```bash
docker build -t demo3 -f docker/Dockerfile .
```

Most of the hour is compiling two libraries (earth2grid, NATTEN). Later rebuilds reuse
Docker's cache and take seconds unless `docker/locks/` changes. The image holds the page
and every model's Python environment; model weights are not inside it.

**Faster alternative:** copy the image from a Spark that already built it.

```bash
# on the Spark that has it
docker save demo3 | gzip > demo3-image.tar.gz
# copy the file over (scp/rsync over Tailscale), then on the new Spark
docker load < demo3-image.tar.gz
```

## 4. Start it

```bash
bash docker/run.sh
```

This starts a container named `demo3` that restarts by itself after reboots, and keeps:

| What | Where on the Spark | Size |
|---|---|---|
| Saved forecasts, event movies, job logs | `training-program-2026/demos/demo3/outputs/` | grows with use |
| Model weights (downloaded on each model's first run) | `~/.cache/demo3/` | up to ~75 GB |

Options (put them before `bash docker/run.sh`):

- `DEV=1`: use the code in this folder instead of the copy inside the image, so
  `git pull` (or editing a file) changes the page without a rebuild. Use this while developing.
- `DEMO3_CACHE=/path`: keep model weights somewhere else (e.g. an existing cache).
- `DEMO3_IFS_SOURCE=azure`: download ECMWF data from the Azure mirror if AWS is slow.

**Skip the first-run downloads and get the saved runs** by copying them from a Spark that
has them (replace `<user>@<spark>` and the source paths with the right ones):

```bash
rsync -a --info=progress2 <user>@<spark>:<its weights cache>/ ~/.cache/demo3/
rsync -a --info=progress2 <user>@<spark>:ai_weather/demos/demo3/outputs/ ai_weather/demos/demo3/outputs/
```

On `rwanda-spark` the weights cache is `~/e2s-spark/root_cache`.

## 5. Open the page

From your laptop (on Tailscale):

```bash
ssh -L 8501:localhost:8501 <user>@<spark>
```

Leave that terminal open and browse to **http://localhost:8501**.

### Trying it

1. Pick **Deterministic** or **Probabilistic**, then a model. The card shows its run time
   on the Spark and what it can't do (e.g. NeuralGCM has no 2 m temperature).
2. Pick a start date (the caption under it gives the model's allowed range and why),
   lead time, members, country and use case.
3. **Load saved run** appears when a matching forecast is already saved (instant);
   otherwise **Run forecast** runs the model on the Spark with a progress bar.
4. Explore the result tabs, and the **Event movie** tab for forecast vs observations.

Slow models (Aurora 1.5 ~26 min, Atlas CRPS ~19 min for 10 days) are best pre-run and
loaded in class.

---

## Everyday commands

```bash
docker logs -f demo3                 # page log
docker restart demo3                 # restart the page
git pull                             # update the code (live with DEV=1; otherwise rebuild)
docker build -t demo3 -f docker/Dockerfile . && docker rm -f demo3 && bash docker/run.sh
                                     # rebuild and restart with the new image
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker run` fails: `unresolvable CDI devices` | `systemctl --user restart docker`, then `docker start demo3` (see step 1 to prevent it) |
| `docker: Conflict ... name "/demo3" is already in use` | a container exists already: `docker start demo3`, or `docker rm -f demo3` and run again |
| Browser can't reach localhost:8501 | the SSH tunnel isn't open, or port 8501 is busy on your laptop: use `-L 8502:localhost:8501` and open :8502 |
| A run fails while downloading starting conditions | ECMWF/Google mirror busy: try again, or start with `DEMO3_IFS_SOURCE=azure` |
| First run of a model is slow | it is downloading the weights once (Atlas CRPS ~30 min); later runs start in seconds |
