# Lab 00 — Connect to your Spark (Setting Up Your AI Weather Lab)

**Goal:** get from your laptop to a running Jupyter on the Spark, securely. 15 min.
Your instructor gives you: the Spark hostname (reached over Tailscale or the local network), a username, and a port.

## 1. SSH in (key-based)

```bash
# once, on your laptop
ssh-keygen -t ed25519            # skip if you already have ~/.ssh/id_ed25519
ssh-copy-id <user>@<spark-host>  # or send your .pub key to the instructor
ssh <user>@<spark-host>          # should not ask for a password
```

Tip: add to `~/.ssh/config` on your laptop so you can type `ssh spark`:

```
Host spark
    HostName <spark-host>
    User <user>
```

## 2. Look around (on the Spark)

```bash
nvidia-smi                                                   # GPU, driver, running processes
free -h                                                      # unified memory: this is also your GPU memory
df -h ~                                                      # disk
docker ps                                                    # who else is running containers
docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi -L   # can containers see the GPU?
```

If the last command fails with `unresolvable CDI devices`, see [Demo 3 SETUP step 1](../../demo3/SETUP.md).

## 3. Get the labs and start the lab container (on the Spark)

```bash
git clone https://github.com/MBZUAI-Aim-For-scale/training-program-2026.git   # skip if already cloned
cd training-program-2026
docker build -t spark-intro demos/intro-to-spark/docker      # once per Spark, 30-60 min; skip if `docker images` lists it
bash demos/intro-to-spark/docker/run.sh <your-port>          # prints a token; keep this terminal open (or use tmux)
```

## 4. Open Jupyter through an SSH tunnel (on your laptop, new terminal)

```bash
ssh -N -L <your-port>:localhost:<your-port> spark
```

Open `http://localhost:<your-port>`, paste the token, and open `labs/lab01_know_your_spark.ipynb`.

## Why it's done this way

- Jupyter listens only on the Spark's localhost; the tunnel is the only way in, and it's encrypted.
- One container per person or team keeps sessions separate. Note it does **not** limit GPU memory (Lab 01).
- Your notebooks and outputs live on the Spark's disk (mounted into the container), and model weights (~35 GB)
  are cached in `~/.cache/spark-intro`, so nothing is lost when the container stops.

## If something fails

| Symptom | Fix |
|---|---|
| `Permission denied (publickey)` | Your key isn't on the Spark; redo `ssh-copy-id` |
| `bind: address already in use` | That port is taken; pick yours from the instructor's list |
| `Conflict. The container name ... is already in use` | You already have one running: `docker rm -f spark-intro-$USER-<port>` |
| `unresolvable CDI devices nvidia.com/gpu=all` | Docker started before the GPU was ready; see Demo 3 SETUP step 1 |
| Browser can't connect | Tunnel terminal closed; rerun step 4 |
