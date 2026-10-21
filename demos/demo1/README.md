# Demo 1: set up the Spark

Prepares a DGX Spark for the other demos: checks the machine, then builds and starts
a Docker container for each demo listed in `demos.toml`.

```bash
python3 demos/demo1-setup/spark_setup.py
```

It first starts a progress dashboard in the background and prints its link
(http://localhost:8500; from a laptop, `ssh -L 8500:localhost:8500 <spark>`). Then it works
through the steps, printing each one in the terminal while the dashboard shows the same
progress, flags errors, and suggests fixes.

```bash
python3 spark_setup.py --only demo3       # host checks + one demo
python3 spark_setup.py --rebuild          # rebuild images even if they exist
python3 spark_setup.py --host-only        # host checks only
python3 spark_setup.py status             # last run, with suggested fixes
python3 spark_setup.py dashboard [--stop] # start or stop just the dashboard
```

**Host checks:** GPU, Docker, GPU access from containers, the after-reboot GPU wait
(SETUP.md step 1, installed if missing), disk space, and a PyTorch + flash-attention
smoke test (the same test as `Untitled.ipynb`).

**For each demo:** build its image (only if it's missing), start its container (a stopped
one is recreated), and wait until its page answers.

The dashboard stays up after the script finishes. It keeps checking each container,
so it also flags crashed or restarting containers, pages that don't answer, port clashes,
and error lines in container logs. To add a demo, add a `[[demo]]` block to `demos.toml`.

`docker-compose.yml` still starts the JupyterLab sandbox (`docker compose up`, port 8888).
