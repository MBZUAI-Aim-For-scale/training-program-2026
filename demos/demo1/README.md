# Demo 1: set up the Spark

**New to this? Follow the step-by-step [Beginner's Guide](../BEGINNER_GUIDE.md#demo-1--set-up-the-spark).**

Prepares a DGX Spark for the other demos: checks the machine, then builds and starts
a Docker container for each demo listed in `demos.toml`.

Run the site to launch all the demos:

```bash
python3 demos/demo1/spark_setup.py dashboard
```

It first starts a progress dashboard in the background and prints its link
(http://localhost:8500; from a laptop, `ssh -L 8500:localhost:8500 <spark>`). Then it works
through the steps, printing each one in the terminal while the dashboard shows the same
progress, flags errors, and suggests fixes.

```bash
python3 spark_setup.py --only demo3       # host checks + one demo
python3 spark_setup.py --build-only       # host checks + builds, start nothing (Launch starts a demo)
python3 spark_setup.py --rebuild          # rebuild images even if they're up to date
python3 spark_setup.py --host-only        # host checks only
python3 spark_setup.py status             # last run, with suggested fixes
python3 spark_setup.py dashboard [--stop] # start or stop just the dashboard
python3 country_data.py [--plan]          # download this Spark's country data (or just show its size)
bash shutdown.sh                          # stop any setup run, download, the dashboard and all containers
```

**Rebuilds:** an image is rebuilt when the code it's built from changed (after a `git pull`,
say), not on every run. Each build is labelled with a fingerprint of its committed sources,
its `watch` paths in `demos.toml` (by default its build context and Dockerfile); uncommitted
edits don't count. An image built before this, or copied from another Spark with
`docker save`/`load` and no label, is kept as it is until you run `--rebuild` once.
`--build-only` is for unattended setup, after a pull, for example: it builds what changed and
starts nothing, since only one demo runs at a time. A demo container that's already running
keeps its old image until it's stopped and launched again.

**Country data:** the button at the bottom of the Demo 1 page downloads this Spark's data from the
program's Google bucket, keeping going if you close the page. The country comes from the Spark's
name (`kenya-spark` is Kenya, `sierra-leone-spark` is Sierra Leone). It fetches the country's 312
files, one a year for 2000-2025, into `~/<country>_data` (such as `~/kenya_data`) with the same paths as the bucket
(`SPARK_DATA_DIR` changes the location):

- `CHIRPSv3_Country_0p25deg/<COUNTRY>/`
- `CHIRTS_Country_0p25deg/<Tmax|Tmin>/<COUNTRY>/`
- `Forecasts_Country_0p25deg/<AIFS_ENS_v2|AIFS_v2|graphcast>/<tp|t2m_max|t2m_min>/<COUNTRY>/`

Sizes range from about 2 GB (Rwanda) to about 269 GB (Chile), and it checks the disk first. Clicking
again skips files already downloaded. Nothing else in the bucket is downloaded.

**Host checks:** GPU, Docker, GPU access from containers, the after-reboot GPU wait
(SETUP.md step 1, installed if missing), disk space, read access to the program's Google
bucket (with the key at `/etc/fleet-gcs-key.json`; also `python3 verify_bucket_access.py`),
and a PyTorch + flash-attention smoke test (the same test as `Untitled.ipynb`).

**For each demo:** build its image (only if it's missing), start its container (a stopped
one is recreated), and wait until its page answers.

The dashboard stays up after the script finishes. It keeps checking each container,
so it also flags crashed or restarting containers, pages that don't answer, port clashes,
and error lines in container logs. To add a demo, add a `[[demo]]` block to `demos.toml`.

`docker-compose.yml` still starts the JupyterLab sandbox (`docker compose up`, port 8889).
