# Demo 1: set up the Spark

**New to this? Follow the step-by-step [Beginner's Guide](../BEGINNER_GUIDE.md#demo-1--set-up-the-spark).**

Prepares a DGX Spark for the other demos: checks the machine, then builds and starts
a Docker container for each demo listed in `demos.toml`.

Start the dashboard, which is where you run the checks and launch the demos:

```bash
python3 demos/demo1/spark_setup.py dashboard
```

It starts the dashboard in the background and prints its link (http://localhost:8500). It
doesn't start any demo. The dashboard's **Run checks** and **Launch** buttons run the steps below
and show their progress, flag errors, and suggest fixes.

The same steps from the terminal, printed as they run:

```bash
python3 spark_setup.py                    # host checks + every demo, and leaves them all running
python3 spark_setup.py --only demo3       # host checks + one demo
python3 spark_setup.py --build-only       # host checks + builds, start nothing (Launch starts a demo)
python3 spark_setup.py --rebuild          # rebuild images even if they're up to date
python3 spark_setup.py --host-only        # host checks only
python3 spark_setup.py status             # last run, with suggested fixes
python3 spark_setup.py dashboard [--stop] # start or stop just the dashboard
python3 country_data.py [--plan]          # download this Spark's country data (or just show its size)
bash shutdown.sh                          # stop any setup run, download, the dashboard and all containers
```

**AI Almanac:** the dashboard's AI Almanac tab launches [AI Almanac](https://github.com/AI-Almanac/ai-almanac)
from its own checkout, `~/ai-almanac` (the fleet clones and sets it up on the demo Sparks), with
`scripts/spark.sh up`, and shows it with its access token. It counts as a demo: one at a time.

**AI Weather Lab:** on the fleet's demo Sparks, the desktop has an **AI Weather Lab** shortcut
(icon: `ai-weather-lab.svg`) that runs `spark_setup.py dashboard --open`: it starts the dashboard
and opens it in the browser.

**`spark-setup`:** every run also installs this command (in `~/.local/bin`), the same script from
any folder: `spark-setup dashboard`, `spark-setup --only demo3`, `spark-setup status`, ...

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

Demos 3 and 4 read these files there first, and the bucket for anything else: their start scripts
mount this folder read-only at `/country_data` (`python3 country_data.py --path` says where it is).
They're all on the forecasts' 0.25° grid, which is the one both demos compare on.

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
