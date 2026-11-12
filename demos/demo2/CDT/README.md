# Climate Data Tools

Climate Data Tools (`CDT`) is a set of utility functions for meteorological data quality control,
homogenization, and merging station data with satellite and other proxies such as reanalysis. All
of its functions are available in its GUI.

On the Spark, CDT is Demo 2. It is already set up: it runs in its own Docker container, and you use
it in your web browser. You don't need to install R, CDT or anything else.

To install CDT on another computer instead (Windows, macOS or Linux), follow the instructions in
the CDT repository: <https://github.com/rijaf-iri/CDT>.

## Start CDT

1. Open a web browser on the Spark and go to the dashboard, **http://localhost:8500**.
2. Open **Demo 2 · Climate Data Tools** and click **Launch CDT**.

The first launch builds CDT's image, which takes a few minutes; a new tab shows the progress. When
it's ready, CDT opens in the browser at **http://localhost:8504** (its window on a virtual desktop,
shown by noVNC). Later launches open it straight away.

From your own computer, forward the port over SSH first, then open the same address there:

```bash
ssh -L 8504:localhost:8504 <your account>@<your Spark>
```

## Your files

CDT works in the folder `/data`. On the Spark that's the training program's
`demos/demo2/data` folder, in your home folder: `~/training-program-2026/demos/demo2/data`.

- In CDT, open and save files under `/data`.
- In the dashboard, Demo 2's **Data folder** section shows the same files. You can upload, download,
  view, edit, rename and delete them there, and make folders. Changes show up in CDT at once.
- You can also copy files into that folder from a terminal on the Spark.

CDT's own settings, including the details you enter the first time it starts, are kept in
`~/.cdt-docker`. They stay when CDT or the Spark restarts.

## If something goes wrong

- **You closed the CDT window.** CDT starts again by itself after a moment. Reload the browser tab if
  the window doesn't come back.
- **The page doesn't load.** Click **Launch CDT** again in the dashboard. It starts the container if
  it has stopped.
- **CDT is stuck.** Restart its container from a terminal on the Spark: `docker restart cdt`.

## For developers

- `Dockerfile`: CDT's image (based on `rocker/r2u:noble`; amd64 and arm64).
- `docker/entrypoint.sh`: starts CDT inside the container (`vnc` mode on the Spark).
- `../docker/run.sh`: how the dashboard starts the container (`spark_setup.py --only cdt` builds the
  image first).
- `cdt-docker.sh` and `docker/README.md`: running this container on another Linux server, over
  SSH with X11 forwarding or noVNC.
- The rest of this folder is CDT's source code, from <https://github.com/rijaf-iri/CDT>.
