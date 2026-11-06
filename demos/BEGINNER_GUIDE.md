# Beginner's Guide to the Demos

This guide walks you through every demo in the training program, one step at a time, working
**directly on the Spark** (sitting at its own keyboard and screen). You do not need to know
Docker or Linux well. When a step says "type", copy the command into a terminal exactly as
shown and press **Enter**.

| Demo | What it is | Page address |
|---|---|---|
| [Demo 1](#demo-1--set-up-the-spark) | Sets up the Spark and starts the other demos | http://localhost:8500 |
| [Demo 3](#demo-3--run-your-first-ai-weather-forecast) | Run AI weather models and compare them with observations | http://localhost:8502 |
| [Demo 4](#demo-4--the-ai-weather-model-scorecard) | Benchmark AI models for rainy-season onset | http://localhost:8503 |
| Demo 5 ([lesson](../docs/_episodes/05-AI-Forest-test.md)) | Verify GraphCast and IFS over years of past forecasts (ForecastCheck) | http://localhost:8505 |

---

## Before you start

### Words used in this guide

- **Spark**: the NVIDIA DGX Spark computer the demos run on, for example `rwanda-spark`.
- **Terminal**: the window where you type commands.
- **localhost**: "this computer". The demo pages run on the Spark, so you open them in the
  Spark's own web browser at addresses like `http://localhost:8502`.
- **Container**: a self-contained box on the Spark holding one demo and everything it needs.

### What you need

1. A user account on the Spark (ask the organisers if you don't have one).
2. The Spark's keyboard, mouse and screen.

### Get ready (do this every time)

1. **Log in** to the Spark's desktop with your user name and password.
2. **Open a terminal**: press **Ctrl + Alt + T**, or open the applications menu and search
   for "Terminal". You'll see a prompt ending in `$`; that's where you type commands.
3. **Open a web browser** (Firefox or Chrome) from the applications menu or the dock.

Keep both windows open while you use the demos. Use the terminal for commands and the
browser for the demo pages.

---

## Demo 1 · Set up the Spark

Demo 1 checks that the Spark is healthy (GPU, Docker, disk space, access to the program's data
bucket) and then starts the other demos. It has a **dashboard**: a web page that shows what is
running and has a **Launch** button for each demo.

### Step 1: Start the dashboard

The code is already on the Spark. In the terminal, type:

```bash
cd ~/training-program-2026
python3 demos/demo1/spark_setup.py dashboard
```

This starts the dashboard only. No demo starts until you launch it.

Then open **http://localhost:8500** in the browser.

You will see:

- A **top bar** with one button per demo.
- A **card for each demo** with its status:
  - **Running**: it's started; click it to open the demo.
  - **Ready to launch**: built but not started.
  - **Stopped** or **Image not built**: click it, then **Launch ↗** on the page that opens.
- A **Run checks** button on the Demo 1 card.
- A **Country data** section at the bottom of the page.

### Step 2: Check the Spark

Click **Run checks** on the Demo 1 card. It checks that the GPU is visible, Docker is running,
containers can use the GPU, there is enough disk space, the program's Google bucket can be read,
and a short GPU test passes. A failed check is shown in red with a suggested fix.

### Step 3: Download your country's data

Under **Country data**, click the button to download your country's data from the program's
bucket. The dashboard works out the country from the Spark's name. The download keeps going if
you close the page, and clicking again skips files you already have.

### Step 4: Launch a demo

1. Click a demo in the top bar (for example **Demo 3**).
2. Read the short summary of what you'll learn.
3. Click **Launch ↗**. A new browser tab opens, shows the start-up progress, then shows the demo.
   The first launch of each demo also builds it, which can take up to an hour. For Demo 2, click
   **Launch CDT ↗**.

**Only one demo runs at a time** on a Spark, because they share the GPU and memory. If another
demo is running, the dashboard asks before stopping it. Anyone else using that demo loses their
place, so check with your group first.

To stop a demo, click **Stop** on its page in the dashboard.

### Useful Demo 1 commands

First type `cd ~/training-program-2026/demos/demo1`, then any of these:

| Command | What it does |
|---|---|
| `python3 spark_setup.py status` | Shows the result of the last setup, with fixes for anything that failed |
| `python3 spark_setup.py --host-only` | Runs only the Spark health checks |
| `python3 spark_setup.py --only demo3` | Health checks, then sets up just one demo (`demo3` or `demo4`) |
| `python3 spark_setup.py --build-only` | Health checks, then builds the demos whose code changed, without starting any |
| `python3 spark_setup.py --rebuild` | Rebuilds every demo, even if its code didn't change |
| `python3 verify_bucket_access.py` | Checks only that the Spark can read the program's data bucket |
| `bash shutdown.sh` | Stops everything: setup runs, the dashboard and all demos. Nothing is deleted. |

### If something goes wrong

| What you see | What to do |
|---|---|
| `unresolvable CDI devices` (often right after a reboot) | `systemctl --user restart docker`, then click **Run checks** again |
| `Cannot connect to the Docker daemon` | `systemctl --user start docker` |
| `no space left on device` | Ask an organiser; old images need clearing (`docker system df` shows what uses the space) |
| `Google bucket not readable` | The Spark's key at `/etc/fleet-gcs-key.json` is missing or wrong; ask an organiser |
| The browser says "Unable to connect" or "This site can't be reached" at localhost:8500 | The dashboard isn't running. Start it with the command in [Step 1](#step-1-start-the-dashboard) |
| `setup already running` | Another setup is still going. Wait, or check the dashboard |

---

## Demo 3 · Run your first AI weather forecast

Demo 3 is a web page where you pick AI weather models, run a forecast for your country, and
compare it with what was actually observed.

### Step 1: Open Demo 3

From the dashboard, click **Demo 3** in the top bar, then **Launch ↗**. Or, if Demo 3 is
already running, open **http://localhost:8502** directly.

### Step 2: Pick your models

- Choose **up to four models**.
- Hover over a model's card to see how long it takes to run, what it starts from, and what it
  can't do (for example, NeuralGCM has no 2 m temperature).
- **Tip for class:** some models are slow (Aurora 1.5 takes about 26 minutes, Atlas CRPS about
  19 minutes for 10 days). Pick a start date that already has a **saved run**, which loads
  instantly.

### Step 3: Set up the forecast

1. **Start date**: the date the forecast starts from. Hover over the ⓘ to see each model's allowed dates.
2. **Lead time**: how many days ahead to forecast.
3. **Members**: for ensemble models, how many versions of the forecast to run. More members
   take longer.
4. **Region**: your country, or draw a custom box.
5. **Use case**:
   - **Heat**: daily maximum temperature.
   - **Precipitation**: daily rainfall.
   - **Onset**: when the rainy season starts, using a definition you choose (ICPAC,
     Moron–Robertson, or Moron–Robertson with the MOK filter). You can change every threshold.

### Step 4: Run

Above the button, a small table lists each model with a **Status**:

- **saved run (loads instantly)**: already on this Spark.
- **saved run in the shared bucket**: downloaded from the program's bucket first.
- **will run on the Spark**: the model runs now, with the expected time in brackets.

The button says what will happen, for example **Load 2 saved runs and run 1 model**. Click it.
Models that need to run go one after another, and the table shows their progress.

The first time a model is used on a Spark, it downloads its weights first. This can take up
to 30 minutes, but only happens once.

### Step 5: Read the results

From top to bottom:

1. **How the forecast was made**: the steps (load, inputs, run, save) and how long each took.
2. **Event movie**: the forecast next to observations, day by day, plus the event's track
   on a map. For **Onset**, you see onset maps (forecast vs observed) and an onset movie instead.
3. **Time-series comparison**: every model you picked, on one chart, against observations.
4. **Forecast maps**: press **Play** to step through the days. You can also download the
   forecast as a NetCDF file to use in your own tools.

> **"Possible onset"** means a wet spell was found, but the forecast is too short to confirm
> that no dry spell follows. Treat it as "maybe".

### If something goes wrong

| What you see | What to do |
|---|---|
| The page doesn't load | Check that Demo 3 shows **Running** in the dashboard; if not, launch it again |
| A run fails while downloading starting conditions | The data server is busy. Wait a minute and try again |
| Onset maps say observations aren't available | Observed rainfall (CHIRPS) runs about a month behind. Pick an earlier start date |
| A model's first run is very slow | It is downloading its weights once. Later runs start in seconds |

More detail for organisers: [demo3/SETUP.md](demo3/SETUP.md) and [demo3/README.md](demo3/README.md).

---

## Demo 4 · The AI weather model scorecard

Demo 4 measures how well AI models forecast the **rainy-season onset**, compared with
observed rainfall and with a simple climatology forecast. Climatology means "onset happens on
its usual date". A model is only useful if it beats climatology.

### Step 1: Where the data is (nothing to do)

Demo 4 reads its forecasts and observations from the program's Google bucket,
`aim4scale_training_2026`, with the Spark's own key. That's the default, so there's nothing to
set up. To use another bucket, put `DEMO_BUCKET=<bucket-name>` in `~/training-program-2026/.env`,
then restart Demo 4 (`docker rm -f ai_weather`, then launch it again from the dashboard).

### Step 2: Open Demo 4

From the dashboard, click **Demo 4** in the top bar, then **Launch ↗**. It opens straight
on the benchmarking page. Or open **http://localhost:8503** and choose **Benchmarking**.

### Step 3: Choose what to benchmark

1. **Ground truth**: the observations to compare against, either `CHIRPS` (satellite) or `ENACTS`
   (gauges plus satellite).
2. **Benchmark coverage**: the region, for example `Ethiopia`.
3. **Forecast window**: `1-15 days` or `16-30 days` ahead.
4. **Models**: filter by **Deterministic** (one forecast per model) or **Probabilistic**
   (many ensemble members), then tick the models you want.
5. **Run settings**: evaluation start and end dates, baseline years, and which days of the
   week forecasts start on.
6. **Onset settings**: for example, a wet-day threshold of 20 mm and a dry-spell limit of 7 days.

Write down the settings you choose. Scores only mean something when you know the settings
behind them.

### Step 4: Run the benchmark

Click **Run Benchmark**. The log appears live on the page. When it finishes, the results
tabs fill in.

### Step 5: Explore the results

| Tab | What it shows |
|---|---|
| **Metric Map** | Maps of False Alarm Rate, Miss Rate and Mean Absolute Error. Use swipe or subtract to compare two models |
| **All Metrics** | A table of every model's scores next to climatology. The best value in each row is in **bold** |
| **Benchmark Outputs** | The saved figures, plus a NetCDF explorer to look inside the data files |
| **Onset Analysis** | Onset over time at a point. Type coordinates or pick a Woreda |
| **Skill vs Baseline & Uncertainty** | Whether a model is better than climatology, and whether that difference is real (confidence intervals) |
| **Benchmark Report & Export** | Download a PDF report with a summary, the models used and all figures |

**How to read the scores:**

- **Lower is better** for False Alarm Rate, Miss Rate, MAE, Brier Score and RPS.
- **Higher is better** for skill scores (BSS, RPSS) and AUC. A skill score above 0 means the
  model beats climatology.
- Read False Alarm Rate, Miss Rate and MAE **together**. A model can have few misses simply by
  raising many false alarms.

### Advanced: run from the command line

You can also run a benchmark without the web page. On the Spark, type:

```bash
docker exec -it ai_weather bash                          # step inside the Demo 4 container
python run_momp.py --mode det --model GraphCast     # one deterministic model
python run_momp.py --mode prob --model AIFS_ENS     # one ensemble (probabilistic) model
exit                                                # leave the container
```

Results are saved on the Spark in `~/data/ROMP_OUT/` and show up in the web page.

### If something goes wrong

| What you see | What to do |
|---|---|
| `DEMO_BUCKET is ... not a valid Google bucket name` | Fix or remove the `DEMO_BUCKET=` line in `~/training-program-2026/.env` ([Step 1](#step-1-where-the-data-is-nothing-to-do)), then restart Demo 4 |
| `Observation folder not found` or `Region mask not found` | Demo 4 can't reach the bucket. Run `python3 demos/demo1/verify_bucket_access.py` on the Spark and show the result to an organiser |
| The page doesn't load | Check that Demo 4 shows **Running** in the dashboard; if not, launch it again |

---

## When you're done for the day

- **Just leaving:** close your browser tabs and log out of the desktop. The demos keep
  running on the Spark for next time.
- **Shutting everything down** (organisers): in the terminal, type
  `bash ~/training-program-2026/demos/demo1/shutdown.sh`. Nothing is deleted. Start again with
  `python3 ~/training-program-2026/demos/demo1/spark_setup.py dashboard`.
