# Beginner's Guide to the Demos

This guide walks you through every demo in the training program, one step at a time, working
**directly on the Spark** (sitting at its own keyboard and screen). You do not need to know
Docker or Linux well. When a step says "type", copy the command into a terminal exactly as
shown and press **Enter**.

| Demo | What it is | Page address |
|---|---|---|
| [Demo 1](#demo-1--set-up-the-spark) | Sets up the Spark and starts the other demos | http://localhost:8500 |
| [Demo 3](#demo-3--run-ai-forecast) | Run AI weather models and compare them with observations | http://localhost:8502 |
| [Demo 4](#demo-4--africlimate-ai-forecastcheck) | Verify GraphCast and IFS over years of past forecasts (AfriClimate AI ForecastCheck) | http://localhost:8505 |
| [Demo 5](#demo-5--onset-benchmarking) | Benchmark AI models for rainy-season onset | http://localhost:8503 |
| [AI Almanac](#ai-almanac--assess-blend-and-forecast) | Benchmark AI models for rainy-season onset, blend them, and forecast this season | http://localhost:8765 (open it from the dashboard) |

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

After the first time, you can start the dashboard from any folder with:

```bash
spark-setup dashboard
```

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

1. Click a demo in the top bar (for example **3 · Run AI Forecast**).
2. Read the short summary of what you'll learn.
3. Click **Launch ↗**. A new browser tab opens, shows the start-up progress, then shows the demo.
   The first launch of each demo also builds it. For Demo 2, click
   **Launch CDT ↗**.

**Only one demo runs at a time** on a Spark, because they share the GPU and memory. If another
demo is running, the dashboard asks before stopping it. Anyone else using that demo loses their
place, so check with your group first.

To stop a demo, click **Stop** on its page in the dashboard.

### Useful Demo 1 commands

`spark-setup` is the setup script (`python3 ~/training-program-2026/demos/demo1/spark_setup.py`), from any folder:

| Command | What it does |
|---|---|
| `spark-setup status` | Shows the result of the last setup, with fixes for anything that failed |
| `spark-setup --host-only` | Runs only the Spark health checks |
| `spark-setup --only demo3` | Health checks, then sets up just one demo (`demo2` to `demo5`) |
| `spark-setup --build-only` | Health checks, then builds the demos whose code changed, without starting any |
| `spark-setup --rebuild` | Rebuilds every demo, even if its code didn't change |
| `python3 ~/training-program-2026/demos/demo1/verify_bucket_access.py` | Checks only that the Spark can read the program's data bucket |
| `bash ~/training-program-2026/demos/demo1/shutdown.sh` | Stops everything: setup runs, the dashboard and all demos. Nothing is deleted. |

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

## Demo 3 · Run AI Forecast

Demo 3 is a web page where you pick AI weather models, run a forecast for your country, and
compare it with what was actually observed.

### Step 1: Open Demo 3

From the dashboard, click **3 · Run AI Forecast** in the top bar, then **Launch ↗**. Or, if Demo 3 is
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

The Spark's setup downloads every model's weights in advance, so a model's first run doesn't
wait for them. If the setup hasn't finished downloading a model's weights yet, its first run
downloads them first, which can take up to 30 minutes. This only happens once.

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
| A model's first run is very slow | The setup hasn't finished downloading its weights, so the run is downloading them. This only happens once; later runs start in seconds |

More detail for organisers: [demo3/SETUP.md](demo3/SETUP.md) and [demo3/README.md](demo3/README.md).

---

## Demo 4 · AfriClimate AI ForecastCheck

Demo 4 uses **ForecastCheck** to answer the question a forecaster asks before trusting a
model: over years of past forecasts, how good is it here, in my regions and my seasons? It
scores GraphCast and IFS against observations, then **calibrates** them (a method called IDR)
and scores them again. There are two tracks:

- **Heat** (daily maximum temperature): Senegal and Sierra Leone.
- **Rainfall** (daily rainfall): Kenya and Rwanda.

### Step 1: Open the Demo 4 page

In the dashboard, click **4 · AfriClimate AI ForecastCheck** in the top bar.

If the buttons in Step 2 are greyed out with **Launch Demo 4 first**, click **Launch ↗** once.
The first launch builds Demo 4. Then come back to this page.

### Step 2: Score your country

Under **Country data for ForecastCheck**:

1. Under **Download data for**, click your country.
2. Click **Run pipeline**. It downloads your country's forecasts and observations, then scores
   and calibrates them. Two progress bars show how far it has got.

The pipeline keeps going if you close the page.

Your trainer may have run it already. A country marked **✓ scored** is ready.

### Step 3: Choose the country Demo 4 shows

Above **Launch ↗**, under **Show in Demo 4**, click your country. Each country is marked
**✓ scored**, **data, no scores yet** or **no data**.

### Step 4: Launch Demo 4

Click **Launch ↗**. A new browser tab opens Demo 4's own dashboard. Or, if Demo 4 is already
running, open **http://localhost:8505** directly.

### Step 5: Explore the results

1. **Pick an area** at the top of every page: the whole country, a region, or **More regions**
   for smaller areas.
2. **Set what to show** in the **Configuration** panel on the left of each chart:
   - **Forecast & Truth**: the variable (Tmax, Tmin or Rainfall), the models, and the
     observations to compare against (CHIRTS, ERA5 or CHIRPS).
   - **Score**: the metric.
   - **Conditioning**: the year, season, lead time and altitude band.
3. **Read the pages:**

| Page | What it shows |
|---|---|
| **National Overview → Raw Forecast Metrics** | How far off the forecasts are before calibration (MAE, bias, correlation), by lead time, by month, or as a map |
| **National Overview → Calibrated Forecast Metrics** | The error left after calibration (CRPS) |
| **Score Decomposition → Miscalibration** | How much error calibration removed. **Below zero means calibration helped** |
| **Threshold Diagnostics** | Hot-day or heavy-rain warnings: can I trust the probability (reliability), can the model tell event days apart (ROC), and is it worth acting on (cost-loss value) |

**How to read the scores:**

- **Always say which observations you compared against.** CHIRTS and ERA5 can disagree by more
  than the two models disagree with each other.
- **Rainfall scores are in mm per hour.** Multiply by 24 for mm per day.
- **Check the number of events** in the box under each Threshold Diagnostics chart. A curve built
  on a few events can look excellent by chance.

### Advanced: run from the command line

Everything above also works from the terminal on the Spark, with the `fcstcheck` command:

```bash
fcstcheck setup senegal    # choose the country, download its data, run the pipeline, start the dashboard
fcstcheck status           # what's running
fcstcheck logs pipeline    # follow the pipeline (Ctrl + C stops following, not the run)
fcstcheck stop             # stop Demo 4's dashboard (your results are kept)
```

### If something goes wrong

| What you see | What to do |
|---|---|
| The buttons say **Launch Demo 4 first** | Click **Launch ↗** once (Step 1), then come back |
| Demo 4 says there are no scores for your country | Run its pipeline ([Step 2](#step-2-score-your-country)), or wait for the one that's running to finish |
| New results don't appear in Demo 4 | Refresh the browser. If they still don't appear, type `fcstcheck restart` in the terminal and refresh again |
| Rainfall values look 24 times too small | They're in mm per hour. Multiply by 24 for mm per day |
| The page doesn't load | Check that Demo 4 shows **Running** in the dashboard; if not, launch it again |
| `fcstcheck: command not found` | Log out of the desktop and back in, then try again |

More detail: the [Demo 4 lesson](https://mbzuai-aim-for-scale.github.io/training-program-2026/05-ai-forest-test/).

---

## Demo 5 · Onset Benchmarking

Demo 5 measures how well AI models forecast the **rainy-season onset**, compared with
observed rainfall and with a simple climatology forecast. Climatology means "onset happens on
its usual date". A model is only useful if it beats climatology.

### Step 1: Where the data is (nothing to do)

Demo 5 reads its forecasts and observations from the program's Google bucket,
`aim4scale_training_2026`, with the Spark's own key. That's the default, so there's nothing to
set up. To use another bucket, put `DEMO_BUCKET=<bucket-name>` in `~/training-program-2026/.env`,
then restart Demo 5 (`docker rm -f ai_weather`, then launch it again from the dashboard).

### Step 2: Open Demo 5

From the dashboard, click **5 · Onset Benchmarking** in the top bar, then **Launch ↗**. It opens straight
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
docker exec -it ai_weather bash                          # step inside the Demo 5 container
python run_momp.py --mode det --model GraphCast     # one deterministic model
python run_momp.py --mode prob --model AIFS_ENS     # one ensemble (probabilistic) model
exit                                                # leave the container
```

Results are saved on the Spark in `~/training-program-2026/demos/demo4/ai_weather/data/ROMP_OUT/` and show up in the web page.

### If something goes wrong

| What you see | What to do |
|---|---|
| `DEMO_BUCKET is ... not a valid Google bucket name` | Fix or remove the `DEMO_BUCKET=` line in `~/training-program-2026/.env` ([Step 1](#step-1-where-the-data-is-nothing-to-do)), then restart Demo 5 |
| `Observation folder not found` or `Region mask not found` | Demo 5 can't reach the bucket. Run `python3 demos/demo1/verify_bucket_access.py` on the Spark and show the result to an organiser |
| The page doesn't load | Check that Demo 5 shows **Running** in the dashboard; if not, launch it again |

---

## AI Almanac · Assess, blend and forecast

AI Almanac is a web app for the **rainy-season onset**. Pick a region, see how well AI and
conventional weather models predicted the onset there, combine the best of them into a
**blend**, and run that blend for the current season. A built-in **assistant** can set up
each step from a plain-language request.

AI Almanac is a separate program from the demos. It lives in its own folder on the Spark,
**`~/ai-almanac`** (not in `~/training-program-2026`), and keeps its data and results in
`~/.local/share/ai-almanac-spark`. The Spark's setup installs it there for you.

### Step 1: Open AI Almanac

1. In the dashboard, click **AI Almanac** in the top bar.
2. Click **Launch ↗**. A new browser tab opens AI Almanac.

Always open it from the dashboard. The link the dashboard opens carries a private **access
token**; typing `http://localhost:8765` by hand gets you a page that asks for it.

AI Almanac counts as a demo: **only one runs at a time**, and launching it stops the
others (the dashboard asks first).

### Step 2: Find your way around

The bar at the top of AI Almanac has:

| Link | What it's for |
|---|---|
| **Home** | Where to start, and what each part does |
| **Almanac** | Reference pages on AI weather models, their designs and observation datasets, plus a glossary |
| **Benchmarks** | Score models against observations |
| **Blends** | Combine several models into one forecast |
| **Forecasts** | Run a blend for the current season |
| **Data** | The observations and model forecasts AI Almanac knows about |

Click **?** at the top right of any page for a short guided tour of that page.

### Step 3: Run a benchmark

1. Open **Benchmarks**. Example results are shown first: maps of each model's skill at every
   grid point.
2. Click **New benchmark**.
3. Either describe what you want to the assistant (for example, *"which models best predict
   Kiremt onset in Ethiopia?"*), or choose **Manual configuration** and fill in the region,
   season, models and observations yourself.
4. Click **Run benchmark**.

The results show, for every grid point, the **false alarm rate**, the **miss rate** and the
**onset error in days**, plus how skill changes with lead time. For ensemble models there are
also probability scores (Brier, RPS, AUC, reliability). You can limit the scores to a smaller
area by drawing a box or picking administrative areas.

### Step 4: Train a blend

1. Open **Blends** and click **New blend**.
2. Choose the models to combine (ones you benchmarked), or ask the assistant to set it up.
3. Click **Train blend**.

A blend learns how much weight to give each model from past seasons, then is checked on
years it didn't learn from (the **holdout years**). You can download its weights and scores.

### Step 5: Run a forecast

1. Open **Forecasts** and click **New forecast**.
2. Choose a trained blend.
3. Click **Run forecast**.

The map shows the **blended probability of onset** across the region. Click a grid cell to see
its probabilities in detail. The same numbers are in a table under the map, which you can
download.

### Using the assistant

Type a question or request into the assistant, in plain language. It can fill in a benchmark
or blend for you, explain results, and look into a run that failed. **Nothing runs until you
click Approve.** You can always do the same by hand instead.

Runs keep going in the background, even if you close the tab. Come back to the page to see
the results.

### If something goes wrong

| What you see | What to do |
|---|---|
| The page asks for an access token, or says you're not allowed in | Open AI Almanac from the dashboard (Step 1) |
| The dashboard says AI Almanac isn't on this Spark | Ask an organiser. The dashboard shows the command that installs it into `~/ai-almanac` |
| The assistant doesn't answer | Use **Manual configuration** instead; every step works without the assistant |
| The page doesn't load | Check that AI Almanac shows **Running** in the dashboard; if not, launch it again |

More detail: [AI Almanac on GitHub](https://github.com/AI-Almanac/ai-almanac) (the Spark runs its
`hcwf-demo-spark` branch). There is also a hosted version at
[ai-almanac.org](https://ai-almanac.org), where you can browse example results without
installing anything.

---

## When you're done for the day

- **Just leaving:** close your browser tabs and log out of the desktop. The demos keep
  running on the Spark for next time.
- **Shutting everything down** (organisers): in the terminal, type
  `bash ~/training-program-2026/demos/demo1/shutdown.sh`. Nothing is deleted. Start again with
  `spark-setup dashboard`.

---

## Using the dashboard from your own computer (over SSH)

Everything above assumes you are sitting at the Spark. If instead you connect to the Spark
from your own computer with **SSH**, the dashboard and the demos still run on the Spark, but
you open them in your own computer's browser. SSH carries each page across, one **port** per
page.

### Step 1: Connect with the ports forwarded

On **your own computer**, open a terminal (on Windows, PowerShell) and type, all on one line:

```bash
ssh -L 8500:localhost:8500 -L 8502:localhost:8502 -L 8503:localhost:8503 -L 8504:localhost:8504 -L 8505:localhost:8505 -L 8765:localhost:8765 <user>@<spark>
```

Replace `<user>` with your user name on the Spark and `<spark>` with the Spark's name, for
example `rwanda-spark`. Enter your password when asked.

Each `-L` brings one page to your computer:

| Port | Page |
|---|---|
| 8500 | The dashboard |
| 8502 | Demo 3 (Run AI Forecast) |
| 8503 | Demo 5 (Onset Benchmarking) |
| 8504 | Demo 2 (CDT) |
| 8505 | Demo 4 (AfriClimate AI ForecastCheck) |
| 8765 | AI Almanac |

### Step 2: Start the dashboard

In the **same** terminal, now connected to the Spark, type:

```bash
spark-setup dashboard
```

If this is the Spark's first setup, use `python3 ~/training-program-2026/demos/demo1/spark_setup.py dashboard`
instead.

### Step 3: Open it

Open **http://localhost:8500** in **your own computer's** browser. Everything works as in
the rest of this guide: when you click **Launch ↗**, the demo opens in a new tab on your
computer.

**Keep the SSH terminal open** while you use the demos. If you close it, or your computer
sleeps, the pages stop loading. Connect again with the command in Step 1 and refresh the
browser. The dashboard and demos keep running on the Spark meanwhile.

### If something goes wrong

| What you see | What to do |
|---|---|
| "Unable to connect" or "This site can't be reached" | Check that the SSH terminal is still connected; if not, run the Step 1 command again. If it is, start the dashboard (Step 2) |
| The dashboard opens, but a demo's tab doesn't load | That demo's port isn't forwarded. Reconnect with the full command in Step 1 |
| `bind: Address already in use` when connecting | Something on your computer already uses that port, often an earlier SSH session. Close other SSH terminals and connect again |
| `ssh: Could not resolve hostname` | The Spark's name is wrong, or your computer isn't on the Spark's network. Ask an organiser |
