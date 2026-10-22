# Demo 4 

Because the AI models and datasets are large, the benchmarking environment is containerized using Docker and hosted on a remote server (the "Spark"). You will launch the backend via Docker and access the interactive Streamlit UI from your local machine using an SSH tunnel. Alternatively, a local development setup using `uv` is provided.

---

## Environment Setup

Choose the setup that best fits your workflow. Option A is the recommended approach for running the full benchmark. Option B is for local code development.

### Option A: Remote Server Setup with Docker (Recommended)
You will need two terminal windows: one connected to the remote server to run the Docker container, and one on your local machine to create the SSH tunnel.

1. **SSH into the Spark server** using your preferred terminal:
   ```bash
   ssh <you>@country-spark
   ```
2. **Navigate to the project directory** (if not already there).
3. **Start the container** by running the `run.sh` script. This builds the image, mounts the necessary cache volumes (for the model weights), and starts the Streamlit server on port 8501:
   ```bash
   bash docker/run.sh rebuild
   ```

or 

  ```bash
   docker restart ai_weather
   ```



   *(Tip: If you are actively editing the code and want changes to reflect live without rebuilding the image, use `DEV=1 bash docker/run.sh` instead).*
4. **Create the SSH Tunnel** (Local Machine): Open a new terminal window on your local laptop. Set up port forwarding so your local browser can access the remote Streamlit server:
   ```bash
   ssh -L 8501:localhost:8501 <you>@country-spark
   ```
   *(Leave this terminal open while you work).*
5. **Access the Dashboard**: Open your local web browser and navigate to `http://localhost:8501`.



---

## Command-Line Execution (`run_momp.py`)

The `run_momp.py` script is the backend execution engine. It processes one model per run, categorizing them into Deterministic (`det`) or Probabilistic (`prob`) modes. If you are using the Docker setup, you must execute these commands inside the running container.

1. **Enter the Docker Container** (Remote Server Terminal):
   ```bash
   docker exec -it ai_weather bash
   ```
2. **Run a Deterministic Model** (e.g., GraphCast, FuXi, AIFS):
   ```bash
   python run_momp.py --mode det --model GraphCast
   ```
3. **Run a Probabilistic (Ensemble) Model** (e.g., AIFS_ENS, GenCast):
   ```bash
   python run_momp.py --mode prob --model AIFS_ENS
   ```
4. **Run with a Benchmark Specification File**:
   If you have a predefined `benchmark_spec.json` containing evaluation windows and shared settings:
   ```bash
   python run_momp.py --mode det --model FuXi --spec path/to/benchmark_spec.json
   ```
5. **Exit the container** when finished:
   ```bash
   exit
   ```

---

## Part 3: UI-Based Execution (Streamlit Dashboard)

The interactive dashboard provides a visual interface to configure, execute, and analyze the benchmark without needing to type CLI commands.

1. **Navigate to the Benchmarking Page**: In your local browser at `http://localhost:8501`, look at the sidebar on the left. Click on the page named "Short Medium Benchmarking" (or similar, depending on the main `ui/app.py` configuration).
2. **Configure Benchmark Inputs**:
   - **Ground Truth**: Select the observational dataset (e.g., CHIRPS or ENACTS).
   - **Benchmark Coverage**: Select the region (e.g., Ethiopia). Note: The run button will be disabled if the region's data or masks are not configured in `config.py`.
   - **Forecast Window**: Choose the lead time window (1-15 days or 16-30 days).
3. **Select Models**:
   - The UI auto-detects available models from your data directories.
   - Use the "Filter by category" radio buttons to view Deterministic or Probabilistic models.
   - Click the rows in the table to select the models you want to benchmark.
4. **Adjust Model Run Windows & Settings**:
   - Expand the model details to set the Evaluation Start/End dates and Baseline Years.
   - Select Initialization Days (e.g., day 0, day 3).
   - (Optional) Expand "Advanced Temporal & Verification Settings" to tweak tolerances and verification windows.
   - Scroll down to "Shared Settings" to adjust event detection thresholds (e.g., Wet-day threshold, Dry spell limit) and area masks.
5. **Execute the Benchmark**:
   - Click the "Run Benchmark" button.
   - A live log stream will appear at the bottom of the page showing the progress of the `run_momp.py` subprocess running inside the Docker container.
   - Wait for the completion message. This can take several minutes per model depending on the server's hardware.

---

## Part 4: Exploring Results & Exporting

After the benchmark finishes, the UI will automatically refresh and populate the Results section with several tabs:

1. **Metric Map**: Visualize spatial metrics (False Alarm Rate, Miss Rate, MAE). You can swipe between models, compare lead times, and clip to specific administrative regions if a shapefile is configured.
2. **All Metrics**: View a comprehensive, color-coded HTML table comparing all models across all windows, alongside detailed distribution statistics (Mean, Min, Median, P75, P90, Max).
3. **Benchmark Outputs**: Browse the generated PNG figures categorized by model type (Deterministic, Probabilistic, Climatology) and explore the raw NetCDF datasets.
4. **Onset Analysis**: Run a localized time-series onset analysis for specific coordinates or administrative regions.
5. **Skill vs Baseline**: View verification scores comparing the AI models against the Traditional Climatology baseline.
6. **Uncertainty**: Compute and view bootstrap confidence intervals and statistical significance.
7. **Benchmark Report & Export**: 
   - Click "Generate Complete PDF Report" to compile an executive summary, fair-comparison plan, verification scores, and all generated figures into a single scientific PDF.
   - Click "Download Complete PDF Report" to save it to your local machine.

