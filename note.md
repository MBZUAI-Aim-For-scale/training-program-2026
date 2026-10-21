# Demo 4 Facilitator Notes: AI Weather Model Scorecard

For the person running the session. Not for participants.

## 1. Before the session (the day before)

| Check | How | Pass looks like |
| --- | --- | --- |
| Data is on the machine | `python ai_weather/config.py` | Every line `ok`, preferably `[local]` |
| Bucket login (only if a line says `[bucket]`) | `gcloud auth login`, then rerun the check | No `ERROR` lines |
| Dashboard starts | Open the Forecast Benchmarking page | Models table lists the models, no "config check failed" banner |
| Run Benchmark is enabled | Select Ethiopia | Button is not greyed out |
| One full deterministic run | Run AIFS once, note the time | Status tile shows Completed; Metric Map shows maps |
| Output folder is clean | Look in `data/ROMP_OUT/et/output` | Only files you expect (runs overwrite each other) |
| Probabilistic timing | Run AIFS ENS once | Know how long it takes; start it early in class |

Stage all data locally if you can. Bucket reads need a Google login on every machine and are slower.

## 2. Things that will go wrong, and the fastest fix

| What participants see | Cause | Fix |
| --- | --- | --- |
| "No data is configured for Ethiopia yet" | The page file was damaged by copy and paste (a space was added inside every string) | Replace the page with a clean copy. Never copy code through a chat or document viewer |
| Empty Models table | No `.nc` files found | `python ai_weather/config.py`, fix the `MISSING` line |
| 401 / "Anonymous caller" | `[bucket]` input and no Google login | `gcloud auth login`, or copy the data locally |
| Exit code -15 | They clicked during a run | Wait for the run, then rerun |
| Two people overwrite each other | One shared output folder | Run one at a time, or give each person their own output folder |
| Tabs missing (Onset, Skill, Uncertainty) | Optional modules not installed | Skip those parts, or install them beforehand |

## 3. Known gaps in this version (fix or tell participants)

1. **The dashboard does not write the run specification.** `run_momp.py` reads the evaluation window, baseline years, initialization days, forecast window, ground truth, coverage and the Shared settings from a spec file (`--spec` / `ROMP_SPEC_PATH`). The page sets environment variables that `run_momp.py` does not read, so those panels have no effect. Only the Advanced Temporal & Verification Settings are wired. The lesson says so in "Which Settings Reach the Run?". Remove that section when the page writes the spec.
2. **The Shared settings default would change the science.** The panel's dry-spell search extension defaults to 0, while `config.py` uses `dry_extent = 21`. Wiring the panel as it is would switch off the dry-spell veto. Change the panel default to 21 first.
3. **ENACTS is selectable but not supported.** `config.py` defines CHIRPS only (ENACTS is commented out), and the ground-truth catalog that `run_momp.py` looks up is not defined, so the choice is ignored. Use CHIRPS.
4. **`init_type = "daily"` behaves like `"weekly"`** in `momp/io/input.py`.
5. **Only Ethiopia is enabled.** Other countries need data, a country-specific region mask and an entry in `CONFIGURED_COVERAGE`. The local region mask is Ethiopia's regardless of `ROMP_REGION`.
6. **Figure 6 (portrait panel) is missing from the lesson**, and the deterministic run summary says 0.2° while the rest says 0.25°. Both are marked with `TODO` comments in the lesson source.
7. **Outputs are local.** Results and figures are written to `data/ROMP_OUT/<code>/` and are not uploaded to the bucket.

## 4. Questions that need a quick, clear answer

- *Why is the dry-spell extension 0 in the panel but 21 in the lesson?* See gap 2. The run uses the configuration.
- *Why did my change in Shared settings do nothing?* See gap 1. Use the configuration file (Exercise 7).
- *Why is BSS better late in the window when AUC is 0.5?* Base rate, not skill.
- *Is "Skill" in the Metric Map the skill score?* No, it is a threshold pass/fail map.

## 5. Pacing tips

- Start the probabilistic run before the first break, and read the deterministic results while it runs.
- Do the "Break it on purpose" exercise last; it works well only after participants have seen a normal run.
- If time is short, skip Exercise 3 and Exercise 7. Keep Exercises 4 and 6.
- After each participant's first run, ask them to read the log header: it shows the configuration that was really used.

## 6. After the session

- Collect the feedback from the AI Almanac activity by track (deterministic and probabilistic).
- Note which questions from the FAQ came up, and which came up that are not in it, and add them.