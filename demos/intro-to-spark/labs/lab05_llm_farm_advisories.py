# %% [markdown]
# # Lab 05 — Farmer advisories from a local LLM
# **Goal:** turn the Lab 04 rainfall ensemble into short SMS advisories in English, written by an open LLM
# running on the same Spark. The forecast, the model and the text never leave the machine.
#
# One rule makes this safe enough to use: **code computes every number, the LLM only words them.** At the end
# we check each message against the facts we gave it.
#
# Model: **Qwen3-8B** (Apache 2.0, no Hugging Face login needed). First run downloads ~16 GB of weights; they
# need ~16 GB of memory in bf16. Requires `outputs/lab04_africa_ensemble.zarr`
# from Lab 04 and the `spark-intro` container (Lab 00), which includes `transformers`. Run this after Lab 04 has finished, not at
# the same time, on a shared Spark. Expected time: ~3 min with weights cached.

# %%
import json
import re
import numpy as np
import pandas as pd
import xarray as xr

ds = xr.open_zarr("outputs/lab04_africa_ensemble.zarr").isel(time=0)
ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")   # wrapped 342..52 -> -18..52
init = pd.Timestamp(ds["time"].values)
NENS = ds.sizes["ensemble"]

CITIES = {"Nairobi": (-1.29, 36.82), "Addis Ababa": (9.03, 38.74), "Kampala": (0.35, 32.58),
          "Dar es Salaam": (-6.79, 39.21), "Mogadishu": (2.05, 45.32)}
WET_MM = 25       # 5-day total that counts as a wet spell (same placeholder as Lab 04)
RAIN_DAY_MM = 5   # a day with at least this much rain counts as a "rain day" (placeholder)
SPRAY_MAX_WIND = 4  # m/s, safe spraying limit (placeholder)

# Lead 0 is the initial state. Each later tp06 value is the rain of the 6 h ending at that lead time, so
# 4 consecutive values make one UTC day (03:00–03:00 East Africa Time).
rain = (ds["tp06"].isel(lead_time=slice(1, None)).clip(min=0) * 1000).load()          # mm per 6 h
daily_rain = rain.coarsen(lead_time=4, boundary="trim").sum()                          # mm per day
t2m = (ds["t2m"].isel(lead_time=slice(1, None)) - 273.15).load()
wind = np.hypot(ds["u10m"], ds["v10m"]).isel(lead_time=slice(1, None)).load()
daily_tmax = t2m.coarsen(lead_time=4, boundary="trim").max()   # max of four 6-hourly samples: can miss the peak
daily_wind = wind.coarsen(lead_time=4, boundary="trim").max()
NDAYS = daily_rain.sizes["lead_time"]
# Full day names are clearer in an SMS than "Thu".
dates = [(init + pd.Timedelta(days=d)).strftime("%A %d %B") for d in range(NDAYS)]
print("init:", init, "| members:", NENS, "| days:", dates)

# %% [markdown]
# ## Step 1: facts, computed by code
# With only a few members a "probability" is really "how many forecast runs agree", so we say it that way and
# map it to a fixed word. The same goes for how much rain ("light", "heavy") and for what to do: code decides,
# from rules you can show an agronomist; the LLM only picks words. In a first version the LLM chose these
# itself and overstated: "heavy rain" for 14 mm, "showers possible" on "unlikely" days, "clear drains" in a
# dry week.

# %%
def likelihood(frac):
    return ("very unlikely" if frac < 0.2 else "unlikely" if frac < 0.4 else "possible" if frac < 0.6
            else "likely" if frac < 0.8 else "very likely")

def rain_amount(mm):  # mm/day; placeholder bands, set your own
    return "dry" if mm < 1 else "light" if mm < 10 else "moderate" if mm < 30 else "heavy"

WET = ("likely", "very likely")

DRY = ("unlikely", "very unlikely")

def day_summary(wet, typical):
    # One phrase per day, so the model never has to combine "light" with "very unlikely" itself.
    return f"rain {wet}" if wet in DRY else f"{rain_amount(typical)} rain {wet}, about {typical} mm"

def actions(wet_spell, spray_days):
    """Advisory policy (placeholders): what a farmer should do, given the forecast."""
    if wet_spell in WET:
        return ["clear drains before the rain", "delay spraying and fertilizer until after the rain"]
    if wet_spell == "possible":
        return ["prepare drains in case rain comes", "wait for the rain before planting"]
    acts = ["put harvested grain in the sun to dry", "irrigate where you can"]
    if spray_days:
        acts.insert(0, "spray on " + ", ".join(spray_days) + ", when it is calm and dry")
    return acts[:2]  # asked for "one or two", the 8B model still copied all three

def city_facts(name, la, lo):
    pick = lambda da: da.sel(lat=la, lon=lo, method="nearest")
    r, tx, w = pick(daily_rain), pick(daily_tmax), pick(daily_wind)
    total = r.sum("lead_time")
    days, spray_days = [], []
    for i, date in enumerate(dates):
        ri = r.isel(lead_time=i)
        wet = likelihood(int((ri >= RAIN_DAY_MM).sum()) / NENS)
        calm = bool(float(w.isel(lead_time=i).mean()) < SPRAY_MAX_WIND)
        if calm and wet in DRY:
            spray_days.append(date.split()[0])
        typical = int(round(float(ri.median())))
        days.append({"date": date,
                     "summary": day_summary(wet, typical),
                     "rain_day": wet,
                     "rain_mm_typical": typical,
                     "rain_mm_range": [int(round(float(ri.min()))), int(round(float(ri.max())))],
                     "max_temp_C": int(round(float(tx.isel(lead_time=i).mean())))})
    n_wet = int((total > WET_MM).sum())
    wet_spell = likelihood(n_wet / NENS)
    return {"place": name,
            "forecast_made": f"{init:%d %B %Y}",
            "forecast_days": NDAYS,
            "total_rain_mm_typical": int(round(float(total.median()))),
            "total_rain_mm_range": [int(round(float(total.min()))), int(round(float(total.max())))],
            "wet_spell_means_mm_over_5_days": WET_MM,
            "chance_of_wet_spell": wet_spell,
            "forecast_runs_agreeing_on_wet_spell": f"{n_wet} of {NENS}",
            "days": days,
            "suggested_actions": actions(wet_spell, spray_days)}

facts = {name: city_facts(name, *ll) for name, ll in CITIES.items()}
print(json.dumps(facts["Nairobi"], indent=1))

# %% [markdown]
# ## Step 2: load the LLM on the GPU
# On unified memory the weights sit in the same 128 GB as everything else; check `free -g` first on a shared box.

# %%
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
print(f"loaded in {time.perf_counter()-t0:.0f}s | weights: {torch.cuda.memory_allocated()/2**30:.1f} GiB")

# %% [markdown]
# ## Step 3: the prompt
# The system prompt is the advisory policy. Edit it with your agronomists, not just your engineers.

# %%
SYSTEM = """You write SMS weather advisories for smallholder farmers in East Africa.
Rules:
- Use only the facts in the JSON you are given. Do not add numbers, dates, places or weather that are not in it.
- At most 3 short sentences and under 320 characters. Plain words, no jargon, no emojis, no markdown.
- Write dates as full day names (e.g. "Thursday 29 October"), never abbreviations.
- Start with the overall picture, with the chance of a wet spell (copy its likelihood word). Mention at most the two most important days.
- When you mention a day, describe it with that day's summary, keeping its likelihood and amount words exactly.
  Never make a likelihood stronger or weaker, and use no other rain words (showers, storms, downpours, floods).
- In a sentence without a day name, the only likelihood word allowed is the one in chance_of_wet_spell.
- Never write field names from the JSON (words with underscores).
- Do not describe temperature changes or trends; give only the max_temp_C values if you mention temperature.
- End with the actions in suggested_actions, including any day names in them. No other advice."""

def build_prompt(f):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Facts:\n" + json.dumps(f)}]
    # Qwen3 "thinks" before answering by default; for short messages that only adds latency.
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

prompts = [build_prompt(facts[c]) for c in CITIES]
print(prompts[0][-700:])

# %% [markdown]
# ## Step 4: generate — one at a time, then all at once
# Generating text is memory-bandwidth bound: each new token reads all ~16 GB of weights, and Lab 01 measured
# ~232 GB/s. Measured on Spark: 5–7 tokens/s for one message. A batch reads the weights once per step for every
# message in it, so total throughput rises with batch size (3–4x for 5 messages).

# %%
def generate(batch, max_new_tokens=160):
    enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)  # greedy: reproducible
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    new = out[:, enc["input_ids"].shape[1]:]
    n_tokens = int((new != tok.pad_token_id).sum())
    texts = [re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
             for t in tok.batch_decode(new, skip_special_tokens=True)]
    return texts, dt, n_tokens

_, dt, n = generate(prompts[:1])
print(f"1 message:  {dt:5.1f} s, {n/dt:5.1f} tokens/s")
english, dt, n = generate(prompts)
print(f"{len(prompts)} messages: {dt:5.1f} s, {n/dt:5.1f} tokens/s total")
print(f"peak GPU alloc: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
for city, text in zip(CITIES, english):
    print(f"\n{city}: {text}")

# %% [markdown]
# ## Step 5: check before sending
# Every number must appear in that city's facts; each weekday named must match its date; every
#   likelihood word must match the days named in its sentence (or the wet spell, if no day is named); rain-amount
#   words must appear in the day summaries; weather words the facts never use (showers, storm, ...) are flagged;
#   any action keyword must belong to a suggested action; at most 2 actions; no field names from the JSON.
#
# This catches invented numbers, wrong days and overstated wording. It cannot judge whether the message reads
# well or could be misunderstood, so a person still signs off before anything is sent.

# %%
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
split = lambda s: [x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x]
numbers = lambda s: {int(x) for x in re.findall(r"\d+", s)}

def check_english(text, f):
    issues = []
    bad = sorted(numbers(text) - numbers(json.dumps(f)))
    if bad:
        issues.append(f"numbers not in facts: {bad}")
    for day in re.findall(r"(" + "|".join(DAYS) + r") (\d+)", text):
        if " ".join(day) not in " ".join(d["date"] for d in f["days"]):
            issues.append(f"wrong date: {' '.join(day)}")
    low = text.lower()
    by_day = {d["date"].split()[0]: d["rain_day"] for d in f["days"]}
    for sentence in split(text):
        named = [d for d in by_day if d in sentence]
        allowed = {by_day[d] for d in named} if named else {f["chance_of_wet_spell"]}
        # Longest first, so "very unlikely" is not also counted as "unlikely" or "likely".
        words = {w.replace("possibly", "possible")
                 for w in re.findall(r"very unlikely|very likely|unlikely|likely|possibl[ey]", sentence.lower())}
        if bad := sorted(words - allowed):
            issues.append(f"likelihood {bad} where facts say {sorted(allowed)}: '{sentence}'")
    used_amounts = set(re.findall(r"\b(light|moderate|heavy)\b", low))
    if bad := sorted(used_amounts - set(re.findall(r"light|moderate|heavy", json.dumps(f["days"])))):
        issues.append(f"rain amount not in facts: {bad}")
    if bad := sorted(set(re.findall(r"shower|storm|thunder|downpour|flood|drizzle|cool|drop|ris(?:e|ing)", low))):
        issues.append(f"weather words not in facts: {bad}")
    if leaked := re.findall(r"\b\w+_\w+\b", text):
        issues.append(f"field names leaked: {leaked}")
    verbs = {"spray", "put", "irrigate", "clear", "delay", "prepare", "wait"}
    if (n := len(verbs & set(re.findall(r"\w+", low)))) > 2:
        issues.append(f"{n} actions (max 2)")
    acts = " ".join(f["suggested_actions"]).lower()
    for kw in ["drain", "spray", "fertili", "plant", "harvest", "grain", "irrigat"]:
        if kw in low and kw not in acts:
            issues.append(f"action not suggested: {kw}")
    if len(text) > 320:
        issues.append(f"too long ({len(text)} chars)")
    return issues

rows = []
for city, text in zip(CITIES, english):
    issues = check_english(text, facts[city])
    rows.append({"place": city, "chars": len(text),
                 "status": "REVIEW" if issues else "ok", "issues": "; ".join(issues), "text": text})

pd.DataFrame(rows)[["place", "chars", "status", "issues"]]

# %%
with open("outputs/lab05_advisories.json", "w") as fh:
    json.dump({"model": MODEL_ID, "forecast_init": str(init), "facts": facts, "advisories": rows},
              fh, indent=1, ensure_ascii=False)
with open("outputs/lab05_advisories.md", "w") as fh:
    fh.write(f"# Advisories from AIFS-ENS init {init:%Y-%m-%d %H} UTC ({NENS} members)\n\n"
             f"Written by {MODEL_ID}. Not reviewed by a person.\n\n")
    for r in rows:
        fh.write(f"**{r['place']}** — {r['status']} {r['issues']}\n\n{r['text']}\n\n")
print("wrote outputs/lab05_advisories.json and .md")

# %% [markdown]
# ## Exercise
# 1. What does the LLM add over a fixed template ("{day}: {summary}. {action}."), now that code decides the
#    facts, the words and the action? Delete the likelihood/amount/action rules from `SYSTEM` and rerun: what does the
#    model overstate, and which of those does the checker catch?
# 2. Change one fact by hand (e.g. set Nairobi's total rain to 300 mm) and rerun Steps 3–5. Does the advice
#    change sensibly? Does the checker still pass? What does that tell you about what it does not check?
# 3. Try a larger LLM (e.g. `Qwen/Qwen3-32B`, ~64 GB in bf16). How do tokens/s and quality change, and can it
#    still share the Spark with an AIFS-ENS run (Lab 04)?
# 4. Local languages: we tried Swahili. Qwen3-8B could not write it (it called rain "michezo", games), and a
#    translation model (NLLB-200) wrote readable Swahili with errors no automatic check caught ("likely" became
#    "very likely"). What would you need before sending a message in a language nobody in the room reads?
#
# Caveats: the advisories inherit every limit of the forecast (Lab 04 caveats) and of a small ensemble.
# An LLM makes the message easier to read, not more accurate.
