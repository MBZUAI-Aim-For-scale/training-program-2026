"""Model choices and numeric lead ordering shared by All Metrics views."""

import math
import re
from pathlib import Path

import pandas as pd


def lead_window_sort_key(label):
    numbers = [int(value) for value in re.findall(r"\d+", str(label))]
    return (numbers[0], numbers[1] if len(numbers) > 1 else numbers[0], str(label)) if numbers else (math.inf, math.inf, str(label))


def metric_model_label(name):
    normalized = re.sub(r"[\s_-]+", "", str(name).casefold())
    aliases = {
        "aifs": "AIFS Single", "aifsv2": "AIFS Single", "aifssingle": "AIFS Single", "aifssinglev2": "AIFS Single",
        "aifsens": "AIFS Ensemble", "aifsensv2": "AIFS Ensemble", "aifsensemble": "AIFS Ensemble", "aifsensemblev2": "AIFS Ensemble",
        "gencast": "GenCast", "fuxi": "FuXi", "graphcast": "GraphCast",
        "climatology": "Traditional Climatology", "traditionalclimatology": "Traditional Climatology",
    }
    return aliases.get(normalized, str(name))


def metric_model_options(catalog, discovered_models):
    categories = {}
    for key, meta in catalog.items():
        label = metric_model_label(meta.get("model_name") or key)
        categories[label] = "Probabilistic" if meta.get("probabilistic", False) else "Deterministic"
    for model in discovered_models:
        label = metric_model_label(model)
        categories.setdefault(label, "Climatology" if "climatology" in label.casefold() else "Other")
    preferred = ["AIFS Single", "FuXi", "GraphCast", "AIFS Ensemble", "GenCast", "Traditional Climatology"]
    models = [name for name in preferred if name in categories]
    models += sorted(set(categories) - set(preferred), key=str.casefold)
    return models, categories


def metric_comparison_groups(models, categories):
    """Keep onset-error and probabilistic score columns in their own tables."""
    groups = [
        ("Onset error metrics", [model for model in models if categories[model] != "Probabilistic"]),
        ("Probabilistic skill metrics", [model for model in models if categories[model] == "Probabilistic"]),
    ]
    return [(label, group) for label, group in groups if group]


def load_probabilistic_metric_table(output_root):
    """Return saved scores, per-model comparison baselines, and read errors."""
    columns = {
        "Fair_Brier_Score": ("Fair Brier Score", 1),
        "Fair_Brier_Skill_Score": ("Fair Brier Skill Score (%)", 100),
        "Fair_RPS": ("Fair RPS", 1),
        "Fair_RPS_Skill_Score": ("Fair RPS Skill Score (%)", 100),
        "AUC": ("AUC", 1),
        "AUC_ref": ("Reference AUC", 1),
    }
    table, baselines, failures = {}, {}, []
    for source in sorted(Path(output_root).glob("*/overall_skill_scores_*.csv")):
        match = re.fullmatch(r"overall_skill_scores_(.+)_(\d+)-(\d+)\.csv", source.name)
        if not match:
            continue
        model, start, end = match.groups()
        window, label = f"Days {int(start)}–{int(end)}", metric_model_label(model)
        try:
            data = pd.read_csv(source, nrows=1)
            if data.empty:
                continue

            def number(column, summary=data):
                value = pd.to_numeric(summary[column], errors="coerce").iloc[0] if column in summary else None
                return float(value) if pd.notna(value) and math.isfinite(float(value)) else None

            for column, (metric, scale) in columns.items():
                if column not in data:
                    continue
                value = number(column)
                table[(metric, window, label)] = value * scale if value is not None else None
            baselines[("AUC", window, label)] = number("AUC_ref")
            baselines[("Fair Brier Skill Score (%)", window, label)] = 0.0
            baselines[("Fair RPS Skill Score (%)", window, label)] = 0.0
            for metric, score_column, skill_column, reference_column in (
                ("Fair Brier Score", "Fair_Brier_Score", "Fair_Brier_Skill_Score", "Reference_Fair_Brier_Score"),
                ("Fair RPS", "Fair_RPS", "Fair_RPS_Skill_Score", "Reference_Fair_RPS"),
            ):
                reference = number(reference_column)
                if reference is None:
                    score, skill = number(score_column), number(skill_column)
                    if score is not None and skill is not None and skill != 1:
                        reference = score / (1 - skill)
                baselines[(metric, window, label)] = (
                    reference if reference is not None and math.isfinite(reference) else None)
        except (OSError, ValueError) as exc:
            failures.append(f"{source.name}: {exc}")
    return table, baselines, failures
