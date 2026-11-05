import os
import math
from numbers import Integral, Real
from datetime import datetime
from types import SimpleNamespace


# dry_extent > dry_spell

# model name no space

#window should include day_bins

# probabilistic -- must be only one window in window_list, window end == max_forecast_day

#Search window (days) after onset to check for the dry spell. Must be >= dry_spell.
# check trailing comma in _list


# plot_bar_bss_rpss_auc = True, <- save_csv_score = True

# plot_panel_heatmap_error = True <--- veri window list must have >1 bins

# 'plot_panel_heatmap_skill' <---  veri window list must be single item
# 'plot_bar_bss_rpss_auc' <---  veri window list must be single item


class ROMPConfigError(Exception):
    """Custom exception for configuration failures."""
    pass

class ROMPValidator:
    def __init__(self, config_dict):
        # Convert the dictionary to an object so we can use self.cfg.param
        self.cfg = SimpleNamespace(**config_dict)
        self.errors = []

    def _add_error(self, msg):
        self.errors.append(msg)

    def validate(self):
        """Runs all consistency and logic checks."""
        self.errors = []
        self._check_structure()
        if self.errors:
            self._raise_errors()
        self._check_temporal()
        self._check_list_lengths()
        self._check_model_names()
        self._check_tuple_ranges()
        self._check_forecast_limits()
        self._check_probabilistic_constraints()
        self._check_monsoon_logic()
#        self._check_graphics_dependencies()
        
        if self.errors:
            self._raise_errors()
        return True

    def _raise_errors(self):
        report = "\n".join(f"  • {err}" for err in self.errors)
        raise ROMPConfigError(f"ROMP Configuration is invalid:\n{report}")

    def _check_structure(self):
        # Reject malformed collections before indexing pairs or taking max().
        for key in ("model_list", "verification_window_list", "tolerance_days_list", "day_bins"):
            value = getattr(self.cfg, key, None)
            if not isinstance(value, (tuple, list)) or not value:
                self._add_error(f"{key} must be a non-empty list or tuple.")
        models = getattr(self.cfg, "model_list", ())
        if isinstance(models, (tuple, list)):
            if any(not isinstance(model, str) or not model.strip() for model in models):
                self._add_error("model_list must contain non-empty model names.")
            elif len(set(models)) != len(models):
                self._add_error("model_list contains duplicate model names.")
        for key in ("verification_window_list", "day_bins"):
            windows = getattr(self.cfg, key, ())
            if isinstance(windows, (tuple, list)):
                for window in windows:
                    if (not isinstance(window, (tuple, list)) or len(window) != 2
                            or any(not isinstance(day, Integral) or isinstance(day, bool) for day in window)):
                        self._add_error(f"{key} entries must be (start, end) integer pairs: {window!r}")
                    elif window[0] < 1:
                        self._add_error(f"{key} starts before forecast day 1: {window!r}")
        tolerances = getattr(self.cfg, "tolerance_days_list", ())
        if isinstance(tolerances, (tuple, list)) and any(
                not isinstance(day, Integral) or isinstance(day, bool) or day < 0 for day in tolerances):
            self._add_error("tolerance_days_list must contain non-negative integer days.")
        for key, minimum in (("max_forecast_day", 1), ("wet_spell", 1), ("dry_spell", 0), ("dry_extent", 0)):
            value = getattr(self.cfg, key, None)
            if value is None and key == "wet_spell":
                continue
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                self._add_error(f"{key} must be an integer >= {minimum}.")
        for key in ("wet_init", "wet_threshold", "dry_threshold", "onset_percentage_threshold"):
            value = getattr(self.cfg, key, None)
            if value is not None and (not isinstance(value, Real) or not math.isfinite(value)
                                      or value < 0 or (key == "onset_percentage_threshold" and value > 1)):
                self._add_error(f"{key} has an invalid value: {value!r}.")

    def _check_temporal(self):
        try:
            sd = datetime(*self.cfg.start_date)
            ed = datetime(*self.cfg.end_date)
            if sd >= ed:
                self._add_error(f"start_date ({sd.date()}) is after end_date ({ed.date()})")

            sy = self.cfg.start_year_clim
            ey = self.cfg.end_year_clim
            if sy > ey:
                self._add_error(f"start_year_clim ({sy}) is after end_year_clim ({ey})")

        except Exception as e:
            self._add_error(f"Date formatting error: {e}")

    def _check_list_lengths(self):
        # We use the length of model_list as the 'Master' length
        master_len = len(self.cfg.model_list)
        
        # Access the underlying dictionary to find all keys ending in '_list'
        for key, value in vars(self.cfg).items():
            if key.endswith('_list') and isinstance(value, (list, tuple)):
                if key not in ('verification_window_list', 'tolerance_days_list'):
                    if len(value) != master_len:
                        self._add_error(f"Length Mismatch: '{key}' has {len(value)} items, \
                                        expected {master_len}")
                if key in ('tolerance_days_list',):
                    if len(value) != len(self.cfg.verification_window_list):
                        self._add_error(f"Length Mismatch: '{key}' has {len(value)} items, \
                                        expected {len(self.cfg.verification_window_list)}")

    def _check_model_names(self):
        # 1. Ensure no spaces in model names (critical for file paths/naming)
        for model in self.cfg.model_list:
            if " " in model:
                self._add_error(f"Invalid model name: '{model}' (Spaces are not allowed).")

    def _check_tuple_ranges(self):
        # 2. Ensure (start, end) pairs are logically ordered (start < end)
        for attr_name in ['verification_window_list', 'day_bins']:
            windows = getattr(self.cfg, attr_name)
            for window in windows:
                if window[0] >= window[1]:
                    self._add_error(f"Invalid range in {attr_name}: {window}. \
                            First number must be smaller than the second.")

        bins = sorted(self.cfg.day_bins)
        if any(left[1] >= right[0] for left, right in zip(bins, bins[1:])):
            self._add_error("day_bins must not overlap (bin endpoints are inclusive).")

    def _check_forecast_limits(self):
        # 3. Largest window value cannot exceed max_forecast_day
        max_v_day = max([pair[1] for pair in self.cfg.verification_window_list])
        if max_v_day > self.cfg.max_forecast_day:
            self._add_error(f"Verification window limit ({max_v_day}) \
                    exceeds max_forecast_day ({self.cfg.max_forecast_day}).")

        if max(pair[1] for pair in self.cfg.day_bins) > self.cfg.max_forecast_day:
            self._add_error("day_bins exceed max_forecast_day.")

    def _check_probabilistic_constraints(self):
        # Using .get() in case the key is missing to avoid AttributeError
        is_prob = getattr(self.cfg, 'probabilistic', False)

        if is_prob:
            ## 4. Probabilistic runs must have exactly one verification window
            #if len(self.cfg.verification_window_list) != 1:
            #    self._add_error("Probabilistic Mode Error: \
            #            verification_window_list must contain exactly one tuple.")

            # 5. Window max cannot be smaller than day_bins max
            max_v_day = max([pair[1] for pair in self.cfg.verification_window_list])
            max_bin_day = max([pair[1] for pair in self.cfg.day_bins])
            if max_v_day < max_bin_day:
                self._add_error(f"Window mismatch: Largest verification day \
                        ({max_v_day}) is smaller than largest day_bin ({max_bin_day}).")

    def _check_monsoon_logic(self):
        if self.cfg.dry_extent > 0 and self.cfg.dry_extent < self.cfg.dry_spell:
            self._add_error(f"Logic Conflict: \
                    dry_extent ({self.cfg.dry_extent}) < dry_spell ({self.cfg.dry_spell})")

        if getattr(self.cfg, "thresh_file", None) and not os.path.exists(self.cfg.thresh_file):
            self._add_error(f"File missing: thresh_file not found at '{self.cfg.thresh_file}'")

    def _check_graphics_dependencies(self):
        """Validates that plot settings match data availability and output requirements."""
        
        # 1. Bar plot requires CSV data for input
        if getattr(self.cfg, 'plot_bar_bss_rpss_auc', False):
            if not getattr(self.cfg, 'save_csv_score', False):
                self._add_error("Graphics Conflict: 'plot_bar_bss_rpss_auc' \
                        requires 'save_csv_score' to be True.")

        # 2. Panel Error Heatmap requires multiple items to compare (Matrix layout)
        if getattr(self.cfg, 'plot_panel_heatmap_error', False):
            if len(self.cfg.verification_window_list) <= 1 or len(self.cfg.model_list) <= 1:
                self._add_error("Graphics Conflict: 'plot_panel_heatmap_error' \
                        requires >1 model AND >1 verification window.")

        # 3. Panel Skill Heatmap logic
        if getattr(self.cfg, 'plot_panel_heatmap_skill', False):
            if len(self.cfg.verification_window_list) != 1:
                self._add_error("Graphics Conflict: 'plot_panel_heatmap_skill' \
                        only supports exactly ONE verification window.")

        # 4. Bar plots for Skill Scores (BSS/RPSS/AUC)
        if getattr(self.cfg, 'plot_bar_bss_rpss_auc', False):
            if len(self.cfg.verification_window_list) != 1:
                self._add_error("Graphics Conflict: 'plot_bar_bss_rpss_auc' \
                        only supports exactly ONE verification window.")


if __name__ == "__main__":
    from momp.lib.loader import get_cfg
    cfg = get_cfg()
    try:
        validator = ROMPValidator(vars(cfg))
        validator.validate()
        print("Configuration validated!")
    except ROMPConfigError as e:
        print(e)
