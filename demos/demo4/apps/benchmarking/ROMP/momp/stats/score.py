import numpy as np
import pandas as pd
import warnings
from scipy import stats
from momp.stats.bins import extract_day_range


def match_samples(forecast_df, reference_df):
    """Return forecast/reference rows for the same initialization, grid cell, and event bin."""
    keys = ("init_time", "lat", "lon", "bin_label")
    for name, frame in (("forecast", forecast_df), ("reference", reference_df)):
        missing = set(keys).difference(frame.columns)
        if missing:
            raise KeyError(f"{name} pairs are missing sample-match columns: {sorted(missing)}")

    def normalized_keys(frame, name):
        key_frame = frame.loc[:, keys].copy()
        key_frame["init_time"] = pd.to_datetime(key_frame["init_time"], utc=True, errors="raise")
        for column in ("lat", "lon"):
            key_frame[column] = pd.to_numeric(key_frame[column], errors="raise").round(8)
        if key_frame.isna().any().any():
            raise ValueError(f"Missing {name} sample keys prevent reliable forecast matching")
        if not np.isfinite(key_frame[["lat", "lon"]].to_numpy()).all():
            raise ValueError(f"Nonfinite {name} spatial sample keys")
        key_frame["bin_label"] = key_frame["bin_label"].astype(str)
        duplicate = key_frame.duplicated(keep=False)
        if duplicate.any():
            examples = key_frame.loc[duplicate].head(3).to_dict("records")
            raise ValueError(f"Duplicate {name} forecast-observation sample keys: {examples}")
        return key_frame

    forecast_keys = normalized_keys(forecast_df, "forecast")
    reference_keys = normalized_keys(reference_df, "reference")
    forecast_index = pd.MultiIndex.from_frame(forecast_keys)
    reference_index = pd.MultiIndex.from_frame(reference_keys)
    common = forecast_index.intersection(reference_index, sort=False)
    if common.empty:
        raise ValueError(
            "Forecast and reference pairs have no matching samples on "
            "(init_time, lat, lon, bin_label); "
            f"forecast={len(forecast_df):,}, reference={len(reference_df):,} rows"
        )

    matched_forecast = forecast_df.iloc[forecast_index.get_indexer(common)].copy()
    matched_reference = reference_df.iloc[reference_index.get_indexer(common)].copy()
    if 'observed_onset' in matched_forecast and 'observed_onset' in matched_reference:
        observed_forecast = pd.to_numeric(matched_forecast['observed_onset'], errors='coerce').to_numpy()
        observed_reference = pd.to_numeric(matched_reference['observed_onset'], errors='coerce').to_numpy()
        if not np.array_equal(observed_forecast, observed_reference):
            raise ValueError("Matched forecast and reference samples have inconsistent observations")
    print(
        "Matched forecast/reference samples: "
        f"{len(common):,} keys ({len(matched_forecast):,} forecast rows, "
        f"{len(matched_reference):,} reference rows); "
        f"discarded {len(forecast_df) - len(matched_forecast):,}/"
        f"{len(reference_df) - len(matched_reference):,} unmatched rows"
    )
    return matched_forecast, matched_reference


def _validated_probability_pairs(frame):
    """Reject malformed score inputs instead of silently averaging partial rows."""
    result = frame.copy()
    for column, label in (("predicted_prob", "probability"), ("observed_onset", "observation")):
        try:
            values = pd.to_numeric(result[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {label} values in {column}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {label} values in {column}")
        valid = values.between(0, 1) if column == "predicted_prob" else values.isin((0, 1))
        if not valid.all():
            raise ValueError(f"Invalid {label}: expected {'[0, 1]' if column == 'predicted_prob' else 'binary 0/1'}")
        result[column] = values
    return result


def _effective_members(frame):
    column = "total_members_with_onset" if "total_members_with_onset" in frame else "total_members"
    members = pd.to_numeric(frame[column], errors="coerce")
    if not (np.isfinite(members) & (members >= 1) & (members == np.floor(members))).all():
        raise ValueError("Effective ensemble member counts must be finite positive integers")
    return members


def calculate_paired_brier(forecast_pairs, reference_pairs):
    """Compare Fair BS on common rows supported by both ensemble estimators."""
    counts = [len(forecast_pairs), len(reference_pairs)]
    forecast_pairs, reference_pairs = match_samples(forecast_pairs, reference_pairs)
    supported = ((_effective_members(forecast_pairs).to_numpy() >= 2) &
                 (_effective_members(reference_pairs).to_numpy() >= 2))
    if not supported.all():
        warnings.warn("Fair Brier comparison excludes paired samples with fewer than two usable members",
                      RuntimeWarning, stacklevel=2)
    scores = [calculate_brier_score(frame.iloc[np.flatnonzero(supported)])
              for frame in (forecast_pairs, reference_pairs)]
    for score, count in zip(scores, counts):
        score["n_excluded_samples"] = count - int(supported.sum())
    return tuple(scores)


# Function to calculate Brier Score and Fair Brier Score for the model forecasts (both overall and bin-wise)
def calculate_brier_score(forecast_obs_df):
    """
    Calculate Brier Score and Fair Brier Score for probabilistic forecasts.
    
    Brier Score = (1/n*m) * Σ(Y_ij - p_ij)²
    Fair Brier Score = (1/n*m) * Σ[(Y_ij - p_ij)² - p_ij(1-p_ij)/(ens-1)]
    
    where:
    - n = number of forecasts
    - m = number of bins per forecast  
    - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_ij = predicted probability for bin j in forecast i
    - ens = number of ensemble members
    
    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset', 'total_members'
    
    Returns:
    --------
    dict with Brier score metrics
    """
    
    forecast_obs_df = _validated_probability_pairs(forecast_obs_df)
    members = _effective_members(forecast_obs_df)
    if (members < 2).any():
        warnings.warn("Fair Brier score is undefined for fewer than two effective members",
                      RuntimeWarning, stacklevel=2)

    # Calculate squared differences
    squared_diffs = (forecast_obs_df['observed_onset'] - forecast_obs_df['predicted_prob'])**2
    
    # Calculate overall Brier Score
    brier_score = squared_diffs.mean()
    
    # Calculate Fair Brier Score correction term
    # ens-1 where ens is the number of ensemble members
    correction_term = (forecast_obs_df['predicted_prob'] * (1 - forecast_obs_df['predicted_prob'])) / (members.where(members >= 2) - 1)
    
    # Fair Brier Score
    fair_brier_components = squared_diffs - correction_term
    fair_brier_score = fair_brier_components.mean()
    # Calculate squared differences for bin-wise analysis
    forecast_obs_df['squared_diff'] = squared_diffs
    forecast_obs_df['fair_brier_component'] = fair_brier_components
    
    # Bin-wise Brier scores
    bin_brier_scores = forecast_obs_df.groupby('bin_label')['squared_diff'].mean()
    bin_fair_brier_scores = forecast_obs_df.groupby('bin_label')['fair_brier_component'].mean()
    
    brier_results = {
        'brier_score': brier_score,
        'fair_brier_score': fair_brier_score,
        'bin_brier_scores': bin_brier_scores.to_dict(),
        'bin_fair_brier_scores': bin_fair_brier_scores.to_dict(),
    }
    
    print(f"Brier Score: {brier_results['brier_score']:.4f}")
    print(f"Fair Brier Score: {brier_results['fair_brier_score']:.4f}")

    return brier_results

# Function to calculate Area Under the Curve (AUC) for the model forecasts (both overall and bin-wise)
def calculate_auc(forecast_obs_df):
    """
    Calculate Area Under the Curve (AUC) for probabilistic forecasts.
    
    AUC = Σ_{i,j,i',j'} Y_{ij}(1-Y_{i'j'}) · 1[p_{ij} > p_{i'j'}] / 
          [(Σ_{i,j} Y_{ij})(Σ_{i,j} (1-Y_{ij}))]
    
    where:
    - Y_{ij} = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_{ij} = predicted probability for bin j in forecast i
    - 1[p_{ij} > p_{i'j'}] = indicator function (1 if true, 0 if false)
    
    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset'
    
    Returns:
    --------
    dict with AUC metrics
    """
    
    forecast_obs_df = _validated_probability_pairs(forecast_obs_df)

    # Extract probabilities and observations
    p_ij = forecast_obs_df['predicted_prob'].values
    y_ij = forecast_obs_df['observed_onset'].values
    
    # Count total positive and negative cases
    n_positive = np.sum(y_ij)  # Σ Y_{ij}
    n_negative = np.sum(1 - y_ij)  # Σ (1-Y_{ij})
    
    if n_positive == 0 or n_negative == 0:
        print("Warning: Cannot calculate AUC - all cases are either positive or negative")
        return {
            'auc': np.nan,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'bin_auc_scores': {},
            'forecast_obs_df_with_ranks': forecast_obs_df
        }
    
    # Calculate AUC using the Mann-Whitney U statistic approach
    # This is equivalent to the formula but more computationally efficient
    
    # Separate positive and negative cases
    positive_probs = p_ij[y_ij == 1]
    negative_probs = p_ij[y_ij == 0]
    
    # Calculate Mann-Whitney U statistic
    u_statistic, _ = stats.mannwhitneyu(positive_probs, negative_probs, alternative='greater')
    
    # AUC is U statistic divided by (n_positive * n_negative)
    auc = u_statistic / (n_positive * n_negative)
    
    # Alternative direct calculation (less efficient for large datasets)
    # concordant_pairs = 0
    # for i in range(len(p_ij)):
    #     if y_ij[i] == 1:  # positive case
    #         for j in range(len(p_ij)):
    #             if y_ij[j] == 0:  # negative case
    #                 if p_ij[i] > p_ij[j]:
    #                     concordant_pairs += 1
    # auc_direct = concordant_pairs / (n_positive * n_negative)
    
    # Calculate AUC by bin
    bin_auc_scores = {}
    unique_bins = forecast_obs_df['bin_label'].unique()
    
    for bin_label in unique_bins:
        bin_data = forecast_obs_df[forecast_obs_df['bin_label'] == bin_label]
        
        if len(bin_data) > 0:
            bin_p = bin_data['predicted_prob'].values
            bin_y = bin_data['observed_onset'].values
            
            bin_n_positive = np.sum(bin_y)
            bin_n_negative = np.sum(1 - bin_y)
            
            if bin_n_positive > 0 and bin_n_negative > 0:
                bin_positive_probs = bin_p[bin_y == 1]
                bin_negative_probs = bin_p[bin_y == 0]
                
                bin_u_stat, _ = stats.mannwhitneyu(bin_positive_probs, bin_negative_probs, alternative='greater')
                bin_auc = bin_u_stat / (bin_n_positive * bin_n_negative)
            else:
                bin_auc = np.nan
            
            bin_auc_scores[bin_label] = bin_auc


    auc_results = {
        'auc': auc,
        'bin_auc_scores': bin_auc_scores,
    }
    
    return auc_results


def probe_prob(model, label, forecast_obs_df, *, day_bins=None):
    """Print compact event-wise probability and probability-mass diagnostics."""
    print(f"\nPROBE_PROB model={model!r} label={label!r} day_bins={day_bins!r}")
    if forecast_obs_df is None or forecast_obs_df.empty:
        print("PROBE_PROB: no forecast-observation rows")
        return

    required = {"bin_label", "predicted_prob", "observed_onset"}
    missing = required.difference(forecast_obs_df.columns)
    if missing:
        print(f"PROBE_PROB: missing columns {sorted(missing)}")
        return

    data = forecast_obs_df.copy()
    data["predicted_prob"] = pd.to_numeric(data["predicted_prob"], errors="coerce")
    data["observed_onset"] = pd.to_numeric(data["observed_onset"], errors="coerce")
    print(
        "rows={:,} invalid_probability={} probability_outside_0_1={} nonbinary_observation={}".format(
            len(data),
            int(data["predicted_prob"].isna().sum()),
            int((~data["predicted_prob"].between(0, 1)).sum()),
            int((~data["observed_onset"].isin([0, 1])).sum()),
        )
    )

    auc_by_event = calculate_auc(data).get("bin_auc_scores", {})
    summary = data.groupby("bin_label", sort=False, dropna=False).agg(
        n=("predicted_prob", "count"),
        probability_mean=("predicted_prob", "mean"),
        probability_min=("predicted_prob", "min"),
        probability_max=("predicted_prob", "max"),
        probability_unique=("predicted_prob", "nunique"),
        probability_zero_fraction=("predicted_prob", lambda values: values.eq(0).mean()),
        probability_one_fraction=("predicted_prob", lambda values: values.eq(1).mean()),
        observed_frequency=("observed_onset", "mean"),
        positive_count=("observed_onset", "sum"),
    )
    summary["auc"] = [auc_by_event.get(str(event), np.nan) for event in summary.index]
    print("PROBE_PROB per event:")
    print(summary.round(4).to_string())

    key_columns = [column for column in ("year", "init_time", "lat", "lon") if column in data]
    if key_columns:
        mass = data.groupby(key_columns, dropna=False).agg(
            probability_sum=("predicted_prob", "sum"),
            observation_sum=("observed_onset", "sum"),
        )
        print(
            "PROBE_PROB per-forecast probability_sum min/mean/max="
            f"{mass['probability_sum'].min():.4f}/"
            f"{mass['probability_sum'].mean():.4f}/"
            f"{mass['probability_sum'].max():.4f}; "
            "observation_sum min/mean/max="
            f"{mass['observation_sum'].min():.0f}/"
            f"{mass['observation_sum'].mean():.4f}/"
            f"{mass['observation_sum'].max():.0f}"
        )

    for column in ("members_with_onset", "total_members", "total_members_with_onset", "n_contributing_years"):
        if column in data:
            values = pd.to_numeric(data[column], errors="coerce").dropna()
            if not values.empty:
                print(
                    f"PROBE_PROB {column} min/median/max="
                    f"{values.min():g}/{values.median():g}/{values.max():g}"
                )


def probe_prob(model, label, forecast_obs_df, *, day_bins=None):
    """Print compact event-wise probability and probability-mass diagnostics."""
    print(f"\nPROBE_PROB model={model!r} label={label!r} day_bins={day_bins!r}")
    if forecast_obs_df is None or forecast_obs_df.empty:
        print("PROBE_PROB: no forecast-observation rows")
        return

    required = {"bin_label", "predicted_prob", "observed_onset"}
    missing = required.difference(forecast_obs_df.columns)
    if missing:
        print(f"PROBE_PROB: missing columns {sorted(missing)}")
        return

    data = forecast_obs_df.copy()
    data["predicted_prob"] = pd.to_numeric(data["predicted_prob"], errors="coerce")
    data["observed_onset"] = pd.to_numeric(data["observed_onset"], errors="coerce")
    print(
        "rows={:,} invalid_probability={} probability_outside_0_1={} nonbinary_observation={}".format(
            len(data),
            int(data["predicted_prob"].isna().sum()),
            int((~data["predicted_prob"].between(0, 1)).sum()),
            int((~data["observed_onset"].isin([0, 1])).sum()),
        )
    )

    auc_by_event = calculate_auc(data).get("bin_auc_scores", {})
    summary = data.groupby("bin_label", sort=False, dropna=False).agg(
        n=("predicted_prob", "count"),
        probability_mean=("predicted_prob", "mean"),
        probability_min=("predicted_prob", "min"),
        probability_max=("predicted_prob", "max"),
        probability_unique=("predicted_prob", "nunique"),
        probability_zero_fraction=("predicted_prob", lambda values: values.eq(0).mean()),
        probability_one_fraction=("predicted_prob", lambda values: values.eq(1).mean()),
        observed_frequency=("observed_onset", "mean"),
        positive_count=("observed_onset", "sum"),
    )
    summary["auc"] = [auc_by_event.get(str(event), np.nan) for event in summary.index]
    print("PROBE_PROB per event:")
    print(summary.round(4).to_string())

    key_columns = [column for column in ("year", "init_time", "lat", "lon") if column in data]
    if key_columns:
        mass = data.groupby(key_columns, dropna=False).agg(
            probability_sum=("predicted_prob", "sum"),
            observation_sum=("observed_onset", "sum"),
        )
        print(
            "PROBE_PROB per-forecast probability_sum min/mean/max="
            f"{mass['probability_sum'].min():.4f}/"
            f"{mass['probability_sum'].mean():.4f}/"
            f"{mass['probability_sum'].max():.4f}; "
            "observation_sum min/mean/max="
            f"{mass['observation_sum'].min():.0f}/"
            f"{mass['observation_sum'].mean():.4f}/"
            f"{mass['observation_sum'].max():.0f}"
        )

    for column in ("members_with_onset", "total_members", "total_members_with_onset", "n_contributing_years"):
        if column in data:
            values = pd.to_numeric(data[column], errors="coerce").dropna()
            if not values.empty:
                print(
                    f"PROBE_PROB {column} min/median/max="
                    f"{values.min():g}/{values.median():g}/{values.max():g}"
                )


# Function to calculate Ranked Probability Score (RPS) and Fair RPS for the model forecasts for (either 15 day or 30 day forecast)
def calculate_rps(forecast_obs_df):
    """
    Calculate Ranked Probability Score (RPS) and Fair RPS for probabilistic forecasts.
    
    RPS = (1/n*m) * Σ_i Σ_k (Σ_j≤k (Y_ij - p_ij))²
    Fair RPS = (1/n*m) * Σ_i Σ_k [(Σ_j≤k (Y_ij - p_ij))² - (Σ_j≤k p_ij)(1 - Σ_j≤k p_ij)/(ens-1)]
    
    where:
    - n = number of forecasts
    - m = number of bins per forecast  
    - Y_ij = 1 if onset occurred in bin j for forecast i, 0 otherwise
    - p_ij = predicted probability for bin j in forecast i
    - k = cumulative index (1 to m)
    - ens = number of ensemble members
    
    Parameters:
    -----------
    forecast_obs_df : DataFrame
        Output from create_forecast_observation_pairs_with_bins()
        Must contain columns: 'predicted_prob', 'observed_onset', 'total_members', 'bin_index'
    
    Returns:
    --------
    dict with RPS metrics
    """
    
    forecast_obs_df = _validated_probability_pairs(forecast_obs_df)
    member_column = 'total_members_with_onset' if 'total_members_with_onset' in forecast_obs_df else 'total_members'
    forecast_obs_df[member_column] = _effective_members(forecast_obs_df)
    indices = pd.to_numeric(forecast_obs_df['bin_index'], errors='coerce')
    if not (np.isfinite(indices) & (indices >= 0) & (indices == np.floor(indices))).all():
        raise ValueError("RPS category indices must be finite nonnegative integers")
    forecast_obs_df['bin_index'] = indices

    # Group by forecast (init_time, lat, lon) to get all bins for each forecast
    forecast_groups = forecast_obs_df.groupby(['init_time', 'lat', 'lon'])
    
    rps_values = []
    fair_rps_values = []
    
    for (init_time, lat, lon), group in forecast_groups:
        # Sort by bin_index to ensure proper ordering
        group_sorted = group.sort_values('bin_index')
        
        # Get predicted probabilities and observations for this forecast
        p_ij = group_sorted['predicted_prob'].values
        y_ij = group_sorted['observed_onset'].values
        member_column = 'total_members_with_onset' if 'total_members_with_onset' in group_sorted else 'total_members'
        total_members = group_sorted[member_column].iloc[0]
        
        if group_sorted['bin_index'].duplicated().any():
            raise ValueError("RPS categories must have unique bin indices")
        if {'bin_start', 'bin_end'}.issubset(group_sorted.columns):
            starts = pd.to_numeric(group_sorted['bin_start'], errors='coerce').to_numpy()
            ends = pd.to_numeric(group_sorted['bin_end'], errors='coerce').to_numpy()
            if (np.isnan(starts).any() or np.isnan(ends).any() or (starts > ends).any()):
                raise ValueError("RPS categories require valid inclusive day bounds")
            if np.any(starts[1:] <= ends[:-1]):
                raise ValueError("RPS category indices must be chronological and nonoverlapping")
            if not np.all(starts[1:] == ends[:-1] + 1):
                raise ValueError("RPS requires a contiguous categorical partition without missing day bins")
        if not np.isclose(p_ij.sum(), 1., atol=1e-8, rtol=0) or y_ij.sum() != 1:
            raise ValueError("RPS requires a complete categorical partition: probabilities and observations must sum to one")
        if group_sorted[member_column].nunique() != 1:
            raise ValueError("RPS requires one consistent effective member count per forecast")

        m = len(p_ij)  # Number of bins
        
        # Calculate RPS for this forecast
        rps_forecast = 0
        fair_rps_forecast = 0
        
        for k in range(1, m + 1):  # k from 1 to m
            # Cumulative sum up to bin k
            cum_p = np.sum(p_ij[:k])
            cum_y = np.sum(y_ij[:k])
            
            # RPS component
            diff_cum = cum_y - cum_p
            rps_component = diff_cum**2
            rps_forecast += rps_component
            
            # Fair RPS correction term
            fair_correction = (cum_p * (1 - cum_p)) / (total_members - 1) if total_members > 1 else np.nan
            fair_rps_component = rps_component - fair_correction
            fair_rps_forecast += fair_rps_component
        
        rps_values.append(rps_forecast)
        fair_rps_values.append(fair_rps_forecast)
    
    # Calculate overall RPS (average over all forecasts)
    rps = np.mean(rps_values) if rps_values else np.nan
    fair_rps = np.mean(fair_rps_values) if fair_rps_values else np.nan
    

    
    rps_results = {
        'rps': rps,
        'fair_rps': fair_rps,
        'n_forecasts': len(forecast_groups),

    }

    print(f"RPS: {rps_results['rps']:.4f}")
    print(f"Fair RPS: {rps_results['fair_rps']:.4f}")
    print(f"Number of forecasts: {rps_results['n_forecasts']}")
    
    return rps_results


def calculate_paired_rps(forecast_pairs, reference_pairs):
    """Score the same cases on both sides, requiring two effective members.

    A single unsupported climatology case must not poison the mean Fair RPS.
    Exclude the corresponding model case too, so RPSS compares like samples.
    """
    keys = ['init_time', 'lat', 'lon']
    supported = []
    input_counts = []
    for label, pairs in (("forecast", forecast_pairs), ("reference", reference_pairs)):
        # Category coverage and row matching must use the same canonical keys.
        pairs = pairs.copy()
        pairs['init_time'] = pd.to_datetime(pairs['init_time'], utc=True, errors='raise')
        for column in ('lat', 'lon'):
            pairs[column] = pd.to_numeric(pairs[column], errors='raise').round(8)
        if (pairs[keys].isna().any().any() or
                not np.isfinite(pairs[['lat', 'lon']].to_numpy()).all()):
            raise ValueError(f"Missing or nonfinite {label} sample keys")
        input_counts.append(pairs.groupby(keys).ngroups)
        member_column = ('total_members_with_onset' if 'total_members_with_onset' in pairs
                         else 'total_members')
        members = pd.to_numeric(pairs[member_column], errors='coerce')
        valid = pairs[keys].copy()
        valid['_supported'] = np.isfinite(members) & (members >= 2)
        mask = valid.groupby(keys)['_supported'].transform('all')
        filtered = pairs.loc[mask].copy()
        excluded = input_counts[-1] - filtered.groupby(keys).ngroups
        if excluded:
            warnings.warn(
                f"Fair RPS excluded {excluded} {label} cases with fewer than two "
                "usable members or a nonfinite member count; RPSS uses common supported cases.",
                RuntimeWarning, stacklevel=2,
            )
        supported.append(filtered)

    if all(not frame.empty for frame in supported):
        # Intersect whole forecasts, never truncate one categorical distribution
        # to the subset of labels present in the other forecast.
        categories = [frame.groupby(keys)['bin_label'].agg(frozenset) for frame in supported]
        common_cases = categories[0].index.intersection(categories[1].index)
        for case in common_cases:
            if categories[0].loc[case] != categories[1].loc[case]:
                raise ValueError("Paired RPS requires the same complete categories on both sides")
        supported = match_samples(*supported)
    else:
        supported = [frame.iloc[:0] for frame in supported]
    scores = [calculate_rps(frame) for frame in supported]
    for result, count in zip(scores, input_counts):
        result['n_excluded_forecasts'] = count - result['n_forecasts']
    return tuple(scores)



def calculate_brier_score_climatology(forecast_obs_df):
    """Score climatology using its effective ensemble, excluding before-window bins."""
    filtered = forecast_obs_df.loc[~forecast_obs_df['bin_label'].str.contains('Before', na=False)].copy()
    result = calculate_brier_score(filtered)
    result.update(n_samples=len(filtered), filtered_bins=sorted(filtered['bin_label'].unique()),
                  excluded_bins=['Before initialization'])
    return result


def calculate_auc_climatology(forecast_obs_df):
    """Use the same AUC estimator and input checks as forecast verification."""
    filtered = forecast_obs_df.loc[~forecast_obs_df['bin_label'].str.contains('Before', na=False)].copy()
    return calculate_auc(filtered)


def calculate_skill_scores(brier_forecast, rps_forecast, 
                          brier_climatology, rps_climatology):
    """
    Calculate skill scores for forecast model relative to climatology.
    
    Skill Score = 1 - (forecast_score / climatology_score)
    
    Parameters:
    -----------
    brier_forecast : dict
        Brier score results from forecast model
    rps_forecast : dict  
        RPS results from forecast model
    brier_climatology : dict
        Brier score results from climatology
    rps_climatology : dict
        RPS results from climatology
        
    Returns:
    --------
    dict with skill scores
    """
    
    brier_forecast = brier_forecast or {'fair_brier_score': np.nan, 'bin_fair_brier_scores': {}}
    brier_climatology = brier_climatology or {'fair_brier_score': np.nan, 'bin_fair_brier_scores': {}}
    rps_forecast = rps_forecast or {'fair_rps': np.nan}
    rps_climatology = rps_climatology or {'fair_rps': np.nan}

    skill_scores = {}
    
    print("="*60)
    print("SKILL SCORE CALCULATIONS")
    print("="*60)
    
    # Overall skill for the configured verification window.
    if not np.isfinite(brier_climatology['fair_brier_score']) or brier_climatology['fair_brier_score'] == 0:
        fair_bss_overall = float("nan")
        warnings.warn("\n brier_climatology is zero → fair BSS set to NaN \n")
    else:
        fair_bss_overall = 1 - (brier_forecast['fair_brier_score'] / brier_climatology['fair_brier_score'])

    skill_scores['fair_brier_skill_score'] = fair_bss_overall
    
    print(f"Fair Brier Skill Score (overall): {fair_bss_overall:.4f}")
    
    # Fair RPS Skill Score for the same window.
    reference_rps = rps_climatology['fair_rps']
    fair_rpss_overall = 1 - (rps_forecast['fair_rps'] / reference_rps) if np.isfinite(reference_rps) and reference_rps != 0 else np.nan
    skill_scores['fair_rps_skill_score'] = fair_rpss_overall
    
    print(f"Fair RPS Skill Score (overall): {fair_rpss_overall:.4f}")
    
    # Automatically extract target bins from the data, excluding unwanted bins
    all_forecast_bins = set(brier_forecast['bin_fair_brier_scores'].keys())
    all_clim_bins = set(brier_climatology['bin_fair_brier_scores'].keys())
    
    # Get intersection of bins present in both forecast and climatology
    common_bins = all_forecast_bins.intersection(all_clim_bins)
    
    # Filter out unwanted bins and keep only "Days X-Y" format bins
    target_bins = []
    excluded_bins = []
    
    for bin_label in common_bins:
        # Include only bins that start with "Days " and don't contain "After" or "Before"
        if (bin_label.startswith('Days ') and 
            not bin_label.startswith('After') and 
            not bin_label.startswith('Before')):
            target_bins.append(bin_label)
        else:
            excluded_bins.append(bin_label)
    
    # Sort bins by their day ranges
    # function extract_day_range already defined in stats.bins, import from there
    #def extract_day_range(bin_label):
    #    # Extract the start day from "Days X-Y" format
    #    if 'Days ' in bin_label:
    #        try:
    #            day_part = bin_label.replace('Days ', '').split('-')[0]
    #            return int(day_part)
    #        except:
    #            return 999  # Put unparseable bins at the end
    #    return 999
    
    target_bins = sorted(target_bins, key=extract_day_range)
    
    print(f"\nAutomatically detected target bins: {target_bins}")
    print(f"Excluded bins: {excluded_bins}")
    
    # Bin-wise Fair Brier Skill Scores
    bin_fair_bss = {}
    
    print(f"\nBin-wise Fair Brier Skill Scores:")
    for bin_label in target_bins:
        if bin_label in brier_forecast['bin_fair_brier_scores'] and bin_label in brier_climatology['bin_fair_brier_scores']:
            forecast_fair_brier_bin = brier_forecast['bin_fair_brier_scores'][bin_label]
            clim_fair_brier_bin = brier_climatology['bin_fair_brier_scores'][bin_label]
            
            fair_bss_bin = 1 - (forecast_fair_brier_bin / clim_fair_brier_bin) if np.isfinite(clim_fair_brier_bin) and clim_fair_brier_bin != 0 else np.nan
            bin_fair_bss[bin_label] = fair_bss_bin
            
            print(f"  {bin_label}: Fair BSS = {fair_bss_bin:.4f}")
        else:
            bin_fair_bss[bin_label] = np.nan
            print(f"  {bin_label}: Fair BSS = NaN (missing data)")
    
    skill_scores['bin_fair_brier_skill_scores'] = bin_fair_bss
    
    # Dynamic table header based on detected bins
    header = f"{'Metric':<30} {'Overall':<18}"
    for bin_name in target_bins:
        # Shorten bin names for table display
        short_name = bin_name.replace('Days ', '')
        header += f" {short_name:<12}"
    
    # Calculate table width
    table_width = 30 + 18 + 12 * len(target_bins)
    
    # Summary table
    print(f"\n" + "="*table_width)
    print("SKILL SCORE SUMMARY TABLE")
    print("="*table_width)
    print(header)
    print("-"*table_width)
    
    # Fair Brier Skill Score row
    fair_bss_row = f"{'Fair Brier Skill Score':<30} {fair_bss_overall:<18.4f}"
    for bin_name in target_bins:
        if bin_name in bin_fair_bss and not pd.isna(bin_fair_bss[bin_name]):
            fair_bss_row += f" {bin_fair_bss[bin_name]:<12.4f}"
        else:
            fair_bss_row += f" {'N/A':<12}"
    print(fair_bss_row)
    
    # Fair RPS Skill Score row
    fair_rpss_row = f"{'Fair RPS Skill Score':<30} {fair_rpss_overall:<18.4f}"
    for bin_name in target_bins:
        fair_rpss_row += f" {'N/A':<12}"  # RPS is overall only
    print(fair_rpss_row)
    
    print("-"*table_width)
    
    # Add interpretation guide
    print(f"\nInterpretation Guide:")
    print(f"• Positive skill scores indicate forecast is better than climatology")
    print(f"• Negative skill scores indicate forecast is worse than climatology") 
    print(f"• Skill score = 0 means forecast equals climatology")
    print(f"• Perfect score = 1.0")
    
    # Additional detailed results
    print(f"\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    print(f"\nForecast Fair Brier Score : {brier_forecast['fair_brier_score']:.4f}")
    print(f"Climatology Fair Brier Score : {brier_climatology['fair_brier_score']:.4f}")
    print(f"Fair Brier Skill Score: {fair_bss_overall:.4f}")
    
    print(f"\nForecast Fair RPS : {rps_forecast['fair_rps']:.4f}")
    print(f"Climatology Fair RPS : {rps_climatology['fair_rps']:.4f}")
    print(f"Fair RPS Skill Score: {fair_rpss_overall:.4f}")
    
    print(f"\nBin-wise Fair Brier Score Comparisons:")
    for bin_name in target_bins:
        if bin_name in brier_forecast['bin_fair_brier_scores'] and bin_name in brier_climatology['bin_fair_brier_scores']:
            forecast_val = brier_forecast['bin_fair_brier_scores'][bin_name]
            clim_val = brier_climatology['bin_fair_brier_scores'][bin_name]
            skill_val = bin_fair_bss[bin_name]
            print(f"  {bin_name}:")
            print(f"    Forecast: {forecast_val:.4f}")
            print(f"    Climatology: {clim_val:.4f}")
            print(f"    Skill Score: {skill_val:.4f}")
        else:
            print(f"  {bin_name}: Missing data")
    
    return skill_scores


