"""
custom_metrics.py -- Statistical tests for custom model monitoring.

Pure Python + scipy functions with no Snowflake dependencies,
so they work in stored procedures, ML Jobs, and notebooks.
"""

import numpy as np
from scipy import stats


def chi_squared_drift(baseline_counts: dict, current_counts: dict,
                      p_value_threshold: float = 0.05) -> dict:
    """Chi-squared test for categorical distribution drift.

    Args:
        baseline_counts: {category: count} from baseline data.
        current_counts: {category: count} from current data.
        p_value_threshold: p-value below which drift is flagged.

    Returns:
        Dict with metric_name, statistic, p_value, threshold, is_drift.
    """
    all_categories = sorted(set(baseline_counts) | set(current_counts))
    observed = np.array([current_counts.get(c, 0) for c in all_categories], dtype=float)
    expected_raw = np.array([baseline_counts.get(c, 0) for c in all_categories], dtype=float)

    # Scale expected to match current total
    total_observed = observed.sum()
    total_expected = expected_raw.sum()
    if total_expected == 0 or total_observed == 0:
        return {
            "metric_name": "chi_squared",
            "statistic": 0.0,
            "p_value": 1.0,
            "threshold": p_value_threshold,
            "is_drift": False,
            "details": {"error": "empty data"},
        }

    expected = expected_raw * (total_observed / total_expected)
    # Avoid zero expected values
    expected = np.where(expected == 0, 1e-10, expected)

    stat, p_value = stats.chisquare(observed, f_exp=expected)
    return {
        "metric_name": "chi_squared",
        "statistic": float(stat),
        "p_value": float(p_value),
        "threshold": p_value_threshold,
        "is_drift": bool(p_value < p_value_threshold),
        "details": {
            "categories": all_categories,
            "observed": observed.tolist(),
            "expected": expected.tolist(),
        },
    }


def ks_drift(baseline_values, current_values,
             p_value_threshold: float = 0.05) -> dict:
    """KS test for numerical distribution drift.

    Args:
        baseline_values: 1-D array-like of baseline values.
        current_values: 1-D array-like of current values.
        p_value_threshold: p-value below which drift is flagged.

    Returns:
        Dict with metric_name, statistic, p_value, threshold, is_drift.
    """
    baseline_values = np.asarray(baseline_values, dtype=float)
    current_values = np.asarray(current_values, dtype=float)

    if len(baseline_values) == 0 or len(current_values) == 0:
        return {
            "metric_name": "ks_test",
            "statistic": 0.0,
            "p_value": 1.0,
            "threshold": p_value_threshold,
            "is_drift": False,
            "details": {"error": "empty data"},
        }

    stat, p_value = stats.ks_2samp(baseline_values, current_values)
    return {
        "metric_name": "ks_test",
        "statistic": float(stat),
        "p_value": float(p_value),
        "threshold": p_value_threshold,
        "is_drift": bool(p_value < p_value_threshold),
        "details": {
            "baseline_mean": float(baseline_values.mean()),
            "current_mean": float(current_values.mean()),
            "baseline_std": float(baseline_values.std()),
            "current_std": float(current_values.std()),
        },
    }


def prediction_confidence_shift(baseline_probs, current_probs,
                                shift_threshold: float = 0.05) -> dict:
    """Detect shift in mean prediction confidence.

    Args:
        baseline_probs: 1-D array-like of predicted probabilities (baseline).
        current_probs: 1-D array-like of predicted probabilities (current).
        shift_threshold: absolute mean shift above which drift is flagged.

    Returns:
        Dict with metric_name, statistic (absolute shift), threshold, is_drift.
    """
    baseline_probs = np.asarray(baseline_probs, dtype=float)
    current_probs = np.asarray(current_probs, dtype=float)

    baseline_mean = float(baseline_probs.mean()) if len(baseline_probs) > 0 else 0.0
    current_mean = float(current_probs.mean()) if len(current_probs) > 0 else 0.0
    shift = abs(current_mean - baseline_mean)

    return {
        "metric_name": "prediction_confidence_shift",
        "statistic": shift,
        "p_value": None,
        "threshold": shift_threshold,
        "is_drift": bool(shift > shift_threshold),
        "details": {
            "baseline_mean": baseline_mean,
            "current_mean": current_mean,
            "baseline_std": float(baseline_probs.std()) if len(baseline_probs) > 0 else 0.0,
            "current_std": float(current_probs.std()) if len(current_probs) > 0 else 0.0,
        },
    }


def compute_all_metrics(baseline_df, current_df,
                        numerical_cols: list[str],
                        categorical_cols: list[str],
                        prediction_col: str | None = None,
                        baseline_probs=None,
                        current_probs=None,
                        thresholds: dict | None = None) -> list[dict]:
    """Run all custom metrics and return a list of result dicts.

    Args:
        baseline_df: pandas DataFrame of baseline data.
        current_df: pandas DataFrame of current data.
        numerical_cols: columns to run KS test on.
        categorical_cols: columns to run chi-squared test on.
        prediction_col: unused (kept for API symmetry); pass probs directly.
        baseline_probs: 1-D array of baseline predicted probabilities.
        current_probs: 1-D array of current predicted probabilities.
        thresholds: optional dict of {metric_name: threshold_value}.

    Returns:
        List of metric result dicts.
    """
    thresholds = thresholds or {}
    results = []

    for col in categorical_cols:
        baseline_counts = baseline_df[col].value_counts().to_dict()
        current_counts = current_df[col].value_counts().to_dict()
        result = chi_squared_drift(
            baseline_counts, current_counts,
            p_value_threshold=thresholds.get("chi_squared", 0.05),
        )
        result["metric_name"] = f"chi_squared__{col}"
        results.append(result)

    for col in numerical_cols:
        result = ks_drift(
            baseline_df[col].dropna().values,
            current_df[col].dropna().values,
            p_value_threshold=thresholds.get("ks_test", 0.05),
        )
        result["metric_name"] = f"ks_test__{col}"
        results.append(result)

    if baseline_probs is not None and current_probs is not None:
        result = prediction_confidence_shift(
            baseline_probs, current_probs,
            shift_threshold=thresholds.get("prediction_confidence_shift", 0.05),
        )
        results.append(result)

    return results
