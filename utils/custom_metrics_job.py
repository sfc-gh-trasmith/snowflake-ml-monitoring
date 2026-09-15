"""custom_metrics_job.py -- ML Job for computing custom drift metrics."""

import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
from snowflake.snowpark import Session

# The ML Job runtime provides a pre-configured session
def get_session():
    return Session.builder.getOrCreate()

def chi_squared_test(baseline, current, col):
    baseline_counts = baseline[col].value_counts()
    current_counts = current[col].value_counts()
    all_cats = sorted(set(baseline_counts.index) | set(current_counts.index))
    observed = np.array([current_counts.get(c, 0) for c in all_cats], dtype=float)
    expected_raw = np.array([baseline_counts.get(c, 0) for c in all_cats], dtype=float)
    expected = expected_raw * (observed.sum() / expected_raw.sum())
    expected = np.where(expected == 0, 1e-10, expected)
    stat, p_val = stats.chisquare(observed, f_exp=expected)
    return float(stat), float(p_val)

def ks_test(baseline, current, col):
    stat, p_val = stats.ks_2samp(
        baseline[col].dropna().values.astype(float),
        current[col].dropna().values.astype(float)
    )
    return float(stat), float(p_val)

def main():
    session = get_session()

    categorical_cols = ["CONTRACT_TYPE", "INTERNET_SERVICE"]
    numerical_cols = ["TENURE_MONTHS", "MONTHLY_CHARGES", "TOTAL_CHARGES", "NUM_SUPPORT_TICKETS"]
    p_threshold = 0.05

    baseline = session.table("ML_DEMO.ML_CHURN.CHURN_TRAIN").to_pandas()
    current = session.table("ML_DEMO.ML_CHURN.CHURN_CURRENT").to_pandas()

    results = []

    for col in categorical_cols:
        stat, p_val = chi_squared_test(baseline, current, col)
        results.append({
            "metric_name": f"chi_squared__{col}",
            "metric_value": stat,
            "p_value": p_val,
            "threshold": p_threshold,
            "is_alert": p_val < p_threshold,
        })

    for col in numerical_cols:
        stat, p_val = ks_test(baseline, current, col)
        results.append({
            "metric_name": f"ks_test__{col}",
            "metric_value": stat,
            "p_value": p_val,
            "threshold": p_threshold,
            "is_alert": p_val < p_threshold,
        })

    for r in results:
        session.sql(f"""
            INSERT INTO ML_DEMO.ML_CHURN.CUSTOM_METRICS
                (METRIC_NAME, METRIC_VALUE, P_VALUE, THRESHOLD, IS_ALERT, MODEL_NAME, MODEL_VERSION, DETAILS)
            SELECT
                \'{r["metric_name"]}\',
                {r["metric_value"]},
                {r["p_value"]},
                {r["threshold"]},
                {str(r["is_alert"]).upper()},
                \'CHURN_MODEL\',
                \'V1\',
                NULL
        """).collect()

    drift_count = sum(1 for r in results if r["is_alert"])
    print(f"Computed {len(results)} metrics. {drift_count} drift alerts.")

if __name__ == "__main__":
    main()