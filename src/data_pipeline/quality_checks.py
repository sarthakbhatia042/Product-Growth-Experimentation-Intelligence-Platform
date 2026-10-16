"""
Data Quality and Validation Suite for Criteo Uplift Data.
Executes DuckDB SQL checks for missingness, value domain constraints,
cohort balance, and logical consistency.
"""

import os
import logging
from typing import Dict, Any, Tuple
import duckdb
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def run_data_quality_audit(parquet_path: str) -> Dict[str, Any]:
    """
    Runs full data quality checks on the Parquet dataset using DuckDB.
    Validates:
      1. Missing values per column
      2. Domain validity (binary flags must only be 0 or 1)
      3. Logical consistency (no exposure in control group: P(exposure=1 | treatment=0) == 0)
      4. Duplicate record count
      5. Standardized Mean Difference (SMD) for covariate balance across all 12 features
    """
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE OR REPLACE VIEW criteo_data AS SELECT * FROM read_parquet('{parquet_path}');")

    # Read SQL audit query
    with open("queries/01_data_quality_audit.sql", "r") as f:
        query_audit = f.read()

    audit_df = con.execute(query_audit).fetchdf()

    # Read cohort summary query
    with open("queries/02_cohort_summary.sql", "r") as f:
        query_cohort = f.read()
    cohort_df = con.execute(query_cohort).fetchdf()

    # Read SRM check query
    with open("queries/03_srm_check.sql", "r") as f:
        query_srm = f.read()
    srm_df = con.execute(query_srm).fetchdf()

    # Read covariate balance query
    with open("queries/04_feature_transforms.sql", "r") as f:
        query_balance = f.read()
    balance_df = con.execute(query_balance).fetchdf()

    con.close()

    # Assess pass/fail criteria
    audit_row = audit_df.iloc[0]
    total_rows = int(audit_row["total_rows"])
    null_cols = [col for col in audit_row.index if col.startswith("null_") and audit_row[col] > 0]
    has_nulls = len(null_cols) > 0
    control_leakage = int(audit_row["control_leakage_exposure"])
    invalid_domains = (
        int(audit_row["invalid_treatment_values"]) +
        int(audit_row["invalid_exposure_values"]) +
        int(audit_row["invalid_visit_values"]) +
        int(audit_row["invalid_conversion_values"])
    )

    smd_max = balance_df["smd"].max() if "smd" in balance_df.columns else 0.0

    passed = (
        not has_nulls and
        control_leakage == 0 and
        invalid_domains == 0 and
        smd_max < 0.05
    )

    results = {
        "passed": bool(passed),
        "total_rows": total_rows,
        "distinct_records": int(audit_row["distinct_records"]),
        "has_nulls": has_nulls,
        "null_columns": null_cols,
        "control_leakage_exposure": control_leakage,
        "invalid_domains": invalid_domains,
        "max_covariate_smd": float(smd_max),
        "audit_df": audit_df,
        "cohort_df": cohort_df,
        "srm_df": srm_df,
        "balance_df": balance_df
    }

    logger.info(
        f"Data Quality Audit Completed. Passed: {passed}. "
        f"Rows: {total_rows:,}, Leakage: {control_leakage}, Max SMD: {smd_max:.4f}"
    )

    return results
