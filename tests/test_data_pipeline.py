"""
Unit tests for Data Pipeline, DuckDB Ingestion, and Quality Checks.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.data_pipeline.loader import load_criteo_dataframe, split_criteo_data
from src.data_pipeline.quality_checks import run_data_quality_audit


@pytest.fixture
def sample_parquet_path():
    path = "data/processed/criteo_sample_250k.parquet"
    if not os.path.exists(path):
        pytest.skip("Sample parquet dataset not found. Run data preparation first.")
    return path


def test_data_loader_and_schema(sample_parquet_path):
    df = load_criteo_dataframe(sample_parquet_path, limit=1000)
    assert len(df) == 1000
    expected_cols = [f"f{i}" for i in range(12)] + ["treatment", "conversion", "visit", "exposure"]
    for col in expected_cols:
        assert col in df.columns
    # Check domain values
    assert set(df["treatment"].unique()).issubset({0, 1})
    assert set(df["conversion"].unique()).issubset({0, 1})
    assert set(df["visit"].unique()).issubset({0, 1})
    assert set(df["exposure"].unique()).issubset({0, 1})


def test_compound_stratified_split(sample_parquet_path):
    df = load_criteo_dataframe(sample_parquet_path, limit=10000)
    splits = split_criteo_data(df, test_size=0.30, val_size=0.15, random_state=42)

    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    assert len(train) + len(val) + len(test) == len(df)

    # Verify compound stratification preserves treatment ratio across splits within 2 percentage points
    t_rate_full = df["treatment"].mean()
    assert abs(train["treatment"].mean() - t_rate_full) < 0.02
    assert abs(val["treatment"].mean() - t_rate_full) < 0.02
    assert abs(test["treatment"].mean() - t_rate_full) < 0.02


def test_data_quality_audit_logic(sample_parquet_path):
    audit = run_data_quality_audit(sample_parquet_path)
    assert audit["total_rows"] > 0
    assert audit["has_nulls"] is False
    assert audit["control_leakage_exposure"] == 0
    assert audit["invalid_domains"] == 0
    assert audit["max_covariate_smd"] < 0.05
    assert audit["passed"] is True
