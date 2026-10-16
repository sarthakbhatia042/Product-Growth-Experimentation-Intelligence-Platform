"""
Data Loader and Splitter for the Criteo Uplift Prediction Dataset.
Provides zero-copy DuckDB SQL access and stratified data splitting for causal modeling.
"""

import os
import logging
from typing import Tuple, Dict, Any, Optional
import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
EXPOSURE_COL = "exposure"
CONVERSION_COL = "conversion"
VISIT_COL = "visit"


def get_duckdb_connection(parquet_path: str) -> duckdb.DuckDBPyConnection:
    """
    Creates an in-memory DuckDB connection and registers the Parquet file
    as the 'criteo_data' view for out-of-core SQL queries.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE OR REPLACE VIEW criteo_data AS SELECT * FROM read_parquet('{parquet_path}');")
    return con


def load_criteo_dataframe(
    parquet_path: str,
    limit: Optional[int] = None,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Loads data from Parquet into a Pandas DataFrame using DuckDB with optional column pruning and limits.
    """
    con = get_duckdb_connection(parquet_path)
    col_str = ", ".join(columns) if columns else "*"
    limit_str = f"LIMIT {limit}" if limit else ""
    query = f"SELECT {col_str} FROM criteo_data {limit_str}"
    df = con.execute(query).fetchdf()
    con.close()
    return df


def split_criteo_data(
    df: pd.DataFrame,
    test_size: float = 0.30,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Splits the dataset into Train, Validation, and Test partitions.
    Uses compound stratification across [treatment, conversion] to strictly preserve
    both the assignment ratio (treatment:control) and rare outcome base rate (conversion).

    Returns
    -------
    dict
        Dictionary containing 'train', 'val', 'test' DataFrames.
    """
    # Create compound stratification key
    strat_key = df[TREATMENT_COL].astype(str) + "_" + df[CONVERSION_COL].astype(str)

    # First split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key
    )

    # Next split validation set from train
    rel_val_size = val_size / (1.0 - test_size)
    strat_key_train_val = train_val_df[TREATMENT_COL].astype(str) + "_" + train_val_df[CONVERSION_COL].astype(str)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=rel_val_size,
        random_state=random_state,
        stratify=strat_key_train_val
    )

    logger.info(
        f"Data split successfully: "
        f"Train={len(train_df):,} ({len(train_df)/len(df)*100:.1f}%), "
        f"Val={len(val_df):,} ({len(val_df)/len(df)*100:.1f}%), "
        f"Test={len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)"
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True)
    }
