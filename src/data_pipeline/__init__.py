"""Data Pipeline package for Criteo Uplift modeling."""
from .downloader import stream_sample_to_parquet, download_full_criteo
from .loader import get_duckdb_connection, load_criteo_dataframe, split_criteo_data
from .quality_checks import run_data_quality_audit

__all__ = [
    "stream_sample_to_parquet",
    "download_full_criteo",
    "get_duckdb_connection",
    "load_criteo_dataframe",
    "split_criteo_data",
    "run_data_quality_audit",
]
