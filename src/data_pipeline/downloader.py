"""
Criteo Uplift Dataset Downloader & Ingestion Utility.
Downloads or streams the official Criteo Uplift Prediction Dataset (v2.1)
from the Hugging Face / Criteo research repository.
"""

import os
import gzip
import logging
import urllib.request
from typing import Optional
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"

EXPECTED_COLUMNS = [
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11",
    "treatment", "conversion", "visit", "exposure"
]


def stream_sample_to_parquet(
    url: str = DEFAULT_URL,
    output_parquet_path: str = "data/processed/criteo_sample_250k.parquet",
    sample_size: int = 250000,
    chunk_size: int = 50000
) -> str:
    """
    Streams a sample of rows directly from the official compressed CSV stream
    and saves as an optimized Apache Parquet file using DuckDB, avoiding
    storing the full 1.5GB uncompressed file on disk.

    Parameters
    ----------
    url : str
        URL to the criteo-research-uplift-v2.1.csv.gz file.
    output_parquet_path : str
        Target destination for the parquet file.
    sample_size : int
        Number of valid data records to extract.
    chunk_size : int
        Batch size for streaming chunks.

    Returns
    -------
    str
        Path to the generated parquet file.
    """
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    temp_csv_path = output_parquet_path.replace(".parquet", "_temp.csv")

    logger.info(f"Connecting to official Criteo dataset stream at {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CriteoExperimentationBot/1.0)"})

    rows_collected = 0
    with urllib.request.urlopen(req, timeout=60) as response:
        with gzip.GzipFile(fileobj=response) as gz_stream:
            header_line = gz_stream.readline().decode("utf-8").strip()
            cols = header_line.split(",")
            if cols != EXPECTED_COLUMNS:
                logger.warning(f"Header columns differ from expected! Got: {cols}")

            with open(temp_csv_path, "w", encoding="utf-8") as temp_out:
                temp_out.write(header_line + "\n")
                while rows_collected < sample_size:
                    line = gz_stream.readline()
                    if not line:
                        break
                    temp_out.write(line.decode("utf-8"))
                    rows_collected += 1
                    if rows_collected % chunk_size == 0:
                        logger.info(f"Streamed {rows_collected:,} / {sample_size:,} rows...")

    logger.info(f"Finished streaming {rows_collected:,} records to temporary staging. Converting to Parquet via DuckDB...")
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT
                CAST(f0 AS DOUBLE) AS f0,
                CAST(f1 AS DOUBLE) AS f1,
                CAST(f2 AS DOUBLE) AS f2,
                CAST(f3 AS DOUBLE) AS f3,
                CAST(f4 AS DOUBLE) AS f4,
                CAST(f5 AS DOUBLE) AS f5,
                CAST(f6 AS DOUBLE) AS f6,
                CAST(f7 AS DOUBLE) AS f7,
                CAST(f8 AS DOUBLE) AS f8,
                CAST(f9 AS DOUBLE) AS f9,
                CAST(f10 AS DOUBLE) AS f10,
                CAST(f11 AS DOUBLE) AS f11,
                CAST(treatment AS INT64) AS treatment,
                CAST(conversion AS INT64) AS conversion,
                CAST(visit AS INT64) AS visit,
                CAST(exposure AS INT64) AS exposure
            FROM read_csv_auto('{temp_csv_path}', header=True)
        ) TO '{output_parquet_path}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');
    """)
    con.close()

    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    file_size_mb = os.path.getsize(output_parquet_path) / (1024 * 1024)
    logger.info(f"Successfully created {output_parquet_path} ({file_size_mb:.2f} MB) with {rows_collected:,} rows.")
    return output_parquet_path


def download_full_criteo(
    url: str = DEFAULT_URL,
    dest_path: str = "data/raw/criteo-research-uplift-v2.1.csv.gz"
) -> str:
    """
    Downloads the entire official Criteo dataset archive with progress tracking.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        logger.info(f"File already exists at {dest_path} ({os.path.getsize(dest_path)/(1024*1024):.1f} MB). Skipping download.")
        return dest_path

    logger.info(f"Initiating full download from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_sz = 1024 * 1024  # 1MB chunks
        while True:
            chunk = response.read(chunk_sz)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0 and downloaded % (20 * chunk_sz) == 0:
                logger.info(f"Downloaded {downloaded/(1024*1024):.1f} / {total_size/(1024*1024):.1f} MB ({downloaded*100/total_size:.1f}%)")

    logger.info(f"Completed full download: {dest_path}")
    return dest_path
