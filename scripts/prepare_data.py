"""
CLI Script: Data Preparation and Validation Pipeline.
Extracts a stratified sample from the official Criteo dataset and runs data quality checks.
"""

import os
import argparse
import logging
import yaml
import duckdb
from src.data_pipeline.quality_checks import run_data_quality_audit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare Criteo Uplift dataset and validate schema integrity.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--sample-size", type=int, default=250000, help="Sample size for fast processing")
    parser.add_argument("--raw-file", default="data/raw/criteo-research-uplift-v2.1.csv.gz", help="Path to raw csv.gz file")
    parser.add_argument("--output-parquet", default="data/processed/criteo_sample_250k.parquet", help="Output parquet path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)

    if not os.path.exists(args.raw_file):
        raise FileNotFoundError(f"Raw Criteo dataset not found at {args.raw_file}. Please run download first.")

    logger.info(f"Connecting to DuckDB to extract {args.sample_size:,} reservoir sample from {args.raw_file}...")
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
            FROM read_csv_auto('{args.raw_file}', header=True)
            USING SAMPLE {args.sample_size} (reservoir, 42)
        ) TO '{args.output_parquet}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');
    """)
    con.close()

    logger.info(f"Parquet sample generated at {args.output_parquet}. Running comprehensive DuckDB data quality audit...")
    audit_results = run_data_quality_audit(args.output_parquet)

    logger.info("=== AUDIT SUMMARY ===")
    logger.info(f"Passed: {audit_results['passed']}")
    logger.info(f"Total Rows: {audit_results['total_rows']:,}")
    logger.info(f"Distinct Records: {audit_results['distinct_records']:,}")
    logger.info(f"Exposure Leakage in Control: {audit_results['control_leakage_exposure']}")
    logger.info(f"Max Covariate Standardized Mean Difference: {audit_results['max_covariate_smd']:.4f}")

    if not audit_results["passed"]:
        logger.error("Data quality audit failed! Check schema and cohort definitions.")
    else:
        logger.info("All data quality checks passed successfully!")


if __name__ == "__main__":
    main()
