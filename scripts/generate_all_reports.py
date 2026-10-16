"""
CLI Orchestrator: End-to-End Pipeline Runner.
Executes the full pipeline:
  1. Data quality validation
  2. A/B experiment inference & SRM
  3. Causal ML uplift modeling & decision simulation
  4. Synthetic ground truth verification
  5. Publication figures and summary tables
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_cmd(cmd: list):
    logger.info(f"Running: {' '.join(cmd)}")
    env = dict(os.environ, PYTHONPATH=".")
    res = subprocess.run(cmd, env=env, check=True)
    if res.returncode != 0:
        logger.error(f"Command failed with code {res.returncode}")
        sys.exit(res.returncode)


def main():
    python_bin = sys.executable

    logger.info("================================================================================")
    logger.info("STARTING PRODUCT GROWTH & EXPERIMENTATION INTELLIGENCE PIPELINE")
    logger.info("================================================================================")

    # Step 1: Ensure dataset sample exists and is validated
    if not os.path.exists("data/processed/criteo_sample_250k.parquet"):
        logger.info("Step 1: Preparing and validating Criteo dataset sample...")
        run_cmd([python_bin, "scripts/prepare_data.py"])
    else:
        logger.info("Step 1: Processed Criteo sample parquet already exists. Proceeding...")

    # Step 2: Experimentation and Statistical Inference
    logger.info("Step 2: Executing A/B Experimentation Engine & SRM...")
    run_cmd([python_bin, "scripts/run_experiment_analysis.py"])

    # Step 3: Uplift Modeling & Decision Simulator
    logger.info("Step 3: Training Causal ML Uplift Models & Simulating Product Economics...")
    run_cmd([python_bin, "scripts/run_uplift_models.py"])

    # Step 4: Synthetic Ground Truth Verification
    logger.info("Step 4: Running Synthetic Ground Truth Verification Suite...")
    run_cmd([python_bin, "scripts/run_synthetic_benchmarks.py"])

    logger.info("================================================================================")
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("All publication figures generated in docs/figures/")
    logger.info("All metrics and result tables exported to docs/reports/")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
