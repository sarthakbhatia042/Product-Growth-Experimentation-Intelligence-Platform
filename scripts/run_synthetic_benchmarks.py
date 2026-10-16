"""
CLI Script: Synthetic Benchmark & Statistical Verification Suite.
Evaluates causal estimators against known ground truth CATE:
  - Estimator Bias & PEHE RMSE
  - False Positive Rate (A/A test) under the Null
  - Confidence Interval Coverage
  - Uplift Ranking Recovery (Spearman / Kendall)
"""

import os
import json
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.simulation.synthetic_validator import (
    generate_synthetic_cate_dataset,
    verify_estimator_properties,
    verify_false_positive_rate_under_null,
    verify_confidence_interval_coverage
)
from src.uplift_models.s_learner import SLearner
from src.uplift_models.t_learner import TLearner
from src.uplift_models.x_learner import XLearner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run synthetic ground truth causal verification suite.")
    parser.add_argument("--num-samples", type=int, default=15000, help="Number of synthetic samples")
    parser.add_argument("--output-dir", default="docs/reports", help="Output directory for reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)

    logger.info("Executing Statistical Verification: A/A False Positive Rate (FPR) under Null Hypothesis...")
    fpr_res = verify_false_positive_rate_under_null(num_simulations=150, sample_size=3000, alpha=0.05)
    logger.info(
        f"FPR Validation: Nominal Alpha={fpr_res['nominal_alpha']*100:.1f}%, "
        f"Empirical FPR={fpr_res['empirical_fpr']*100:.2f}%, "
        f"Calibration p-value={fpr_res['fpr_calibration_p_value']:.4f}, Calibrated={fpr_res['is_calibrated']}"
    )

    logger.info("Executing Statistical Verification: 95% Confidence Interval Coverage Test...")
    coverage_res = verify_confidence_interval_coverage(num_simulations=100, sample_size=4000, nominal_confidence=0.95)
    logger.info(
        f"CI Coverage: Nominal Confidence={coverage_res['nominal_confidence']*100:.1f}%, "
        f"Empirical Coverage={coverage_res['empirical_coverage_rate']*100:.2f}%, Valid={coverage_res['is_valid_coverage']}"
    )

    logger.info("Benchmarking Causal Learners against Known Ground Truth Heterogeneous Effects (Nonlinear DGP)...")
    learners = {
        "S-Learner": SLearner(estimator_type="lightgbm", n_estimators=60, learning_rate=0.05, max_depth=4),
        "T-Learner": TLearner(estimator_type="lightgbm", n_estimators=60, learning_rate=0.05, max_depth=4),
        "X-Learner": XLearner(estimator_type="lightgbm", n_estimators=50, learning_rate=0.05, max_depth=4)
    }

    learner_results = {}
    best_learner_data = None

    for name, learner in learners.items():
        logger.info(f"Evaluating {name} on synthetic benchmark...")
        eval_metrics = verify_estimator_properties(learner, n_samples=args.num_samples, n_features=6)
        logger.info(
            f"{name} -> Bias: {eval_metrics['estimator_bias']:+.4f}, "
            f"PEHE RMSE: {eval_metrics['pehe_rmse']:.4f}, "
            f"Spearman Corr: {eval_metrics['spearman_rank_correlation']:.4f}, "
            f"Kendall's Tau: {eval_metrics['kendall_tau']:.4f}"
        )
        learner_results[name] = {
            "mean_true_tau": eval_metrics["mean_true_tau"],
            "mean_pred_tau": eval_metrics["mean_pred_tau"],
            "estimator_bias": eval_metrics["estimator_bias"],
            "pehe_rmse": eval_metrics["pehe_rmse"],
            "spearman_rank_correlation": eval_metrics["spearman_rank_correlation"],
            "kendall_tau": eval_metrics["kendall_tau"]
        }
        if name == "X-Learner" or best_learner_data is None:
            best_learner_data = eval_metrics

    # Plot True CATE vs Predicted Uplift
    fig_path = "docs/figures/fig8_synthetic_cate_ground_truth_recovery.png"
    plt.figure(figsize=(8, 6.5))
    true_sample = np.array(best_learner_data["true_tau_sample"])
    pred_sample = np.array(best_learner_data["pred_tau_sample"])

    plt.scatter(true_sample, pred_sample, alpha=0.45, color="#2563eb", edgecolors="none", s=28)
    min_val = min(true_sample.min(), pred_sample.min())
    max_val = max(true_sample.max(), pred_sample.max())
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="#dc2626", linewidth=2.0, label="Identity (Perfect Recovery)")

    plt.xlabel("True CATE (Ground Truth Treatment Effect)")
    plt.ylabel("Predicted Uplift (Estimated Treatment Effect)")
    plt.title(
        f"Synthetic Ground Truth Recovery (X-Learner)\n"
        f"Spearman Rank Corr: {best_learner_data['spearman_rank_correlation']:.3f} | PEHE RMSE: {best_learner_data['pehe_rmse']:.4f}"
    )
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Synthetic benchmark figure saved to {fig_path}")

    # Export full verification report
    full_report = {
        "false_positive_rate_verification": {
            "num_simulations": fpr_res["num_simulations"],
            "nominal_alpha": fpr_res["nominal_alpha"],
            "empirical_fpr": fpr_res["empirical_fpr"],
            "is_calibrated": fpr_res["is_calibrated"]
        },
        "ci_coverage_verification": {
            "num_simulations": coverage_res["num_simulations"],
            "nominal_confidence": coverage_res["nominal_confidence"],
            "empirical_coverage_rate": coverage_res["empirical_coverage_rate"],
            "is_valid_coverage": coverage_res["is_valid_coverage"]
        },
        "synthetic_learner_benchmarks": learner_results
    }

    report_path = os.path.join(args.output_dir, "synthetic_validation_results.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"Synthetic validation report exported to {report_path}")


if __name__ == "__main__":
    main()
