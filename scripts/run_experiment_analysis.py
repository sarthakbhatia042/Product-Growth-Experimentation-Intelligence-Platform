"""
CLI Script: Experimentation & Statistical Inference Engine Runner.
Performs SRM testing, A/B test inference (ITT), CACE/LATE estimation,
Power/MDE curves, and sequential peeking risk analysis.
"""

import os
import json
import logging
import argparse
import pandas as pd
from src.data_pipeline.loader import load_criteo_dataframe
from src.experimentation.engine import analyze_binary_experiment, estimate_cace, adjust_multiple_testing
from src.experimentation.srm import check_sample_ratio_mismatch
from src.experimentation.sequential import simulate_peeking_risk
from src.power_analysis.power_calc import calculate_mde, generate_mde_curves
from src.reporting.generate_figures import plot_cohort_lift, plot_mde_sensitivity, plot_sequential_peeking

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run A/B experiment and statistical inference.")
    parser.add_argument("--data-path", default="data/processed/criteo_sample_250k.parquet", help="Path to input parquet")
    parser.add_argument("--expected-ratio", type=float, default=0.8460, help="Expected treatment allocation ratio")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--output-dir", default="docs/reports", help="Output directory for reports and metrics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)

    logger.info(f"Loading experiment data from {args.data_path}...")
    df = load_criteo_dataframe(args.data_path)

    n_treated = int((df["treatment"] == 1).sum())
    n_control = int((df["treatment"] == 0).sum())

    # 1. SRM Check
    logger.info("Executing Sample Ratio Mismatch (SRM) Test...")
    srm_res = check_sample_ratio_mismatch(
        observed_treatment=n_treated,
        observed_control=n_control,
        expected_ratio_treatment=args.expected_ratio
    )
    logger.info(f"SRM Result: Detected={srm_res.srm_detected}, Chi2={srm_res.chi2_statistic:.4f}, p={srm_res.p_value:.4e}, Severity={srm_res.severity}")

    # 2. Conversion Outcome (Primary)
    logger.info("Analyzing Primary Outcome: Conversion...")
    conv_metrics = analyze_binary_experiment(df, outcome_col="conversion", alpha=args.alpha)
    logger.info(
        f"Conversion: Control={conv_metrics.control_rate*100:.4f}%, "
        f"Treated={conv_metrics.treatment_rate*100:.4f}%, "
        f"Abs Lift={conv_metrics.absolute_lift*100:+.4f}% (95% CI: [{conv_metrics.absolute_lift_ci_lower*100:+.4f}%, {conv_metrics.absolute_lift_ci_upper*100:+.4f}%]), "
        f"Rel Lift={conv_metrics.relative_lift*100:+.2f}%, p={conv_metrics.p_value:.4e}"
    )

    # 3. Visit Outcome (Secondary)
    logger.info("Analyzing Secondary Outcome: Visit...")
    visit_metrics = analyze_binary_experiment(df, outcome_col="visit", alpha=args.alpha)
    logger.info(
        f"Visit: Control={visit_metrics.control_rate*100:.2f}%, "
        f"Treated={visit_metrics.treatment_rate*100:.2f}%, "
        f"Abs Lift={visit_metrics.absolute_lift*100:+.4f}%, "
        f"Rel Lift={visit_metrics.relative_lift*100:+.2f}%, p={visit_metrics.p_value:.4e}"
    )

    # 4. Multiple Testing Correction across primary & secondary endpoints
    mult_res = adjust_multiple_testing(
        [conv_metrics.p_value, visit_metrics.p_value],
        method="fdr_bh",
        alpha=args.alpha
    )
    logger.info(f"Multiple Testing Adjustments (Benjamini-Hochberg): {mult_res}")

    # 5. CACE / LATE Estimation
    logger.info("Estimating Complier Average Causal Effect (CACE / LATE)...")
    cace_res = estimate_cace(df, outcome_col="conversion", alpha=args.alpha)
    logger.info(
        f"CACE: ITT={cace_res.itt_effect*100:+.4f}%, "
        f"Compliance={cace_res.compliance_rate*100:.2f}%, "
        f"CACE Estimate={cace_res.cace_estimate*100:+.4f}% (95% CI: [{cace_res.cace_ci_lower*100:+.4f}%, {cace_res.cace_ci_upper*100:+.4f}%]), "
        f"p={cace_res.p_value:.4e}"
    )

    # 6. Power Analysis & MDE Curves
    logger.info("Computing MDE Sensitivity Curves...")
    mde_df = generate_mde_curves(p_control=conv_metrics.control_rate, treatment_ratio=args.expected_ratio)
    mde_calc = calculate_mde(p_control=conv_metrics.control_rate, n_treatment=n_treated, n_control=n_control)

    # 7. Sequential Peeking Risk Simulation
    logger.info("Simulating Sequential Peeking Risk under Null Hypothesis...")
    peeking_res = simulate_peeking_risk(
        sample_size=10000,
        base_rate=conv_metrics.control_rate,
        num_peeks=15,
        num_simulations=300,
        alpha=args.alpha
    )
    logger.info(
        f"Peeking Risk: Nominal Alpha={args.alpha*100:.1f}%, "
        f"Empirical Peeking FPR={peeking_res['empirical_peeking_fpr']*100:.1f}%, "
        f"Inflation Factor={peeking_res['error_inflation_factor']:.2f}x"
    )

    # 8. Plot Figures
    logger.info("Generating publication figures...")
    plot_cohort_lift(conv_metrics, visit_metrics, "docs/figures/fig1_cohort_conversion_rates.png")
    plot_mde_sensitivity(mde_df, "docs/figures/fig3_mde_sensitivity_curves.png")
    plot_sequential_peeking(peeking_res, "docs/figures/fig4_sequential_peeking_risk.png")

    # 9. Export Metrics Summary
    summary_data = {
        "sample_size": len(df),
        "treatment_units": n_treated,
        "control_units": n_control,
        "srm_check": {
            "chi2": srm_res.chi2_statistic,
            "p_value": srm_res.p_value,
            "srm_detected": srm_res.srm_detected,
            "severity": srm_res.severity
        },
        "conversion_experiment": {
            "control_rate": conv_metrics.control_rate,
            "treatment_rate": conv_metrics.treatment_rate,
            "absolute_lift": conv_metrics.absolute_lift,
            "absolute_lift_ci": [conv_metrics.absolute_lift_ci_lower, conv_metrics.absolute_lift_ci_upper],
            "relative_lift": conv_metrics.relative_lift,
            "relative_lift_ci": [conv_metrics.relative_lift_ci_lower, conv_metrics.relative_lift_ci_upper],
            "p_value": conv_metrics.p_value,
            "statistically_significant": conv_metrics.statistically_significant
        },
        "visit_experiment": {
            "control_rate": visit_metrics.control_rate,
            "treatment_rate": visit_metrics.treatment_rate,
            "absolute_lift": visit_metrics.absolute_lift,
            "relative_lift": visit_metrics.relative_lift,
            "p_value": visit_metrics.p_value
        },
        "cace_estimate": {
            "itt_effect": cace_res.itt_effect,
            "compliance_rate": cace_res.compliance_rate,
            "cace": cace_res.cace_estimate,
            "cace_ci": [cace_res.cace_ci_lower, cace_res.cace_ci_upper],
            "p_value": cace_res.p_value
        },
        "mde_achieved": mde_calc,
        "sequential_peeking_fpr": peeking_res["empirical_peeking_fpr"]
    }

    metrics_path = os.path.join(args.output_dir, "experiment_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Experiment metrics exported to {metrics_path}")


if __name__ == "__main__":
    main()
