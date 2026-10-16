"""
CLI Script: Causal ML & Uplift Modeling Pipeline.
Trains Baseline ATE, S-Learner, T-Learner, and X-Learner using LightGBM.
Evaluates on held-out test data with AUUC, Qini, Decile Lift, Bootstrap CIs,
and runs the Economic Decision Simulator.
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from src.data_pipeline.loader import load_criteo_dataframe, split_criteo_data
from src.uplift_models.base import BaselineATELearner
from src.uplift_models.s_learner import SLearner
from src.uplift_models.t_learner import TLearner
from src.uplift_models.x_learner import XLearner
from src.policy_eval.metrics import (
    compute_uplift_curve,
    calculate_auuc_and_qini,
    compute_decile_lift_table,
    bootstrap_uplift_uncertainty
)
from src.policy_eval.policy_simulator import simulate_targeting_policies
from src.reporting.generate_figures import plot_uplift_curves, plot_decile_lift, plot_economic_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [f"f{i}" for i in range(12)]


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate causal ML uplift models.")
    parser.add_argument("--data-path", default="data/processed/criteo_sample_250k.parquet", help="Path to input parquet")
    parser.add_argument("--conversion-value", type=float, default=50.0, help="Simulated customer value per conversion ($)")
    parser.add_argument("--contact-cost", type=float, default=0.10, help="Simulated cost per targeted impression ($)")
    parser.add_argument("--budget", type=float, default=15000.0, help="Simulated campaign budget ($)")
    parser.add_argument("--output-dir", default="docs/reports", help="Output directory for reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)

    logger.info(f"Loading data from {args.data_path}...")
    df = load_criteo_dataframe(args.data_path)

    logger.info("Performing compound stratified train / validation / test split (70% train+val, 30% test)...")
    splits = split_criteo_data(df, test_size=0.30, val_size=0.15, random_state=42)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["conversion"].values
    w_train = train_df["treatment"].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["conversion"].values
    w_test = test_df["treatment"].values

    logger.info(f"Training cohorts: Train={len(train_df):,}, Val={len(val_df):,}, Held-out Test={len(test_df):,}")

    models = {
        "Baseline ATE": BaselineATELearner(),
        "S-Learner": SLearner(estimator_type="lightgbm", n_estimators=80, learning_rate=0.05, max_depth=5),
        "T-Learner": TLearner(estimator_type="lightgbm", n_estimators=80, learning_rate=0.05, max_depth=5),
        "X-Learner": XLearner(estimator_type="lightgbm", n_estimators=60, learning_rate=0.05, max_depth=4)
    }

    curves_dict = {}
    model_benchmarks = []
    test_predictions = {}

    for name, model in models.items():
        logger.info(f"Fitting {name}...")
        model.fit(X_train, y_train, w_train)

        logger.info(f"Evaluating {name} on held-out test set ({len(test_df):,} samples)...")
        uplift_preds = model.predict_uplift(X_test)
        test_predictions[name] = uplift_preds

        curve_df = compute_uplift_curve(y_test, w_test, uplift_preds, n_bins=100)
        curves_dict[name] = curve_df

        metrics = calculate_auuc_and_qini(curve_df)
        logger.info(f"{name} -> AUUC: {metrics['auuc']:.4f}, Qini Score: {metrics['qini_score']:.4f}")

        model_benchmarks.append({
            "model_name": name,
            "auuc": metrics["auuc"],
            "qini_score": metrics["qini_score"],
            "mean_predicted_uplift": float(np.mean(uplift_preds)),
            "std_predicted_uplift": float(np.std(uplift_preds))
        })

    # Decile Lift Table for the best model (e.g. X-Learner or T-Learner)
    best_model_name = "X-Learner" if "X-Learner" in test_predictions else "T-Learner"
    logger.info(f"Computing decile lift table for {best_model_name}...")
    decile_df = compute_decile_lift_table(y_test, w_test, test_predictions[best_model_name], n_deciles=10)
    decile_table_path = os.path.join(args.output_dir, "decile_lift_table.csv")
    decile_df.to_csv(decile_table_path, index=False)
    logger.info(f"Decile table saved to {decile_table_path}")

    # Bootstrap Uncertainty on Held-out test set
    logger.info(f"Computing bootstrap confidence intervals for {best_model_name} on held-out test data...")
    boot_res = bootstrap_uplift_uncertainty(y_test, w_test, test_predictions[best_model_name], n_bootstraps=50)
    logger.info(f"Bootstrap Results for {best_model_name}: AUUC 95% CI = [{boot_res['auuc_ci_lower']:.2f}, {boot_res['auuc_ci_upper']:.2f}], Qini 95% CI = [{boot_res['qini_ci_lower']:.2f}, {boot_res['qini_ci_upper']:.2f}]")

    # Economic Decision Simulator
    logger.info("Executing Product Decision & Budget Simulator under simulated economics...")
    sim_res = simulate_targeting_policies(
        y_true=y_test,
        treatment=w_test,
        uplift_scores=test_predictions[best_model_name],
        conversion_value=args.conversion_value,
        contact_cost=args.contact_cost,
        budget=args.budget
    )
    logger.info("=== ECONOMIC POLICY COMPARISON (SIMULATED SCENARIO) ===")
    logger.info("\n" + str(sim_res["comparison_table"][["policy_name", "target_fraction", "units_targeted", "incremental_conversions", "total_cost", "net_profit", "roi_pct"]]))

    # Generate Figures
    logger.info("Generating publication figures...")
    plot_uplift_curves(curves_dict, "docs/figures/fig5_uplift_curves_comparison.png")
    plot_decile_lift(decile_df, "docs/figures/fig6_decile_incremental_lift.png")
    plot_economic_simulation(sim_res["curve_df"], budget=args.budget, output_path="docs/figures/fig7_economic_policy_profit_curves.png")

    # Save benchmark metrics
    summary_export = {
        "model_benchmarks": model_benchmarks,
        "best_model": best_model_name,
        "bootstrap_uncertainty": boot_res,
        "break_even_lift_pct": sim_res["break_even_lift_pct"],
        "optimal_policy": {
            "name": sim_res["optimal_policy"].policy_name,
            "target_fraction": sim_res["optimal_policy"].target_fraction,
            "units_targeted": sim_res["optimal_policy"].units_targeted,
            "incremental_conversions": sim_res["optimal_policy"].incremental_conversions,
            "total_cost": sim_res["optimal_policy"].total_cost,
            "net_profit": sim_res["optimal_policy"].net_profit,
            "roi_pct": sim_res["optimal_policy"].roi_pct
        },
        "economic_assumptions": {
            "conversion_value": args.conversion_value,
            "contact_cost": args.contact_cost,
            "budget": args.budget,
            "disclaimer": "SIMULATED SCENARIO - NOT OBSERVED REVENUE"
        }
    }

    benchmarks_path = os.path.join(args.output_dir, "model_benchmarks.json")
    with open(benchmarks_path, "w") as f:
        json.dump(summary_export, f, indent=2)

    sim_res["comparison_table"].to_csv(os.path.join(args.output_dir, "policy_comparison_table.csv"), index=False)
    logger.info(f"Model benchmarks exported to {benchmarks_path}")


if __name__ == "__main__":
    main()
