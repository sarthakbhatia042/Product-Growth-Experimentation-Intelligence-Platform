"""
Unit tests for Policy Evaluation, Uplift Metrics (AUUC, Qini), and Economic Decision Simulator.
"""

import pytest
import numpy as np
import pandas as pd
from src.policy_eval.metrics import (
    compute_uplift_curve,
    calculate_auuc_and_qini,
    compute_decile_lift_table
)
from src.policy_eval.policy_simulator import simulate_targeting_policies


@pytest.fixture
def evaluation_toy_data():
    rng = np.random.default_rng(100)
    n = 5000
    w = rng.binomial(1, 0.5, size=n)
    # Give high scores to units where w=1 converted
    true_uplift = rng.normal(0.02, 0.05, size=n)
    y = np.where(w == 1, rng.binomial(1, np.clip(0.05 + true_uplift, 0.01, 0.99)), rng.binomial(1, 0.05))
    scores = true_uplift + rng.normal(0, 0.02, size=n)
    return y, w, scores


def test_uplift_curve_and_qini(evaluation_toy_data):
    y, w, scores = evaluation_toy_data
    curve = compute_uplift_curve(y, w, scores, n_bins=20)

    assert len(curve) == 21  # 0.0 + 20 bins
    assert curve.iloc[0]["fraction_targeted"] == 0.0
    assert curve.iloc[-1]["fraction_targeted"] == 1.0

    metrics = calculate_auuc_and_qini(curve)
    assert "auuc" in metrics
    assert "qini_score" in metrics
    assert np.isfinite(metrics["auuc"])
    assert np.isfinite(metrics["qini_score"])


def test_decile_lift_table(evaluation_toy_data):
    y, w, scores = evaluation_toy_data
    deciles = compute_decile_lift_table(y, w, scores, n_deciles=10)

    assert len(deciles) == 10
    assert list(deciles["decile"]) == list(range(1, 11))
    assert deciles["total_users"].sum() == len(y)
    for col in ["treatment_users", "control_users", "treatment_rate_pct", "control_rate_pct", "observed_lift_pct"]:
        assert col in deciles.columns


def test_economic_policy_simulator(evaluation_toy_data):
    y, w, scores = evaluation_toy_data
    conv_val = 100.0
    cost = 0.50
    budget = 1000.0  # Allows targeting up to 2,000 units (40% of 5000)

    sim = simulate_targeting_policies(
        y_true=y,
        treatment=w,
        uplift_scores=scores,
        conversion_value=conv_val,
        contact_cost=cost,
        budget=budget,
        step_pct=0.10
    )

    # Break-even lift should be cost / conversion_value = 0.50 / 100.0 = 0.005 (0.5%)
    assert pytest.approx(sim["break_even_lift"], rel=1e-4) == 0.005
    assert sim["break_even_lift_pct"] == 0.5

    opt = sim["optimal_policy"]
    assert opt.total_cost <= budget
    assert opt.is_budget_feasible is True

    # Treat none policy should have 0 profit, 0 cost
    treat_none = sim["treat_none_policy"]
    assert treat_none.total_cost == 0.0
    assert treat_none.net_profit == 0.0

    # Comparison table must contain all 4 standard policies
    comp_df = sim["comparison_table"]
    policy_names = set(comp_df["policy_name"])
    assert "Optimal Model-Guided Uplift Policy" in policy_names
    assert "Treat All (Indiscriminate Targeting)" in policy_names
    assert "Treat None (Status Quo)" in policy_names
    assert "Random Targeting (Under Budget)" in policy_names
