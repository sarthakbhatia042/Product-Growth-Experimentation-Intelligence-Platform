"""
Unit tests for Synthetic Data Generation and Statistical Verification Suite.
"""

import pytest
import numpy as np
from src.simulation.synthetic_validator import (
    generate_synthetic_cate_dataset,
    verify_estimator_properties,
    verify_false_positive_rate_under_null,
    verify_confidence_interval_coverage
)
from src.uplift_models.t_learner import TLearner


def test_synthetic_cate_generator():
    # Null heterogeneity
    df_null, true_tau_null, mu0_null = generate_synthetic_cate_dataset(
        n_samples=500, heterogeneity_type="null", random_seed=42
    )
    assert len(df_null) == 500
    assert np.all(true_tau_null == 0.0)
    assert np.all(mu0_null > 0.0) and np.all(mu0_null < 1.0)

    # Nonlinear heterogeneity
    df_nl, true_tau_nl, mu0_nl = generate_synthetic_cate_dataset(
        n_samples=500, heterogeneity_type="nonlinear", random_seed=42
    )
    assert np.std(true_tau_nl) > 0.0  # true treatment effect is heterogeneous
    # Ensure outcomes are binary
    assert set(df_nl["y"].unique()).issubset({0, 1})
    assert set(df_nl["treatment"].unique()).issubset({0, 1})


def test_estimator_verification_on_synthetic_ground_truth():
    learner = TLearner(estimator_type="lightgbm", n_estimators=15, max_depth=3, random_state=42)
    res = verify_estimator_properties(learner, n_samples=3000, n_features=6, random_seed=42)

    assert "estimator_bias" in res
    assert "pehe_rmse" in res
    assert "spearman_rank_correlation" in res
    assert "kendall_tau" in res
    # Bias should be reasonably bounded
    assert abs(res["estimator_bias"]) < 0.05
    # Ranking correlation should be positive
    assert res["spearman_rank_correlation"] > 0.20


def test_ci_coverage_statistical_verification():
    # Fast test with 30 simulations
    cov_res = verify_confidence_interval_coverage(num_simulations=30, sample_size=1500, nominal_confidence=0.95, random_seed=42)
    assert 0.80 <= cov_res["empirical_coverage_rate"] <= 1.0
    assert cov_res["is_valid_coverage"] is True


def test_false_positive_rate_under_null_verification():
    # Fast test with 40 A/A simulations
    fpr_res = verify_false_positive_rate_under_null(num_simulations=40, sample_size=1000, alpha=0.05, random_seed=42)
    # Empirical FPR should be within reasonable sampling error bounds of 0.05
    assert 0.0 <= fpr_res["empirical_fpr"] <= 0.15
