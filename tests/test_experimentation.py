"""
Unit and statistical tests for Experimentation Engine, SRM, and CACE.
"""

import pytest
import numpy as np
import pandas as pd
from src.experimentation.engine import (
    analyze_binary_experiment,
    calculate_relative_lift_ci,
    estimate_cace,
    adjust_multiple_testing
)
from src.experimentation.srm import check_sample_ratio_mismatch
from src.power_analysis.power_calc import (
    calculate_two_proportion_power,
    calculate_mde,
    calculate_required_sample_size
)


def test_delta_method_relative_lift_ci():
    p_t = 0.05
    n_t = 10000
    p_c = 0.04
    n_c = 10000

    rel_lift, ci_lower, ci_upper = calculate_relative_lift_ci(p_t, n_t, p_c, n_c, alpha=0.05)
    # Expected relative lift is (0.05 - 0.04) / 0.04 = 0.25 (+25%)
    assert pytest.approx(rel_lift, rel=1e-3) == 0.25
    assert ci_lower < rel_lift < ci_upper
    # Ensure lower bound is sensible
    assert ci_lower > -0.50


def test_analyze_binary_experiment_known_values():
    # Construct synthetic known cohort
    rng = np.random.default_rng(42)
    n = 20000
    treatment = rng.binomial(1, 0.5, size=n)
    # Treatment has higher conversion
    y = np.where(treatment == 1, rng.binomial(1, 0.06, size=n), rng.binomial(1, 0.04, size=n))
    df = pd.DataFrame({"treatment": treatment, "conversion": y})

    metrics = analyze_binary_experiment(df, outcome_col="conversion", alpha=0.05)
    assert metrics.control_count > 0
    assert metrics.treatment_count > 0
    assert metrics.absolute_lift > 0.01  # roughly 2% lift
    assert metrics.absolute_lift_ci_lower < metrics.absolute_lift < metrics.absolute_lift_ci_upper
    assert metrics.p_value < 0.001
    assert metrics.statistically_significant is True


def test_srm_detection():
    # 1. Perfectly matched allocation: 84.6% : 15.4%
    total = 100000
    n_t = int(total * 0.846)
    n_c = total - n_t
    res_clean = check_sample_ratio_mismatch(observed_treatment=n_t, observed_control=n_c, expected_ratio_treatment=0.846)
    assert res_clean.srm_detected is False
    assert res_clean.severity == "NONE"

    # 2. Skewed allocation (e.g. 50:50 instead of 84.6:15.4)
    res_skewed = check_sample_ratio_mismatch(observed_treatment=50000, observed_control=50000, expected_ratio_treatment=0.846)
    assert res_skewed.srm_detected is True
    assert res_skewed.severity == "CRITICAL"
    assert res_skewed.p_value < 1e-10


def test_cace_estimation():
    rng = np.random.default_rng(123)
    n = 15000
    treatment = rng.binomial(1, 0.5, size=n)
    # Control has 0 exposure. Treatment has 20% exposure
    exposure = np.where(treatment == 1, rng.binomial(1, 0.20, size=n), 0)
    # Causal effect exists only if exposed
    conversion = np.where(exposure == 1, rng.binomial(1, 0.25, size=n), rng.binomial(1, 0.05, size=n))
    df = pd.DataFrame({"treatment": treatment, "exposure": exposure, "conversion": conversion})

    cace_res = estimate_cace(df, outcome_col="conversion", alpha=0.05)
    assert cace_res.compliance_rate == pytest.approx(df[df["treatment"] == 1]["exposure"].mean(), rel=1e-3)
    # CACE = ITT / compliance
    expected_cace = cace_res.itt_effect / cace_res.compliance_rate
    assert pytest.approx(cace_res.cace_estimate, rel=1e-3) == expected_cace
    assert cace_res.cace_ci_lower < cace_res.cace_estimate < cace_res.cace_ci_upper


def test_multiple_testing_adjustments():
    p_values = [0.001, 0.02, 0.04, 0.15, 0.80]
    # Bonferroni
    bonf = adjust_multiple_testing(p_values, method="bonferroni", alpha=0.05)
    assert bonf["adjusted_p_values"][0] == 0.005
    assert bonf["adjusted_p_values"][1] == 0.10  # 0.02 * 5
    assert bonf["rejected"] == [True, False, False, False, False]

    # Benjamini-Hochberg FDR
    bh = adjust_multiple_testing(p_values, method="fdr_bh", alpha=0.05)
    # BH is more powerful than Bonferroni
    assert sum(bh["rejected"]) >= sum(bonf["rejected"])
    # Adjusted p-values must be non-decreasing
    for i in range(len(bh["adjusted_p_values"]) - 1):
        assert bh["adjusted_p_values"][i] <= bh["adjusted_p_values"][i+1]


def test_power_and_mde_calculations():
    # Power under large sample should be high
    power_high = calculate_two_proportion_power(p_control=0.01, relative_lift=0.50, n_treatment=20000, n_control=20000)
    assert power_high > 0.95

    # Power under zero sample or zero lift should be alpha (0.05)
    power_zero = calculate_two_proportion_power(p_control=0.01, relative_lift=0.0, n_treatment=5000, n_control=5000)
    assert power_zero == 0.05

    # MDE solver
    mde = calculate_mde(p_control=0.01, n_treatment=50000, n_control=50000, target_power=0.80)
    assert 0.05 < mde["relative_mde"] < 0.30
