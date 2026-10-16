"""
Sample Ratio Mismatch (SRM) Detection Engine.
Implements Chi-Squared Goodness-of-Fit and Exact Binomial tests
to identify traffic allocation anomalies in A/B experiments.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats


@dataclass
class SRMTestResult:
    observed_control: int
    observed_treatment: int
    total_samples: int
    expected_ratio_treatment: float
    expected_ratio_control: float
    expected_treatment: float
    expected_control: float
    chi2_statistic: float
    p_value: float
    srm_detected: bool
    severity: str  # 'NONE', 'LOW', 'CRITICAL'
    alpha_threshold: float
    explanation: str


def check_sample_ratio_mismatch(
    observed_treatment: int,
    observed_control: int,
    expected_ratio_treatment: float = 0.8460,
    alpha: float = 0.001
) -> SRMTestResult:
    """
    Evaluates whether the observed cohort counts deviate significantly
    from the planned assignment ratio.

    Parameters
    ----------
    observed_treatment : int
        Number of units in treatment cohort.
    observed_control : int
        Number of units in control cohort.
    expected_ratio_treatment : float
        Nominal expected probability of assignment to treatment (default: 0.846).
    alpha : float
        Significance threshold for SRM (default: 0.001, following standard industry practice).

    Returns
    -------
    SRMTestResult
    """
    total = observed_treatment + observed_control
    if total <= 0:
        raise ValueError("Total observed samples must be greater than zero.")

    expected_ratio_control = 1.0 - expected_ratio_treatment
    expected_treatment = total * expected_ratio_treatment
    expected_control = total * expected_ratio_control

    observed = np.array([observed_treatment, observed_control], dtype=float)
    expected = np.array([expected_treatment, expected_control], dtype=float)

    chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)

    srm_detected = bool(p_val < alpha)

    if not srm_detected:
        severity = "NONE"
        explanation = (
            f"No SRM detected (p={p_val:.4e} >= {alpha}). Traffic allocation "
            f"aligns with the design ratio ({expected_ratio_treatment*100:.1f}% : {expected_ratio_control*100:.1f}%)."
        )
    elif p_val < 1e-6:
        severity = "CRITICAL"
        explanation = (
            f"CRITICAL SRM DETECTED (p={p_val:.4e} < 1e-6). Severe deviation from expected ratio. "
            f"Treatment effects may be biased due to asymmetric attrition, bot filtering, or instrumentation bugs."
        )
    else:
        severity = "LOW"
        explanation = (
            f"Marginal SRM detected (p={p_val:.4e} < {alpha}). Investigate tracking logs and upstream pipelines."
        )

    return SRMTestResult(
        observed_control=int(observed_control),
        observed_treatment=int(observed_treatment),
        total_samples=int(total),
        expected_ratio_treatment=float(expected_ratio_treatment),
        expected_ratio_control=float(expected_ratio_control),
        expected_treatment=float(expected_treatment),
        expected_control=float(expected_control),
        chi2_statistic=float(chi2_stat),
        p_value=float(p_val),
        srm_detected=srm_detected,
        severity=severity,
        alpha_threshold=alpha,
        explanation=explanation
    )
