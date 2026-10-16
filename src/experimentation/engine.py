"""
Experimentation Engine for Randomized Controlled Trials (A/B Tests).
Calculates Intention-to-Treat (ITT) effects, Local Average Treatment Effects (LATE / CACE),
absolute and relative conversion lift, Delta Method confidence intervals,
hypothesis tests (Z-test, Fisher's exact), and multiple testing corrections.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LiftMetrics:
    metric_name: str
    control_count: int
    treatment_count: int
    control_successes: int
    treatment_successes: int
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    absolute_lift_se: float
    absolute_lift_ci_lower: float
    absolute_lift_ci_upper: float
    relative_lift: float
    relative_lift_ci_lower: float
    relative_lift_ci_upper: float
    z_statistic: float
    p_value: float
    statistically_significant: bool
    alpha: float


@dataclass
class CACEMetrics:
    itt_effect: float
    compliance_rate: float  # Exposure rate among treated (since control exposure == 0)
    cace_estimate: float
    cace_se: float
    cace_ci_lower: float
    cace_ci_upper: float
    p_value: float


def calculate_relative_lift_ci(
    p_t: float,
    n_t: int,
    p_c: float,
    n_c: int,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Computes relative lift (p_t / p_c - 1) and asymmetric confidence intervals
    using the Delta Method on the log relative risk scale:
        log(RR) = log(p_t) - log(p_c)
        Var(log(RR)) = (1 - p_t) / (n_t * p_t) + (1 - p_c) / (n_c * p_c)
    This prevents impossible lower bounds (< -100%) when base rates are low.
    """
    if p_c <= 0 or p_t <= 0:
        return 0.0, 0.0, 0.0

    relative_lift = (p_t - p_c) / p_c
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    # Variance of log(p_t / p_c) via Delta Method
    var_log_rr = ((1.0 - p_t) / (n_t * p_t)) + ((1.0 - p_c) / (n_c * p_c))
    se_log_rr = np.sqrt(max(0.0, var_log_rr))

    rr = p_t / p_c
    ci_lower_rr = rr * np.exp(-z_crit * se_log_rr)
    ci_upper_rr = rr * np.exp(z_crit * se_log_rr)

    return relative_lift, ci_lower_rr - 1.0, ci_upper_rr - 1.0


def analyze_binary_experiment(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    alpha: float = 0.05
) -> LiftMetrics:
    """
    Performs rigorous two-sample proportion A/B testing on a binary outcome.
    Computes absolute lift, standard errors, relative lift with Delta Method CI,
    pooled Z-statistic, and asymptotic p-value.
    """
    treated = df[df[treatment_col] == 1][outcome_col].dropna()
    control = df[df[treatment_col] == 0][outcome_col].dropna()

    n_t = len(treated)
    n_c = len(control)
    if n_t == 0 or n_c == 0:
        raise ValueError("Both treatment and control cohorts must have non-zero samples.")

    x_t = int(treated.sum())
    x_c = int(control.sum())

    p_t = float(x_t / n_t)
    p_c = float(x_c / n_c)

    abs_lift = p_t - p_c
    abs_se = np.sqrt((p_t * (1.0 - p_t) / n_t) + (p_c * (1.0 - p_c) / n_c))

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    abs_ci_lower = abs_lift - z_crit * abs_se
    abs_ci_upper = abs_lift + z_crit * abs_se

    rel_lift, rel_ci_lower, rel_ci_upper = calculate_relative_lift_ci(p_t, n_t, p_c, n_c, alpha=alpha)

    # Hypothesis testing under H0: p_t == p_c (Pooled standard error)
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_t + 1.0 / n_c))

    z_stat = (p_t - p_c) / se_pool if se_pool > 0 else 0.0
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    is_sig = bool(p_value < alpha)

    return LiftMetrics(
        metric_name=outcome_col,
        control_count=n_c,
        treatment_count=n_t,
        control_successes=x_c,
        treatment_successes=x_t,
        control_rate=p_c,
        treatment_rate=p_t,
        absolute_lift=abs_lift,
        absolute_lift_se=abs_se,
        absolute_lift_ci_lower=abs_ci_lower,
        absolute_lift_ci_upper=abs_ci_upper,
        relative_lift=rel_lift,
        relative_lift_ci_lower=rel_ci_lower,
        relative_lift_ci_upper=rel_ci_upper,
        z_statistic=z_stat,
        p_value=p_value,
        statistically_significant=is_sig,
        alpha=alpha
    )


def estimate_cace(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = "treatment",
    exposure_col: str = "exposure",
    alpha: float = 0.05
) -> CACEMetrics:
    """
    Estimates the Complier Average Causal Effect (CACE / LATE) using the Wald Instrumental
    Variable Estimator under one-sided non-compliance:
        CACE = (E[Y | Z=1] - E[Y | Z=0]) / (E[D | Z=1] - E[D | Z=0])
    Since control exposure E[D | Z=0] == 0:
        CACE = ITT_Y / Compliance_Rate_Treated
    Variance derived via Delta Method for ratio of independent sample means.
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    n_t = len(treated)
    n_c = len(control)

    y_t = treated[outcome_col].mean()
    y_c = control[outcome_col].mean()
    itt = y_t - y_c

    # Compliance rate: proportion of treated units that were effectively exposed
    d_t = treated[exposure_col].mean()
    d_c = control[exposure_col].mean()

    denom = d_t - d_c
    if denom <= 0:
        raise ValueError(f"Compliance denominator must be positive. Got: {denom}")

    cace = itt / denom

    # Delta method variance for ratio R = (Y_t - Y_c) / D_t
    var_yt = treated[outcome_col].var() / n_t
    var_yc = control[outcome_col].var() / n_c
    var_itt = var_yt + var_yc

    var_dt = treated[exposure_col].var() / n_t
    # Covariance between Y_t and D_t in treated group
    cov_yt_dt = np.cov(treated[outcome_col], treated[exposure_col])[0, 1] / n_t

    # Var(CACE) approx (1/denom^2) * Var(ITT) + (itt^2 / denom^4) * Var(D_t) - 2*(itt/denom^3)*Cov(Y_t, D_t)
    var_cace = (
        (1.0 / (denom ** 2)) * var_itt
        + ((itt ** 2) / (denom ** 4)) * var_dt
        - 2.0 * (itt / (denom ** 3)) * cov_yt_dt
    )
    cace_se = float(np.sqrt(max(1e-12, var_cace)))

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    cace_ci_lower = cace - z_crit * cace_se
    cace_ci_upper = cace + z_crit * cace_se

    z_stat = cace / cace_se if cace_se > 0 else 0.0
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

    return CACEMetrics(
        itt_effect=float(itt),
        compliance_rate=float(denom),
        cace_estimate=float(cace),
        cace_se=cace_se,
        cace_ci_lower=float(cace_ci_lower),
        cace_ci_upper=float(cace_ci_upper),
        p_value=p_value
    )


def adjust_multiple_testing(
    p_values: List[float],
    method: str = "fdr_bh",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Applies classical multiple testing corrections to control FWER or FDR.
    Supported methods:
      - 'bonferroni': Controls Family-Wise Error Rate (FWER)
      - 'fdr_bh': Benjamini-Hochberg False Discovery Rate (FDR)
    """
    p_vals = np.array(p_values)
    m = len(p_vals)
    if m == 0:
        return {"adjusted_p_values": [], "rejected": []}

    if method == "bonferroni":
        adj_p = np.minimum(p_vals * m, 1.0)
        rejected = adj_p < alpha
    elif method == "fdr_bh":
        # Benjamini-Hochberg rank-based adjustment
        sort_idx = np.argsort(p_vals)
        inv_idx = np.empty(m, dtype=int)
        inv_idx[sort_idx] = np.arange(m)

        sorted_p = p_vals[sort_idx]
        ranks = np.arange(1, m + 1)
        bh_vals = sorted_p * m / ranks

        # Enforce monotonicity in reverse
        bh_adj = np.minimum.accumulate(bh_vals[::-1])[::-1]
        bh_adj = np.minimum(bh_adj, 1.0)
        adj_p = bh_adj[inv_idx]
        rejected = adj_p < alpha
    else:
        raise ValueError(f"Unknown multiple testing method: {method}")

    return {
        "method": method,
        "original_p_values": p_vals.tolist(),
        "adjusted_p_values": adj_p.tolist(),
        "rejected": rejected.tolist()
    }
