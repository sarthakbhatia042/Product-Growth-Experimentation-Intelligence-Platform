"""
Statistical Power and Minimum Detectable Effect (MDE) Analysis Engine.
Implements exact analytical power calculations, sample size determination,
and MDE curves for binary conversion and visit rates under asymmetric allocations.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats, optimize


def calculate_two_proportion_power(
    p_control: float,
    relative_lift: float,
    n_treatment: int,
    n_control: int,
    alpha: float = 0.05
) -> float:
    """
    Calculates analytical statistical power (1 - Beta) for a two-sample proportion test.

    Parameters
    ----------
    p_control : float
        Baseline proportion in control group.
    relative_lift : float
        Hypothesized relative change (e.g. 0.10 for +10% lift).
    n_treatment : int
        Sample size in treatment group.
    n_control : int
        Sample size in control group.
    alpha : float
        Type I error significance level (default: 0.05).

    Returns
    -------
    float
        Statistical power in [0, 1].
    """
    if n_treatment <= 0 or n_control <= 0 or p_control <= 0:
        return 0.0

    p_treatment = p_control * (1.0 + relative_lift)
    p_treatment = np.clip(p_treatment, 1e-6, 1.0 - 1e-6)

    delta = abs(p_treatment - p_control)
    if delta == 0:
        return alpha

    # Null hypothesis pooled variance
    p_pooled = (n_treatment * p_treatment + n_control * p_control) / (n_treatment + n_control)
    se_null = np.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n_treatment + 1.0 / n_control))

    # Alternative hypothesis unpooled variance
    se_alt = np.sqrt(
        (p_treatment * (1.0 - p_treatment) / n_treatment) +
        (p_control * (1.0 - p_control) / n_control)
    )

    if se_alt <= 0 or se_null <= 0:
        return 0.0

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    # Power calculation
    z_beta = (delta - z_crit * se_null) / se_alt
    power = float(stats.norm.cdf(z_beta))
    return float(np.clip(power, 0.0, 1.0))


def calculate_mde(
    p_control: float,
    n_treatment: int,
    n_control: int,
    target_power: float = 0.80,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Solves for the Minimum Detectable Effect (MDE) in both absolute and relative terms
    that achieves the desired target power given sample sizes.
    """
    if n_treatment <= 0 or n_control <= 0 or p_control <= 0:
        return {"absolute_mde": 0.0, "relative_mde": 0.0, "target_power": target_power}

    def loss(rel_lift: float) -> float:
        curr_power = calculate_two_proportion_power(
            p_control=p_control,
            relative_lift=rel_lift,
            n_treatment=n_treatment,
            n_control=n_control,
            alpha=alpha
        )
        return curr_power - target_power

    # Numerical root finding
    try:
        sol = optimize.root_scalar(loss, bracket=[1e-5, 5.0], method="brentq")
        mde_rel = float(sol.root)
    except Exception:
        # Fallback grid search
        grid = np.linspace(1e-4, 5.0, 500)
        powers = [calculate_two_proportion_power(p_control, r, n_treatment, n_control, alpha) for r in grid]
        idx = np.argmin(np.abs(np.array(powers) - target_power))
        mde_rel = float(grid[idx])

    mde_abs = float(p_control * mde_rel)

    return {
        "baseline_rate": float(p_control),
        "absolute_mde": mde_abs,
        "relative_mde": mde_rel,
        "relative_mde_pct": mde_rel * 100.0,
        "target_power": float(target_power),
        "alpha": float(alpha)
    }


def calculate_required_sample_size(
    p_control: float,
    relative_mde: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
    treatment_ratio: float = 0.846
) -> Dict[str, int]:
    """
    Computes required total and per-cohort sample size to detect a target relative MDE.
    """
    p_treatment = p_control * (1.0 + relative_mde)
    delta = abs(p_treatment - p_control)

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(target_power)

    p_bar = treatment_ratio * p_treatment + (1.0 - treatment_ratio) * p_control

    # Formula accounting for asymmetric allocation
    term1 = z_alpha * np.sqrt(p_bar * (1.0 - p_bar) * (1.0 / treatment_ratio + 1.0 / (1.0 - treatment_ratio)))
    term2 = z_beta * np.sqrt(
        (p_treatment * (1.0 - p_treatment) / treatment_ratio) +
        (p_control * (1.0 - p_control) / (1.0 - treatment_ratio))
    )

    n_total = int(np.ceil(((term1 + term2) ** 2) / (delta ** 2)))
    n_treatment = int(np.ceil(n_total * treatment_ratio))
    n_control = n_total - n_treatment

    return {
        "required_total_sample": n_total,
        "required_treatment_sample": n_treatment,
        "required_control_sample": n_control,
        "baseline_rate": float(p_control),
        "relative_mde": float(relative_mde),
        "target_power": float(target_power),
        "treatment_ratio": float(treatment_ratio)
    }


def generate_mde_curves(
    p_control: float,
    treatment_ratio: float = 0.846,
    sample_sizes: Optional[List[int]] = None,
    target_power: float = 0.80,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Generates MDE sensitivity curve data across a range of sample sizes.
    """
    if sample_sizes is None:
        sample_sizes = [5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]

    records = []
    for n in sample_sizes:
        n_t = int(n * treatment_ratio)
        n_c = n - n_t
        res = calculate_mde(p_control, n_t, n_c, target_power=target_power, alpha=alpha)
        records.append({
            "total_sample_size": n,
            "treatment_sample": n_t,
            "control_sample": n_c,
            "absolute_mde": res["absolute_mde"],
            "relative_mde_pct": res["relative_mde_pct"]
        })

    return pd.DataFrame(records)
