"""
Sequential Testing & Repeated Peeking Risk Analysis.
Demonstrates Type I error inflation under continuous monitoring and implements
Group Sequential / Alpha-Spending boundaries (O'Brien-Fleming and Pocock).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def simulate_peeking_risk(
    sample_size: int = 10000,
    base_rate: float = 0.04,
    num_peeks: int = 20,
    num_simulations: int = 500,
    alpha: float = 0.05,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Simulates Type I error rate inflation under naive continuous monitoring (A/A testing with repeated peeking).
    Under H0 (treatment effect is strictly zero), tests how often p-value drops below alpha
    at ANY interim checkpoint.

    Returns
    -------
    dict
        Dictionary containing empirical false positive rates, simulation trajectories,
        and comparison with fixed-horizon evaluation.
    """
    rng = np.random.default_rng(random_seed)
    peek_checkpoints = np.linspace(sample_size // num_peeks, sample_size, num_peeks, dtype=int)

    false_positives_peeking = 0
    false_positives_fixed = 0

    sample_trajectories = []

    for sim in range(num_simulations):
        # Generate control and treatment from identical Bernoulli distributions (H0 true)
        control = rng.binomial(1, base_rate, size=sample_size)
        treatment = rng.binomial(1, base_rate, size=sample_size)

        p_values_trajectory = []
        ever_rejected = False

        for n in peek_checkpoints:
            x_c = control[:n].sum()
            x_t = treatment[:n].sum()

            p_pool = (x_c + x_t) / (2.0 * n)
            se_pool = np.sqrt(2.0 * p_pool * (1.0 - p_pool) / n) if p_pool > 0 else 1.0

            z_stat = (x_t / n - x_c / n) / se_pool if se_pool > 0 else 0.0
            p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
            p_values_trajectory.append(float(p_val))

            if p_val < alpha:
                ever_rejected = True

        if ever_rejected:
            false_positives_peeking += 1

        # Fixed horizon check (only at the very end)
        if p_values_trajectory[-1] < alpha:
            false_positives_fixed += 1

        if sim < 10:
            sample_trajectories.append({
                "simulation_id": sim,
                "checkpoints": peek_checkpoints.tolist(),
                "p_values": p_values_trajectory,
                "ever_rejected": ever_rejected
            })

    empirical_peeking_fpr = false_positives_peeking / num_simulations
    empirical_fixed_fpr = false_positives_fixed / num_simulations

    return {
        "num_simulations": num_simulations,
        "sample_size": sample_size,
        "num_peeks": num_peeks,
        "nominal_alpha": alpha,
        "empirical_fixed_horizon_fpr": float(empirical_fixed_fpr),
        "empirical_peeking_fpr": float(empirical_peeking_fpr),
        "error_inflation_factor": float(empirical_peeking_fpr / alpha) if alpha > 0 else 0.0,
        "sample_trajectories": sample_trajectories,
        "peek_checkpoints": peek_checkpoints.tolist()
    }


def calculate_alpha_spending_boundaries(
    num_looks: int = 5,
    total_alpha: float = 0.05,
    method: str = "obrien_fleming"
) -> pd.DataFrame:
    """
    Computes group sequential boundaries using Lan-DeMets alpha spending functions.
    Supported:
      - 'obrien_fleming': Very conservative early on (hard to stop early), preserves alpha for final analysis.
      - 'pocock': Constant spending across looks.
    """
    info_fractions = np.linspace(1.0 / num_looks, 1.0, num_looks)
    z_alpha_half = stats.norm.ppf(1.0 - total_alpha / 2.0)

    rows = []
    prev_alpha = 0.0

    for i, t in enumerate(info_fractions):
        if method == "obrien_fleming":
            # Lan-DeMets analog of O'Brien-Fleming
            cum_alpha = 2.0 * (1.0 - stats.norm.cdf(z_alpha_half / np.sqrt(t)))
        elif method == "pocock":
            # Lan-DeMets analog of Pocock
            cum_alpha = total_alpha * np.log(1.0 + (np.e - 1.0) * t)
        else:
            raise ValueError(f"Unknown alpha spending method: {method}")

        cum_alpha = min(cum_alpha, total_alpha)
        inc_alpha = max(0.0, cum_alpha - prev_alpha)
        prev_alpha = cum_alpha

        # Approximate critical Z boundary for this look
        crit_z = stats.norm.ppf(1.0 - inc_alpha / 2.0) if inc_alpha > 0 else np.nan
        nom_p = 2.0 * (1.0 - stats.norm.cdf(crit_z)) if not np.isnan(crit_z) else 0.0

        rows.append({
            "look_index": i + 1,
            "information_fraction": float(t),
            "cumulative_alpha_spent": float(cum_alpha),
            "incremental_alpha": float(inc_alpha),
            "critical_z_boundary": float(crit_z),
            "nominal_p_threshold": float(nom_p)
        })

    return pd.DataFrame(rows)
