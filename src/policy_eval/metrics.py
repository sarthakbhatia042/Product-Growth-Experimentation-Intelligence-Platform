"""
Uplift Metrics and Policy Evaluation Engine.
Implements Area Under Uplift Curve (AUUC), Qini Curve, Qini Score,
Cumulative Gains, Decile Analysis, and Bootstrap Uncertainty Estimation.
Never treats individual predicted effects as directly observable ground truth.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


def compute_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 100
) -> pd.DataFrame:
    """
    Computes Cumulative Uplift and Qini curves by ranking individuals
    in descending order of predicted uplift scores.

    Parameters
    ----------
    y_true : np.ndarray
        Observed binary outcomes (e.g. conversion).
    treatment : np.ndarray
        Observed binary treatment indicators (1 or 0).
    uplift_scores : np.ndarray
        Model predicted uplift scores (higher = larger expected treatment effect).
    n_bins : int
        Number of evaluation points along the cumulative population curve.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - fraction_targeted: population fraction [0, 1]
        - cumulative_treatment_units
        - cumulative_control_units
        - cumulative_treatment_conversions
        - cumulative_control_conversions
        - cumulative_uplift: (p_t - p_c) * (n_t + n_c)
        - qini_value: Y_t - Y_c * (N_t / N_c)
        - random_qini_baseline: linear baseline connecting 0 to end
        - random_uplift_baseline: linear baseline connecting 0 to end
    """
    n = len(y_true)
    # Check for constant scores (e.g. Baseline ATE). If all scores are identical,
    # break ties uniformly at random to prevent arbitrary index-order drift from biasing AUUC/Qini.
    if np.all(uplift_scores == uplift_scores[0]) or np.std(uplift_scores) < 1e-12:
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-1e-10, 1e-10, size=n)
        order = np.argsort(-(uplift_scores + jitter))
    else:
        # Standard descending sort by predicted uplift
        order = np.argsort(-uplift_scores)

    y_sorted = y_true[order]
    w_sorted = treatment[order]

    # Pre-calculate cumulative counts
    cum_w = np.cumsum(w_sorted)
    cum_c = np.cumsum(1 - w_sorted)
    cum_y_w = np.cumsum(y_sorted * w_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - w_sorted))

    # Total constants
    total_w = cum_w[-1]
    total_c = cum_c[-1]
    total_y_w = cum_y_w[-1]
    total_y_c = cum_y_c[-1]

    bin_indices = np.linspace(n // n_bins, n - 1, n_bins, dtype=int)

    rows = [{
        "fraction_targeted": 0.0,
        "n_targeted": 0,
        "cum_treatment_units": 0,
        "cum_control_units": 0,
        "cum_treatment_conv": 0,
        "cum_control_conv": 0,
        "cumulative_uplift": 0.0,
        "qini_value": 0.0,
        "random_uplift_baseline": 0.0,
        "random_qini_baseline": 0.0
    }]

    final_qini = total_y_w - (total_y_c * (total_w / total_c) if total_c > 0 else 0.0)
    final_uplift = (
        (total_y_w / total_w - total_y_c / total_c) * n
        if total_w > 0 and total_c > 0 else 0.0
    )

    for idx in bin_indices:
        frac = (idx + 1) / n
        nw = cum_w[idx]
        nc = cum_c[idx]
        yw = cum_y_w[idx]
        yc = cum_y_c[idx]

        # Cumulative uplift
        p_w = yw / nw if nw > 0 else 0.0
        p_c = yc / nc if nc > 0 else 0.0
        cum_uplift = (p_w - p_c) * (nw + nc)

        # Radcliffe Qini formula
        qini = yw - (yc * (nw / nc) if nc > 0 else 0.0)

        rows.append({
            "fraction_targeted": float(frac),
            "n_targeted": int(idx + 1),
            "cum_treatment_units": int(nw),
            "cum_control_units": int(nc),
            "cum_treatment_conv": int(yw),
            "cum_control_conv": int(yc),
            "cumulative_uplift": float(cum_uplift),
            "qini_value": float(qini),
            "random_uplift_baseline": float(frac * final_uplift),
            "random_qini_baseline": float(frac * final_qini)
        })

    return pd.DataFrame(rows)


def calculate_auuc_and_qini(
    curve_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculates Area Under Uplift Curve (AUUC) and normalized Qini score
    relative to the random assignment baseline using trapezoidal integration.
    """
    x = curve_df["fraction_targeted"].values
    uplift = curve_df["cumulative_uplift"].values
    qini = curve_df["qini_value"].values
    rand_uplift = curve_df["random_uplift_baseline"].values
    rand_qini = curve_df["random_qini_baseline"].values

    # Trapezoidal integration
    area_uplift = np.trapezoid(uplift, x)
    area_rand_uplift = np.trapezoid(rand_uplift, x)
    auuc = float(area_uplift - area_rand_uplift)

    area_qini = np.trapezoid(qini, x)
    area_rand_qini = np.trapezoid(rand_qini, x)
    qini_score = float(area_qini - area_rand_qini)

    # Theoretical maximum Qini area (if all treatment successes occur before any control successes)
    # Total treatment conversions: curve_df["cum_treatment_conv"].iloc[-1]
    total_y_w = float(curve_df["cum_treatment_conv"].iloc[-1])
    # Approximate theoretical maximum Qini area is 0.5 * total_y_w
    max_qini_area = 0.5 * total_y_w if total_y_w > 0 else 1.0
    normalized_qini = float(qini_score / max_qini_area) if max_qini_area > 0 else 0.0

    return {
        "auuc": auuc,
        "raw_uplift_area": float(area_uplift),
        "qini_score": qini_score,
        "raw_qini_area": float(area_qini),
        "normalized_qini_score": normalized_qini,
        "units": "Cumulative Incremental Conversions (Qini) / Population-Weighted Uplift (AUUC)"
    }


def compute_decile_lift_table(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Computes observed treatment vs control conversion rates across deciles
    ranked from highest predicted uplift to lowest predicted uplift.
    """
    df = pd.DataFrame({
        "y": y_true,
        "w": treatment,
        "score": uplift_scores
    })

    # Decile 1 = highest predicted uplift, Decile 10 = lowest
    df["decile"] = pd.qcut(-df["score"], q=n_deciles, labels=list(range(1, n_deciles + 1)))

    rows = []
    for d, group in df.groupby("decile", observed=False):
        n_tot = len(group)
        w_grp = group[group["w"] == 1]
        c_grp = group[group["w"] == 0]

        n_w = len(w_grp)
        n_c = len(c_grp)
        y_w = w_grp["y"].sum()
        y_c = c_grp["y"].sum()

        p_w = y_w / n_w if n_w > 0 else 0.0
        p_c = y_c / n_c if n_c > 0 else 0.0
        abs_lift = p_w - p_c

        rows.append({
            "decile": int(d),
            "total_users": int(n_tot),
            "treatment_users": int(n_w),
            "control_users": int(n_c),
            "treatment_conversions": int(y_w),
            "control_conversions": int(y_c),
            "treatment_rate_pct": float(p_w * 100.0),
            "control_rate_pct": float(p_c * 100.0),
            "observed_lift_pct": float(abs_lift * 100.0),
            "mean_predicted_uplift": float(group["score"].mean())
        })

    return pd.DataFrame(rows)


def bootstrap_uplift_uncertainty(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bootstraps: int = 50,
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Computes bootstrap confidence intervals for AUUC and Qini scores on held-out evaluation set.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    auuc_samples = []
    qini_samples = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        b_y = y_true[idx]
        b_w = treatment[idx]
        b_s = uplift_scores[idx]

        b_curve = compute_uplift_curve(b_y, b_w, b_s, n_bins=50)
        res = calculate_auuc_and_qini(b_curve)
        auuc_samples.append(res["auuc"])
        qini_samples.append(res["qini_score"])

    auuc_arr = np.array(auuc_samples)
    qini_arr = np.array(qini_samples)

    lower_p = 100.0 * (alpha / 2.0)
    upper_p = 100.0 * (1.0 - alpha / 2.0)

    return {
        "auuc_mean": float(np.mean(auuc_arr)),
        "auuc_se": float(np.std(auuc_arr)),
        "auuc_ci_lower": float(np.percentile(auuc_arr, lower_p)),
        "auuc_ci_upper": float(np.percentile(auuc_arr, upper_p)),
        "qini_mean": float(np.mean(qini_arr)),
        "qini_se": float(np.std(qini_arr)),
        "qini_ci_lower": float(np.percentile(qini_arr, lower_p)),
        "qini_ci_upper": float(np.percentile(qini_arr, upper_p))
    }
