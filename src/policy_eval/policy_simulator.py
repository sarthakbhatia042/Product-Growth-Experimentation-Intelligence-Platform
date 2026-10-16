"""
Product Decision & Economic Simulator Engine.
Simulates marketing targeting policies under explicit budget, contact cost,
and customer conversion value assumptions.
Compares Model-Guided Uplift against Treat All, Treat None, and Random Targeting.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class PolicyScenario:
    policy_name: str
    target_fraction: float
    units_targeted: int
    incremental_conversions: float
    total_cost: float
    gross_revenue: float
    net_profit: float
    net_profit_ci_lower: float
    net_profit_ci_upper: float
    roi_pct: float
    roi_ci_lower: float
    roi_ci_upper: float
    is_budget_feasible: bool
    assumed_conversion_value: float
    assumed_contact_cost: float
    budget_limit: float
    label: str = "SIMULATED_SCENARIO (NOT OBSERVED REVENUE)"


def simulate_targeting_policies(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    conversion_value: float = 50.0,
    contact_cost: float = 0.10,
    budget: float = 15000.0,
    step_pct: float = 0.05
) -> Dict[str, Any]:
    """
    Simulates marketing targeting policies across population cutoff thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Observed outcomes.
    treatment : np.ndarray
        Observed treatment indicators.
    uplift_scores : np.ndarray
        Predicted uplift scores.
    conversion_value : float
        Simulated gross profit per incremental conversion ($).
    contact_cost : float
        Simulated cost per targeted impression/contact ($).
    budget : float
        Simulated marketing budget cap ($).
    step_pct : float
        Population fraction increment for simulation curve.

    Returns
    -------
    dict
        Contains policy curves, optimal threshold summary, and comparisons
        against 'Treat All', 'Treat None', and 'Random'.
    """
    n = len(y_true)
    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    w_sorted = treatment[order]

    cum_w = np.cumsum(w_sorted)
    cum_c = np.cumsum(1 - w_sorted)
    cum_y_w = np.cumsum(y_sorted * w_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - w_sorted))

    # Overall baseline metrics
    total_w = cum_w[-1]
    total_c = cum_c[-1]
    total_y_w = cum_y_w[-1]
    total_y_c = cum_y_c[-1]

    overall_lift_rate = (total_y_w / total_w) - (total_y_c / total_c)
    break_even_lift = contact_cost / conversion_value if conversion_value > 0 else 0.0

    fractions = np.arange(step_pct, 1.0 + step_pct / 2.0, step_pct)
    curve_records = []

    best_profit = -float("inf")
    best_policy_idx = -1

    for frac in fractions:
        k = int(np.round(frac * n))
        k = min(k, n)
        nw = cum_w[k - 1]
        nc = cum_c[k - 1]
        yw = cum_y_w[k - 1]
        yc = cum_y_c[k - 1]

        p_w = yw / nw if nw > 0 else 0.0
        p_c = yc / nc if nc > 0 else 0.0
        local_lift_rate = p_w - p_c

        # Total expected incremental conversions if k units in population are targeted
        inc_conversions_model = local_lift_rate * k
        cost_model = k * contact_cost
        rev_model = inc_conversions_model * conversion_value
        profit_model = rev_model - cost_model
        roi_model = (profit_model / cost_model * 100.0) if cost_model > 0 else 0.0
        feasible = cost_model <= budget

        # Random targeting benchmark targeting same fraction k
        inc_conversions_rand = overall_lift_rate * k
        cost_rand = cost_model
        rev_rand = inc_conversions_rand * conversion_value
        profit_rand = rev_rand - cost_rand
        roi_rand = (profit_rand / cost_rand * 100.0) if cost_rand > 0 else 0.0

        score_threshold = float(uplift_scores[order[k - 1]])

        curve_records.append({
            "fraction_targeted": float(frac),
            "units_targeted": int(k),
            "score_threshold": score_threshold,
            "incremental_conversions_model": float(inc_conversions_model),
            "cost_model": float(cost_model),
            "profit_model": float(profit_model),
            "roi_model_pct": float(roi_model),
            "incremental_conversions_random": float(inc_conversions_rand),
            "cost_random": float(cost_rand),
            "profit_random": float(profit_rand),
            "roi_random_pct": float(roi_rand),
            "is_budget_feasible": bool(feasible)
        })

        if feasible and profit_model > best_profit:
            best_profit = profit_model
            best_policy_idx = len(curve_records) - 1

    curve_df = pd.DataFrame(curve_records)

    # -------------------------------------------------------------
    # Bootstrap Uncertainty Estimation for Policies on Held-Out Data
    # -------------------------------------------------------------
    opt_row = curve_df.iloc[best_policy_idx] if best_policy_idx >= 0 else curve_df.iloc[0]
    opt_k = int(opt_row["units_targeted"])
    opt_frac = float(opt_row["fraction_targeted"])

    rng = np.random.default_rng(42)
    n_boot = 100
    model_boot_profits = []
    treat_all_boot_profits = []
    rand_boot_profits = []

    for _ in range(n_boot):
        b_idx = rng.integers(0, n, size=n)
        b_y = y_true[b_idx]
        b_w = treatment[b_idx]
        b_s = uplift_scores[b_idx]

        b_order = np.argsort(-b_s)
        # Top-fraction model evaluation
        w_sub = b_w[b_order[:opt_k]]
        y_sub = b_y[b_order[:opt_k]]
        nw_sub = np.sum(w_sub == 1)
        nc_sub = np.sum(w_sub == 0)
        p_w_sub = np.sum(y_sub[w_sub == 1]) / nw_sub if nw_sub > 0 else 0.0
        p_c_sub = np.sum(y_sub[w_sub == 0]) / nc_sub if nc_sub > 0 else 0.0
        b_lift_model = p_w_sub - p_c_sub
        b_prof_model = (b_lift_model * opt_k) * conversion_value - (opt_k * contact_cost)
        model_boot_profits.append(b_prof_model)

        # Treat all evaluation
        nw_all = np.sum(b_w == 1)
        nc_all = np.sum(b_w == 0)
        p_w_all = np.sum(b_y[b_w == 1]) / nw_all if nw_all > 0 else 0.0
        p_c_all = np.sum(b_y[b_w == 0]) / nc_all if nc_all > 0 else 0.0
        b_lift_all = p_w_all - p_c_all
        b_prof_all = (b_lift_all * n) * conversion_value - (n * contact_cost)
        treat_all_boot_profits.append(b_prof_all)

        # Random evaluation under budget
        max_u = min(n, int(budget / contact_cost))
        b_prof_rand = (b_lift_all * max_u) * conversion_value - (max_u * contact_cost)
        rand_boot_profits.append(b_prof_rand)

    # Specific benchmark policies:
    # 1. Model-Guided Policy (Top Decile Grid Search Maximizer)
    opt_cost = float(opt_row["cost_model"])
    opt_prof = float(opt_row["profit_model"])
    opt_prof_ci_l = float(np.percentile(model_boot_profits, 2.5))
    opt_prof_ci_u = float(np.percentile(model_boot_profits, 97.5))
    opt_roi_ci_l = (opt_prof_ci_l / opt_cost * 100.0) if opt_cost > 0 else 0.0
    opt_roi_ci_u = (opt_prof_ci_u / opt_cost * 100.0) if opt_cost > 0 else 0.0

    opt_policy = PolicyScenario(
        policy_name=f"Top-Decile Targeting Policy (Grid Search Maximizer, {opt_frac*100:.0f}%)",
        target_fraction=opt_frac,
        units_targeted=opt_k,
        incremental_conversions=float(opt_row["incremental_conversions_model"]),
        total_cost=opt_cost,
        gross_revenue=float(opt_row["incremental_conversions_model"] * conversion_value),
        net_profit=opt_prof,
        net_profit_ci_lower=opt_prof_ci_l,
        net_profit_ci_upper=opt_prof_ci_u,
        roi_pct=float(opt_row["roi_model_pct"]),
        roi_ci_lower=opt_roi_ci_l,
        roi_ci_upper=opt_roi_ci_u,
        is_budget_feasible=True,
        assumed_conversion_value=conversion_value,
        assumed_contact_cost=contact_cost,
        budget_limit=budget
    )

    # 2. Treat All Policy (100% targeted)
    treat_all_row = curve_df.iloc[-1]
    all_cost = float(n * contact_cost)
    all_prof = float(treat_all_row["profit_model"])
    all_prof_ci_l = float(np.percentile(treat_all_boot_profits, 2.5))
    all_prof_ci_u = float(np.percentile(treat_all_boot_profits, 97.5))
    all_roi_ci_l = (all_prof_ci_l / all_cost * 100.0) if all_cost > 0 else 0.0
    all_roi_ci_u = (all_prof_ci_u / all_cost * 100.0) if all_cost > 0 else 0.0

    treat_all_policy = PolicyScenario(
        policy_name="Treat All (Indiscriminate Targeting, 100%)",
        target_fraction=1.0,
        units_targeted=n,
        incremental_conversions=float(treat_all_row["incremental_conversions_model"]),
        total_cost=all_cost,
        gross_revenue=float(treat_all_row["incremental_conversions_model"] * conversion_value),
        net_profit=all_prof,
        net_profit_ci_lower=all_prof_ci_l,
        net_profit_ci_upper=all_prof_ci_u,
        roi_pct=float(treat_all_row["roi_model_pct"]),
        roi_ci_lower=all_roi_ci_l,
        roi_ci_upper=all_roi_ci_u,
        is_budget_feasible=bool(all_cost <= budget),
        assumed_conversion_value=conversion_value,
        assumed_contact_cost=contact_cost,
        budget_limit=budget
    )

    # 3. Treat None Policy (0% targeted)
    treat_none_policy = PolicyScenario(
        policy_name="Treat None (Status Quo, 0%)",
        target_fraction=0.0,
        units_targeted=0,
        incremental_conversions=0.0,
        total_cost=0.0,
        gross_revenue=0.0,
        net_profit=0.0,
        net_profit_ci_lower=0.0,
        net_profit_ci_upper=0.0,
        roi_pct=0.0,
        roi_ci_lower=0.0,
        roi_ci_upper=0.0,
        is_budget_feasible=True,
        assumed_conversion_value=conversion_value,
        assumed_contact_cost=contact_cost,
        budget_limit=budget
    )

    # 4. Budget-Constrained Random Targeting
    max_units_under_budget = min(n, int(budget / contact_cost))
    random_frac = max_units_under_budget / n
    rand_inc_conv = overall_lift_rate * max_units_under_budget
    rand_cost = max_units_under_budget * contact_cost
    rand_rev = rand_inc_conv * conversion_value
    rand_profit = rand_rev - rand_cost
    rand_roi = (rand_profit / rand_cost * 100.0) if rand_cost > 0 else 0.0
    rand_prof_ci_l = float(np.percentile(rand_boot_profits, 2.5))
    rand_prof_ci_u = float(np.percentile(rand_boot_profits, 97.5))
    rand_roi_ci_l = (rand_prof_ci_l / rand_cost * 100.0) if rand_cost > 0 else 0.0
    rand_roi_ci_u = (rand_prof_ci_u / rand_cost * 100.0) if rand_cost > 0 else 0.0

    random_budget_policy = PolicyScenario(
        policy_name=f"Random Targeting (Under Budget, {random_frac*100:.0f}%)",
        target_fraction=float(random_frac),
        units_targeted=max_units_under_budget,
        incremental_conversions=float(rand_inc_conv),
        total_cost=float(rand_cost),
        gross_revenue=float(rand_rev),
        net_profit=float(rand_profit),
        net_profit_ci_lower=rand_prof_ci_l,
        net_profit_ci_upper=rand_prof_ci_u,
        roi_pct=float(rand_roi),
        roi_ci_lower=rand_roi_ci_l,
        roi_ci_upper=rand_roi_ci_u,
        is_budget_feasible=True,
        assumed_conversion_value=conversion_value,
        assumed_contact_cost=contact_cost,
        budget_limit=budget
    )

    return {
        "curve_df": curve_df,
        "optimal_policy": opt_policy,
        "treat_all_policy": treat_all_policy,
        "treat_none_policy": treat_none_policy,
        "random_budget_policy": random_budget_policy,
        "break_even_lift": float(break_even_lift),
        "break_even_lift_pct": float(break_even_lift * 100.0),
        "comparison_table": pd.DataFrame([
            asdict(opt_policy),
            asdict(random_budget_policy),
            asdict(treat_all_policy),
            asdict(treat_none_policy)
        ])
    }
