"""
Automated Publication-Quality Figure & Visual Asset Generator.
Creates high-resolution charts for research reports, model cards, and documentation.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Modern, clean aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def plot_cohort_lift(
    conv_metrics,
    visit_metrics,
    output_path: str = "docs/figures/fig1_cohort_conversion_rates.png"
):
    """Figure 1: Observed conversion and visit rates with 95% confidence intervals."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Conversion Rate
    rates_conv = [conv_metrics.control_rate * 100, conv_metrics.treatment_rate * 100]
    ci_conv_c = [
        (conv_metrics.control_rate - 1.96 * np.sqrt(conv_metrics.control_rate * (1 - conv_metrics.control_rate) / conv_metrics.control_count)) * 100,
        (conv_metrics.control_rate + 1.96 * np.sqrt(conv_metrics.control_rate * (1 - conv_metrics.control_rate) / conv_metrics.control_count)) * 100
    ]
    ci_conv_t = [
        (conv_metrics.treatment_rate - 1.96 * np.sqrt(conv_metrics.treatment_rate * (1 - conv_metrics.treatment_rate) / conv_metrics.treatment_count)) * 100,
        (conv_metrics.treatment_rate + 1.96 * np.sqrt(conv_metrics.treatment_rate * (1 - conv_metrics.treatment_rate) / conv_metrics.treatment_count)) * 100
    ]
    err_conv = [
        [rates_conv[0] - ci_conv_c[0], rates_conv[1] - ci_conv_t[0]],
        [ci_conv_c[1] - rates_conv[0], ci_conv_t[1] - rates_conv[1]]
    ]

    bars1 = axes[0].bar(
        ["Control (W=0)", "Treatment (W=1)"],
        rates_conv,
        yerr=err_conv,
        capsize=6,
        color=["#64748b", "#2563eb"],
        edgecolor="#1e293b",
        linewidth=1.2,
        alpha=0.9
    )
    axes[0].set_ylabel("Conversion Rate (%)")
    axes[0].set_title(f"Primary Outcome: Conversion Rate\n(Rel Lift: +{conv_metrics.relative_lift*100:.1f}%, p={conv_metrics.p_value:.3e})")
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval / 2.0, f"{yval:.3f}%", ha="center", va="center", color="white", fontweight="bold")

    # Panel B: Visit Rate
    rates_vis = [visit_metrics.control_rate * 100, visit_metrics.treatment_rate * 100]
    ci_vis_c = [
        (visit_metrics.control_rate - 1.96 * np.sqrt(visit_metrics.control_rate * (1 - visit_metrics.control_rate) / visit_metrics.control_count)) * 100,
        (visit_metrics.control_rate + 1.96 * np.sqrt(visit_metrics.control_rate * (1 - visit_metrics.control_rate) / visit_metrics.control_count)) * 100
    ]
    ci_vis_t = [
        (visit_metrics.treatment_rate - 1.96 * np.sqrt(visit_metrics.treatment_rate * (1 - visit_metrics.treatment_rate) / visit_metrics.treatment_count)) * 100,
        (visit_metrics.treatment_rate + 1.96 * np.sqrt(visit_metrics.treatment_rate * (1 - visit_metrics.treatment_rate) / visit_metrics.treatment_count)) * 100
    ]
    err_vis = [
        [rates_vis[0] - ci_vis_c[0], rates_vis[1] - ci_vis_t[0]],
        [ci_vis_c[1] - rates_vis[0], ci_vis_t[1] - rates_vis[1]]
    ]

    bars2 = axes[1].bar(
        ["Control (W=0)", "Treatment (W=1)"],
        rates_vis,
        yerr=err_vis,
        capsize=6,
        color=["#64748b", "#059669"],
        edgecolor="#1e293b",
        linewidth=1.2,
        alpha=0.9
    )
    axes[1].set_ylabel("Visit Rate (%)")
    axes[1].set_title(f"Secondary Outcome: Visit Rate\n(Rel Lift: +{visit_metrics.relative_lift*100:.1f}%, p={visit_metrics.p_value:.3e})")
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval / 2.0, f"{yval:.2f}%", ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_uplift_curves(
    curves_dict: dict,
    output_path: str = "docs/figures/fig5_uplift_curves_comparison.png"
):
    """Figure 5: Qini and Cumulative Uplift curves for model comparisons."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    palette = {
        "X-Learner": "#2563eb",
        "T-Learner": "#059669",
        "S-Learner": "#d97706",
        "Baseline ATE": "#9333ea",
        "Random": "#64748b"
    }

    # Plot Qini Curves
    for name, df in curves_dict.items():
        color = palette.get(name, "#3b82f6")
        axes[0].plot(df["fraction_targeted"] * 100, df["qini_value"], label=name, color=color, linewidth=2.2)

    # Add random baseline from the first dataframe
    first_df = next(iter(curves_dict.values()))
    axes[0].plot(first_df["fraction_targeted"] * 100, first_df["random_qini_baseline"], "--", color="#64748b", label="Random Baseline", linewidth=1.8)
    axes[0].set_xlabel("Targeted Population Fraction (%)")
    axes[0].set_ylabel("Cumulative Incremental Conversions (Qini)")
    axes[0].set_title("Qini Curve Comparison")
    axes[0].legend(loc="lower right")

    # Plot Cumulative Uplift Curves
    for name, df in curves_dict.items():
        color = palette.get(name, "#3b82f6")
        axes[1].plot(df["fraction_targeted"] * 100, df["cumulative_uplift"], label=name, color=color, linewidth=2.2)

    axes[1].plot(first_df["fraction_targeted"] * 100, first_df["random_uplift_baseline"], "--", color="#64748b", label="Random Baseline", linewidth=1.8)
    axes[1].set_xlabel("Targeted Population Fraction (%)")
    axes[1].set_ylabel("Cumulative Uplift Metric")
    axes[1].set_title("Cumulative Uplift Curve Comparison")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_decile_lift(
    decile_df: pd.DataFrame,
    output_path: str = "docs/figures/fig6_decile_incremental_lift.png"
):
    """Figure 6: Decile-by-decile observed lift."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 5))

    colors = ["#2563eb" if x > 0 else "#dc2626" for x in decile_df["observed_lift_pct"]]
    bars = plt.bar(
        [f"Decile {d}" for d in decile_df["decile"]],
        decile_df["observed_lift_pct"],
        color=colors,
        edgecolor="#1e293b",
        linewidth=1.2,
        alpha=0.88
    )

    plt.axhline(0, color="#1e293b", linestyle="--", linewidth=1.0)
    plt.ylabel("Observed Incremental Lift (pp %)")
    plt.xlabel("Predicted Uplift Decile (1 = Highest Predicted Uplift)")
    plt.title("Heterogeneous Treatment Effect: Observed Lift by Decile")

    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        plt.text(bar.get_x() + bar.get_width()/2.0, h, f"{h:+.3f}%", ha="center", va=va, fontsize=9, fontweight="bold")

    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_mde_sensitivity(
    mde_df: pd.DataFrame,
    output_path: str = "docs/figures/fig3_mde_sensitivity_curves.png"
):
    """Figure 3: MDE curves across sample sizes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(mde_df["total_sample_size"], mde_df["relative_mde_pct"], "o-", color="#2563eb", linewidth=2.5, label="Relative MDE (%)")
    ax1.set_xlabel("Total Experiment Sample Size")
    ax1.set_ylabel("Relative MDE (%) at 80% Power", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.set_xscale("log")
    ax1.set_title("Minimum Detectable Effect (MDE) vs. Sample Size\n(Baseline Conversion = 0.22%, alpha = 0.05, power = 0.80)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sequential_peeking(
    peeking_results: dict,
    output_path: str = "docs/figures/fig4_sequential_peeking_risk.png"
):
    """Figure 4: Simulated continuous peeking trajectories vs fixed alpha."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 5.5))

    checkpoints = peeking_results["peek_checkpoints"]
    alpha = peeking_results["nominal_alpha"]

    for traj in peeking_results["sample_trajectories"]:
        color = "#dc2626" if traj["ever_rejected"] else "#94a3b8"
        alpha_val = 0.85 if traj["ever_rejected"] else 0.4
        plt.plot(checkpoints, traj["p_values"], color=color, alpha=alpha_val, linewidth=1.3)

    plt.axhline(alpha, color="#1e293b", linestyle="--", linewidth=2.0, label=f"Nominal alpha ({alpha})")
    plt.yscale("log")
    plt.xlabel("Sample Size (Interim Checks)")
    plt.ylabel("P-Value (Log Scale)")
    plt.title(f"A/A Test Simulation: False Positive Inflation from Repeated Peeking\n(Empirical Peeking FPR: {peeking_results['empirical_peeking_fpr']*100:.1f}% vs Nominal: {alpha*100:.1f}%)")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_economic_simulation(
    curve_df: pd.DataFrame,
    budget: float,
    output_path: str = "docs/figures/fig7_economic_policy_profit_curves.png"
):
    """Figure 7: Simulated net incremental profit comparison."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 5.5))

    x = curve_df["fraction_targeted"] * 100
    plt.plot(x, curve_df["profit_model"], color="#2563eb", linewidth=2.5, label="Model-Guided Targeting Net Profit ($)")
    plt.plot(x, curve_df["profit_random"], "--", color="#64748b", linewidth=2.0, label="Random Targeting Net Profit ($)")

    # Highlight budget cutoff
    feasible = curve_df[curve_df["is_budget_feasible"]]
    if len(feasible) > 0:
        max_feasible_frac = feasible["fraction_targeted"].max() * 100
        plt.axvline(max_feasible_frac, color="#dc2626", linestyle=":", linewidth=2.0, label=f"Budget Cap Cutoff ({max_feasible_frac:.0f}% Traffic)")

    plt.axhline(0, color="#1e293b", linestyle="-", linewidth=0.8)
    plt.xlabel("Percentage of User Population Targeted (%)")
    plt.ylabel("Net Incremental Profit ($)")
    plt.title(f"[SIMULATED SCENARIO] Incremental Economic Profit vs. Targeting Fraction\n(Assumed Conversion Value: $50, Contact Cost: $0.10, Budget: ${budget:,.0f})")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
