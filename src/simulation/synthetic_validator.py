"""
Statistical Verification Suite & Synthetic Ground Truth Simulator.
Generates synthetic benchmark datasets with known Individual Treatment Effects (CATE)
to rigorously evaluate:
  1. Estimator Bias & Precision in Estimating Heterogeneous Effects (PEHE)
  2. Confidence Interval Coverage
  3. False Positive Rate (Type I Error) under the Null Hypothesis (A/A test)
  4. Ranking Recovery (Spearman / Kendall rank correlation with true CATE)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error


def generate_synthetic_cate_dataset(
    n_samples: int = 10000,
    n_features: int = 6,
    treatment_ratio: float = 0.50,
    heterogeneity_type: str = "nonlinear",
    noise_level: float = 0.05,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generates a synthetic dataset with KNOWN ground truth baseline risk mu0(X)
    and KNOWN ground truth treatment effect tau_star(X).

    Parameters
    ----------
    n_samples : int
        Number of instances.
    n_features : int
        Number of pre-treatment covariates.
    treatment_ratio : float
        Probability of assignment to treatment.
    heterogeneity_type : str
        'linear', 'nonlinear', or 'null' (for A/A testing).
    noise_level : float
        Stochastic observation noise.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with features [x0..xn], treatment, observed y.
    true_tau : np.ndarray
        True underlying CATE for each unit.
    true_mu0 : np.ndarray
        True underlying baseline risk for each unit.
    """
    rng = np.random.default_rng(random_seed)

    # Generate continuous covariates
    X = rng.normal(0, 1, size=(n_samples, n_features))
    feature_cols = [f"x{i}" for i in range(n_features)]

    # Baseline response probability mu0(X) in [0.05, 0.40]
    base_logits = -1.5 + 0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]
    mu0 = 1.0 / (1.0 + np.exp(-base_logits))

    # True treatment effect tau_star(X)
    if heterogeneity_type == "null":
        # Strictly zero effect for A/A testing
        tau_star = np.zeros(n_samples, dtype=float)
    elif heterogeneity_type == "linear":
        tau_star = 0.05 + 0.04 * X[:, 0] - 0.03 * X[:, 1]
    elif heterogeneity_type == "nonlinear":
        # Mix of strong responders, neutral, and negative responders ("sleeping dogs")
        tau_star = (
            0.04
            + 0.06 * np.sin(np.pi * X[:, 0])
            + 0.04 * (X[:, 1] > 0.5).astype(float)
            - 0.03 * (X[:, 2] < -1.0).astype(float)
        )
    else:
        raise ValueError(f"Unknown heterogeneity_type: {heterogeneity_type}")

    # Clamp potential outcomes into valid probability domain [0.001, 0.999]
    mu1 = np.clip(mu0 + tau_star, 0.001, 0.999)
    # Effective true tau after boundary clamping
    tau_star_clamped = mu1 - mu0

    # Randomized assignment
    treatment = rng.binomial(1, treatment_ratio, size=n_samples)

    # Potential outcomes
    y0 = rng.binomial(1, mu0)
    y1 = rng.binomial(1, mu1)

    # Observed outcome
    y_observed = treatment * y1 + (1 - treatment) * y0

    df_dict = {feature_cols[i]: X[:, i] for i in range(n_features)}
    df_dict["treatment"] = treatment
    df_dict["y"] = y_observed
    df_dict["y0"] = y0
    df_dict["y1"] = y1
    df = pd.DataFrame(df_dict)

    return df, tau_star_clamped, mu0


def verify_estimator_properties(
    learner,
    n_samples: int = 15000,
    n_features: int = 6,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates a causal learner against KNOWN synthetic ground truth:
      - Estimator Bias: E[tau_hat - tau_true]
      - PEHE: sqrt(E[(tau_hat - tau_true)^2])
      - Spearman rank correlation: rank correlation between tau_hat and tau_true
      - Kendall's Tau: concordance of individual effect rankings
    """
    df, true_tau, true_mu0 = generate_synthetic_cate_dataset(
        n_samples=n_samples,
        n_features=n_features,
        heterogeneity_type="nonlinear",
        random_seed=random_seed
    )

    feature_cols = [f"x{i}" for i in range(n_features)]
    X = df[feature_cols].values
    y = df["y"].values
    w = df["treatment"].values

    # Train learner
    learner.fit(X, y, w)
    pred_tau = learner.predict_uplift(X)

    # Evaluation metrics
    bias = float(np.mean(pred_tau - true_tau))
    pehe = float(np.sqrt(mean_squared_error(true_tau, pred_tau)))
    spearman_corr, _ = stats.spearmanr(true_tau, pred_tau)
    kendall_corr, _ = stats.kendalltau(true_tau, pred_tau)

    return {
        "mean_true_tau": float(np.mean(true_tau)),
        "mean_pred_tau": float(np.mean(pred_tau)),
        "estimator_bias": bias,
        "pehe_rmse": pehe,
        "spearman_rank_correlation": float(spearman_corr),
        "kendall_tau": float(kendall_corr),
        "true_tau_sample": true_tau[:500].tolist(),
        "pred_tau_sample": pred_tau[:500].tolist()
    }


def verify_false_positive_rate_under_null(
    num_simulations: int = 200,
    sample_size: int = 2000,
    alpha: float = 0.05,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executes Monte Carlo A/A tests under H0 (true effect strictly zero).
    Measures the empirical False Positive Rate (FPR) and tests if it matches
    the nominal significance level alpha (e.g. 5%).
    """
    rng = np.random.default_rng(random_seed)
    p_values = []
    rejections = 0

    for i in range(num_simulations):
        sim_seed = random_seed + i
        df, _, _ = generate_synthetic_cate_dataset(
            n_samples=sample_size,
            n_features=4,
            heterogeneity_type="null",
            random_seed=sim_seed
        )

        n_t = (df["treatment"] == 1).sum()
        n_c = (df["treatment"] == 0).sum()
        x_t = df[df["treatment"] == 1]["y"].sum()
        x_c = df[df["treatment"] == 0]["y"].sum()

        p_pool = (x_t + x_c) / (n_t + n_c)
        se_pool = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_t + 1.0 / n_c))
        z = (x_t / n_t - x_c / n_c) / se_pool if se_pool > 0 else 0.0
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

        p_values.append(p_val)
        if p_val < alpha:
            rejections += 1

    empirical_fpr = rejections / num_simulations

    # Binomial test of whether empirical FPR matches nominal alpha
    binom_p = stats.binomtest(k=rejections, n=num_simulations, p=alpha).pvalue

    return {
        "num_simulations": num_simulations,
        "nominal_alpha": alpha,
        "rejections": rejections,
        "empirical_fpr": float(empirical_fpr),
        "fpr_calibration_p_value": float(binom_p),
        "is_calibrated": bool(binom_p >= 0.01),
        "p_values": p_values
    }


def verify_confidence_interval_coverage(
    num_simulations: int = 100,
    sample_size: int = 5000,
    nominal_confidence: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Tests whether the 95% confidence intervals constructed by the experiment engine
    achieve nominal 95% coverage probability of the TRUE population Average Treatment Effect.
    """
    alpha = 1.0 - nominal_confidence
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    covered_count = 0
    true_ate = 0.04  # Population parameter in this DGP

    for i in range(num_simulations):
        df, true_tau, _ = generate_synthetic_cate_dataset(
            n_samples=sample_size,
            heterogeneity_type="linear",
            random_seed=random_seed + i
        )
        treated = df[df["treatment"] == 1]["y"]
        control = df[df["treatment"] == 0]["y"]

        p_t = treated.mean()
        p_c = control.mean()
        diff = p_t - p_c
        se = np.sqrt(p_t * (1 - p_t) / len(treated) + p_c * (1 - p_c) / len(control))

        ci_lower = diff - z_crit * se
        ci_upper = diff + z_crit * se

        # True population parameter
        pop_true = float(np.mean(true_tau))
        if ci_lower <= pop_true <= ci_upper:
            covered_count += 1

    coverage_rate = covered_count / num_simulations

    return {
        "num_simulations": num_simulations,
        "nominal_confidence": nominal_confidence,
        "empirical_coverage_rate": float(coverage_rate),
        "covered_count": covered_count,
        "is_valid_coverage": bool(abs(coverage_rate - nominal_confidence) <= 0.06)
    }
