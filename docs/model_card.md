# Model Card: Causal Uplift & Targeting Models

## 1. Model Details
* **Model Families**: Baseline ATE, S-Learner, T-Learner, X-Learner (Künzel et al. 2019).
* **Base Estimators**: LightGBM (Gradient Boosted Trees) and L2-regularized Logistic Regression.
* **Target Objective**: Conditional Average Treatment Effect (CATE / Uplift): $\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$.
* **Version**: 1.0.0.
* **Developer**: Senior Data Scientist & ML Engineer.
* **License**: Open Source for Research / Portfolio Evaluation.

---

## 2. Intended Use & Target Audience
* **Primary Intended Use**:
  * Marketing campaign optimization and retargeting audience segmentation.
  * Ranking users by expected incremental conversion uplift to maximize Return on Investment (ROI) under fixed budgets.
  * Estimating heterogeneous treatment effects to detect negative responders ("sleeping dogs") and prevent wasted ad spend on "sure things".
* **Out-of-Scope Use Cases**:
  * Individualized credit underwriting or financial solvency scoring (dataset contains anonymized marketing engagement features, not credit bureau history).
  * Medical treatment prescribing or clinical decision-making.

---

## 3. Data & Pre-processing

### 3.1 Dataset Specification
* **Source**: Official Criteo Uplift Prediction Benchmark Dataset (`criteo-research-uplift-v2.1`).
* **Experimental Design**: Randomized controlled trial with centralized hashing.
* **Feature Schema**:
  * `f0` to `f11`: 12 dense continuous anonymized and randomly projected pre-treatment features.
  * `treatment`: Binary treatment indicator (1 = targeted, 0 = control).
  * `exposure`: Binary impression delivery flag.
  * `conversion`: Primary binary business conversion label (baseline rate $\sim 0.22\%$).
  * `visit`: Secondary binary website visit label (baseline rate $\sim 4.00\%$).
* **Cohort Allocation**: $\approx 85.0\%$ treatment, $15.0\%$ control.

### 3.2 Leakage Prevention & Data Separation
* **Compound Stratification**: Stratified splitting on compound key `treatment_conversion` to preserve rare outcome rates and assignment proportions.
* **Data Partitions**:
  * Train Set: 55% ($N = 137,500$)
  * Validation Set: 15% ($N = 37,500$)
  * Held-Out Test Set: 30% ($N = 75,000$)
* **Cross-Fitting**: Optional 3-Fold out-of-fold cross-fitting for unbiased in-sample uplift scoring.

---

## 4. Quantitative Performance & Evaluation

Evaluated on the **Held-Out Test Set** ($N = 75,000$ users):

| Model Name | AUUC Metric | Radcliffe Qini Score | Mean Pred Uplift | Std Pred Uplift |
| :--- | :---: | :---: | :---: | :---: |
| **X-Learner (LightGBM)** | **+10.38** | **+8.97** | +0.0007 | 0.0028 |
| **T-Learner (LightGBM)** | **+15.03** | **+12.87** | +0.0009 | 0.0031 |
| **S-Learner (LightGBM)** | -19.70 | -16.75 | +0.0006 | 0.0018 |
| **Baseline ATE** | -10.71 | -9.10 | +0.0009 | 0.0000 |

### Top-Decile Uplift Concentration
* **Decile 1 (Top 10% highest predicted uplift)**:
  * Treatment Conversion Rate: **2.054%**
  * Control Conversion Rate: **1.210%**
  * Observed Absolute Incremental Lift: **+0.844%** (+84.4 basis points)
* **Deciles 2 - 9**: Near-zero incremental lift ($\le 0.015\%$).
* **Decile 10 (Bottom 10%)**: Negative or diluted incremental response.

### Economic Simulation Results ($V = \$50, C = \$0.10, B = \$15,000$)
* **Optimal Model-Guided Policy (Top 10% Targeted)**:
  * Units Targeted: 7,500
  * Incremental Conversions: **63.3**
  * Campaign Cost: **$750.00**
  * Gross Revenue: **$3,163.98**
  * Net Profit: **+$2,413.98**
  * **ROI**: **+321.86%**
* **Treat All (Indiscriminate Targeting)**:
  * Units Targeted: 75,000
  * Incremental Conversions: 75.8
  * Campaign Cost: $7,500.00
  * Net Profit: **-$3,710.63**
  * **ROI**: **-49.48%** (Severe financial loss)

---

## 5. Synthetic Ground Truth Verification

To verify estimator correctness where true individual counterfactuals $\tau^*(X)$ are mathematically known:

| Metric | Nominal / Target | Empirical Result | Status |
| :--- | :---: | :---: | :---: |
| **Estimator Bias ($\mathbb{E}[\hat{\tau} - \tau^*]$)** | 0.0000 | **-0.0132** | Passed (Minimal Bias) |
| **PEHE RMSE** | Lower is better | **0.0399** | Passed (High Precision) |
| **Spearman Rank Correlation** | Positive | **0.6466** | Passed (Strong Rank Recovery) |
| **Kendall's Tau Concordance** | Positive | **0.4628** | Passed |
| **A/A False Positive Rate (Null)** | 5.00% | **6.67%** ($p = 0.3447$) | Calibrated |
| **95% CI Coverage Probability** | 95.00% | **98.00%** | Valid Coverage |

---

## 6. Limitations, Ethical Considerations & Failure Modes

1. **Non-Direct Observability of Ground Truth**:
   Individual causal effects can never be directly validated for any single individual. Models must be judged through cumulative ranking curves (AUUC, Qini) and synthetic benchmarks.
2. **Rare Outcome Imbalance**:
   Because baseline conversion is $\sim 0.22\%$, variance in small sample slices is elevated. Bootstrap confidence bands must always accompany policy value estimates.
3. **Auction SUTVA Violations**:
   In real advertising environments, budget-constrained bidding auctions create unit interference if targeting changes equilibrium bid prices.
4. **Anonymized Features**:
   Features $f_0 - f_{11}$ are randomly projected for privacy, preventing semantic domain interpretation.
