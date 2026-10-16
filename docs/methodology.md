# Statistical & Causal Inference Methodology

## 1. Introduction & Theoretical Foundations
Randomized Controlled Trials (A/B tests) represent the gold standard for causal inference in product growth and marketing. However, classical A/B experimentation typically focuses on the **Average Treatment Effect (ATE)** or **Intention-to-Treat (ITT)**, implicitly treating user populations as homogeneous. 

In digital marketing and e-commerce interventions, treatment effects are intensely heterogeneous:
* **Persuadables**: Users who convert only when targeted.
* **Sure Things**: Users who would convert regardless of whether an ad is shown.
* **Lost Causes**: Users who will never convert, regardless of ad spend.
* **Sleeping Dogs (Do-Not-Disturbs)**: Users who are alienated or annoyed by the ad, converting less when targeted.

This platform bridges classical randomized experimentation with **Causal Machine Learning (Uplift Modeling)** and strategic unit economics to identify persuadables, prevent wasteful ad spend on sure things/lost causes, and avoid negative responders.

---

## 2. Experimental Setup & Identification Assumptions

### 2.1 Experimental Unit & Intervention
* **Experimental Unit**: An individual user identity (cookie/browser).
* **Randomized Intervention ($T \in \{0, 1\}$)**:
  * $T = 1$: Assigned to the marketing campaign (retargeting ads enabled).
  * $T = 0$: Assigned to the control group (no ads shown / withheld).
* **Intermediate Compliance / Delivery ($D \in \{0, 1\}$)**:
  * $D = 1$: User was effectively exposed to an ad impression.
  * $D = 0$: User received no impression.
  * Crucially, in the Criteo dataset, $\mathbb{P}(D=1 \mid T=0) = 0$ (strict **one-sided non-compliance**).
* **Observed Outcomes**:
  * $Y \in \{0, 1\}$: Binary conversion (primary business outcome, purchase/sign-up).
  * $V \in \{0, 1\}$: Binary website visit (secondary behavioral outcome).

### 2.2 Core Causal Assumptions
For causal estimands to be identified from the observed data, four fundamental conditions must hold:

1. **Stable Unit Treatment Value Assumption (SUTVA)**:
   * **No Interference**: The treatment assignment of user $i$ does not affect the potential outcomes of user $j$ ($\{Y_j(0), Y_j(1)\} \perp T_i$).
   * **Single Version of Treatment**: All treated units assigned to $T=1$ receive the identical retargeting protocol.
   * *Limitation*: In dynamic ad auctions with high budget pressure, withholding control users might slightly shift auction clearing prices for other users, but at the user-cookie level, interference is negligible.

2. **Unconfoundedness (Ignorability / Random Assignment)**:
   $$(Y(0), Y(1), D(0), D(1)) \perp T \mid X$$
   Because treatment assignment was executed via a centralized random hash, assignment is independent of all pre-treatment covariates $X$ and potential outcomes.

3. **Positivity (Overlap)**:
   $$0 < \mathbb{P}(T=1 \mid X) < 1 \quad \forall X \in \mathcal{X}$$
   Empirically verified in the Criteo dataset with $\mathbb{P}(T=1) \approx 0.850$.

4. **Exclusion Restriction (for CACE / Instrumental Variables)**:
   The randomized assignment $T$ affects conversion $Y$ **only** through the actual ad delivery $D$. That is, simply being eligible for ads without actually seeing an impression cannot causally alter user conversion.

---

## 3. Estimand Taxonomy & Mathematical Formulations

### 3.1 Intention-to-Treat (ITT)
The expected causal effect of offering or assigning the intervention across the entire targeted population:
$$\text{ITT}_Y = \mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]$$

#### Relative Lift and the Delta Method
$$\text{Rel Lift} = \frac{\mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]}{\mathbb{E}[Y \mid T=0]} = \frac{p_T}{p_C} - 1$$

When base rates are small (e.g. $p_C \approx 0.22\%$), naive symmetric confidence intervals can yield impossible negative lower bounds ($< -100\%$). We construct asymmetric confidence intervals using the **Delta Method on the logarithmic relative risk scale**:
$$\phi = \log\left(\frac{p_T}{p_C}\right) = \log(p_T) - \log(p_C)$$
$$\mathbb{V}\text{ar}(\hat{\phi}) = \frac{1 - p_T}{N_T p_T} + \frac{1 - p_C}{N_C p_C}$$
$$\text{CI}_{1-\alpha}(\text{Rel Lift}) = \left[\frac{p_T}{p_C} \exp\left(-z_{\alpha/2} \sqrt{\mathbb{V}\text{ar}(\hat{\phi})}\right) - 1, \; \frac{p_T}{p_C} \exp\left(+z_{\alpha/2} \sqrt{\mathbb{V}\text{ar}(\hat{\phi})}\right) - 1\right]$$

### 3.2 Complier Average Causal Effect (CACE / LATE)
Under one-sided non-compliance ($\mathbb{P}(D=1 \mid T=0) = 0$), the Wald Instrumental Variable estimator recovers the treatment effect specifically on **compliers** (those who actually received the impression when targeted):
$$\text{CACE} = \frac{\mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]}{\mathbb{E}[D \mid T=1] - \mathbb{E}[D \mid T=0]} = \frac{\text{ITT}_Y}{\mathbb{E}[D \mid T=1]}$$

#### Asymptotic Variance via Delta Method
Let $R = \frac{\bar{Y}_T - \bar{Y}_C}{\bar{D}_T}$. The asymptotic variance is:
$$\mathbb{V}\text{ar}(\text{CACE}) \approx \frac{1}{\bar{D}_T^2} \left(\mathbb{V}\text{ar}(\bar{Y}_T) + \mathbb{V}\text{ar}(\bar{Y}_C)\right) + \frac{\text{ITT}_Y^2}{\bar{D}_T^4} \mathbb{V}\text{ar}(\bar{D}_T) - 2 \frac{\text{ITT}_Y}{\bar{D}_T^3} \mathbb{C}\text{ov}(\bar{Y}_T, \bar{D}_T)$$

### 3.3 Conditional Average Treatment Effect (CATE / Uplift)
$$\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$$
Estimates individual expected incremental uplift conditioned on the 12 dense continuous context features $X \in \mathbb{R}^{12}$.

---

## 4. Sample Ratio Mismatch (SRM) & Sequential Peeking

### 4.1 SRM Detection
Sample Ratio Mismatch occurs when the observed ratio between variants deviates significantly from the planned design ratio (e.g. 84.6% : 15.4%). Tested using Pearson's Goodness-of-Fit test:
$$\chi^2 = \frac{(O_T - E_T)^2}{E_T} + \frac{(O_C - E_C)^2}{E_C} \sim \chi^2(1)$$
Industry standard threshold: $\alpha_{\text{SRM}} = 0.001$.

### 4.2 Sequential Peeking Risk & Alpha-Spending Functions
Continuous monitoring of A/B tests with naive stopping rules inflates Type I errors from $5\%$ up to $25\%-40\%$. To preserve nominal $\alpha = 0.05$ across interim checks, we employ **Lan-DeMets Alpha Spending Functions**:
* **O'Brien-Fleming Analog**:
  $$\alpha(t) = 2 - 2\Phi\left(\frac{z_{\alpha/2}}{\sqrt{t}}\right) \quad \text{for information fraction } t \in (0, 1]$$
  Very conservative at early stages, preserving statistical power for the final analysis.
* **Pocock Analog**:
  $$\alpha(t) = \alpha \ln(1 + (e - 1)t)$$
  Spends alpha more evenly across interim checkpoints.

---

## 5. Causal Machine Learning Architectures

### 5.1 Baseline ATE
Assigns the overall population ITT estimate $\hat{\tau}_{\text{ATE}} = \bar{Y}_T - \bar{Y}_C$ uniformly to all individuals.

### 5.2 S-Learner (Single Model)
Fits a single model $f(X, W)$ on features and treatment indicator:
$$\hat{\tau}_{\text{S}}(X) = \hat{f}(X, 1) - \hat{f}(X, 0)$$
*Pitfall*: When features $X$ have strong baseline predictive power and treatment effect $\tau$ is small, tree-based models frequently split on $X$ and ignore $W$, shrinking predicted uplift toward zero.

### 5.3 T-Learner (Two Models)
Fits two separate models:
* $\hat{\mu}_1(X)$ trained on treated units $\{X_i \mid W_i = 1\}$
* $\hat{\mu}_0(X)$ trained on control units $\{X_i \mid W_i = 0\}$
$$\hat{\tau}_{\text{T}}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$$
Prevents treatment regularization shrinkage, but can suffer from divergent inductive biases in regions with small sample size.

### 5.4 X-Learner (Künzel et al. 2019)
Specially engineered for imbalanced allocations (e.g., Criteo's 85% : 15% ratio):
1. **Stage 1**: Fit response models $\hat{\mu}_1(X)$ and $\hat{\mu}_0(X)$.
2. **Stage 2**: Compute imputed counterfactual effects:
   $$D_{1,i} = Y_{1,i} - \hat{\mu}_0(X_{1,i}) \quad \text{for treated}$$
   $$D_{0,j} = \hat{\mu}_1(X_{0,j}) - Y_{0,j} \quad \text{for control}$$
3. **Stage 3**: Fit effect regressors:
   $$\hat{\tau}_1(X) \sim (X_1, D_1), \quad \hat{\tau}_0(X) \sim (X_0, D_0)$$
4. **Stage 4**: Combine using propensity score $e(X) = \mathbb{P}(W=1 \mid X)$:
   $$\hat{\tau}_{\text{X}}(X) = e(X) \hat{\tau}_0(X) + (1 - e(X)) \hat{\tau}_1(X)$$

---

## 6. Uplift Evaluation Metrics & Radcliffe Qini Curve

Because true counterfactuals $(Y_i(1), Y_i(0))$ cannot be simultaneously observed for any user, standard supervised metrics (RMSE, AUC, Log-Loss) cannot evaluate uplift models. Instead, evaluation relies on **ranking concordance and cumulative policy gains**:

### 6.1 Radcliffe Qini Curve
For population fraction $k$ targeted (ordered by descending predicted uplift):
$$\text{Qini}(k) = Y_T(k) - Y_C(k) \cdot \frac{N_T(k)}{N_C(k)}$$
* **Qini Score**: Area between model Qini curve and the random targeting diagonal baseline.

### 6.2 Cumulative Uplift Curve & AUUC
$$\text{Cumulative Uplift}(k) = \left(\frac{Y_T(k)}{N_T(k)} - \frac{Y_C(k)}{N_C(k)}\right) \cdot (N_T(k) + N_C(k))$$
* **AUUC**: Area Under the Uplift Curve minus the random diagonal baseline.

---

## 7. Product Economics & Optimal Policy Optimization

### 7.1 Break-Even Uplift Threshold
Let $V$ be the customer conversion value and $C$ be the contact cost per targeted user. A user should only be targeted if:
$$\hat{\tau}(X) > \tau^* = \frac{C}{V}$$
For example, with $V = \$50$ and $C = \$0.10$:
$$\tau^* = \frac{0.10}{50} = 0.002 = \mathbf{0.20\%}$$
Any user with predicted uplift $< 0.20\%$ generates negative expected profit.

### 7.2 Net Incremental Profit under Budget $B$
$$\text{Profit}(\pi_k) = V \cdot \sum_{i \in \pi_k} \hat{\tau}(X_i) - C \cdot k \cdot N$$
$$\text{Subject to: } C \cdot k \cdot N \le B$$
$$\text{ROI}(\pi_k) = \frac{\text{Profit}(\pi_k)}{C \cdot k \cdot N} \times 100\%$$
