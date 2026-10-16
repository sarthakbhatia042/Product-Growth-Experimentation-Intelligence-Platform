"""
X-Learner (Künzel et al. 2019 Uplift Estimator).
Designed specifically for scenarios with unbalanced treatment assignment (such as Criteo's 85:15 ratio).
Combines imputed counterfactual effects weighted by propensity scores.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import clone

from .base import BaseUpliftLearner


class XLearner(BaseUpliftLearner):
    """
    X-Learner implementation using LightGBM or Linear/Logistic models.
    Stage 1: Response functions mu1(X), mu0(X)
    Stage 2: Imputed treatment effects D1 = Y1 - mu0(X1), D0 = mu1(X0) - Y0
    Stage 3: Second stage regressors tau1(X), tau0(X)
    Stage 4: Propensity weighted combination: tau(X) = e(X)*tau0(X) + (1-e(X))*tau1(X)
    """

    def __init__(
        self,
        estimator_type: str = "lightgbm",
        random_state: int = 42,
        **estimator_kwargs
    ):
        self.estimator_type = estimator_type.lower()
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs

        if self.estimator_type == "lightgbm":
            base_params = {
                "n_estimators": 80,
                "learning_rate": 0.05,
                "max_depth": 5,
                "num_leaves": 31,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbose": -1,
                "n_jobs": -1
            }
            base_params.update(estimator_kwargs)

            self.model_1 = lgb.LGBMClassifier(**base_params, random_state=random_state)
            self.model_0 = lgb.LGBMClassifier(**base_params, random_state=random_state + 1)
            self.tau_model_1 = lgb.LGBMRegressor(**base_params, random_state=random_state + 2)
            self.tau_model_0 = lgb.LGBMRegressor(**base_params, random_state=random_state + 3)
            self.propensity_model = LogisticRegression(max_iter=300, random_state=random_state)
        else:
            self.model_1 = LogisticRegression(max_iter=500, random_state=random_state)
            self.model_0 = LogisticRegression(max_iter=500, random_state=random_state + 1)
            self.tau_model_1 = LinearRegression()
            self.tau_model_0 = LinearRegression()
            self.propensity_model = LogisticRegression(max_iter=300, random_state=random_state)

        self.propensity_scores_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "XLearner":
        mask_1 = (treatment == 1)
        mask_0 = (treatment == 0)

        X1, y1 = X[mask_1], y[mask_1]
        X0, y0 = X[mask_0], y[mask_0]

        # Stage 1: Fit base response functions
        self.model_1.fit(X1, y1)
        self.model_0.fit(X0, y0)

        # Stage 2: Calculate imputed counterfactual treatment effects
        if hasattr(self.model_0, "predict_proba"):
            mu0_X1 = self.model_0.predict_proba(X1)[:, 1]
            mu1_X0 = self.model_1.predict_proba(X0)[:, 1]
        else:
            mu0_X1 = self.model_0.predict(X1)
            mu1_X0 = self.model_1.predict(X0)

        D1 = y1 - mu0_X1
        D0 = mu1_X0 - y0

        # Stage 3: Fit effect models on imputed treatment effects
        self.tau_model_1.fit(X1, D1)
        self.tau_model_0.fit(X0, D0)

        # Stage 4: Fit propensity score model
        self.propensity_model.fit(X, treatment)

        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        # Propensity score e(X) = P(W=1 | X)
        e_X = self.propensity_model.predict_proba(X)[:, 1]

        tau1_pred = self.tau_model_1.predict(X)
        tau0_pred = self.tau_model_0.predict(X)

        # Künzel formula: e(X) * tau0(X) + (1 - e(X)) * tau1(X)
        return e_X * tau0_pred + (1.0 - e_X) * tau1_pred

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(self.model_1, "predict_proba"):
            y1_prob = self.model_1.predict_proba(X)[:, 1]
            y0_prob = self.model_0.predict_proba(X)[:, 1]
        else:
            y1_prob = self.model_1.predict(X)
            y0_prob = self.model_0.predict(X)
        return y1_prob, y0_prob
