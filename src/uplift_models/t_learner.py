"""
T-Learner (Two-Model Uplift Estimator).
Estimates CATE using two separate supervised models, one for treatment and one for control:
    tau_hat(X) = mu1_hat(X) - mu0_hat(X)
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from .base import BaseUpliftLearner


class TLearner(BaseUpliftLearner):
    """
    T-Learner (Two Model) Uplift Estimator.
    Fits separate estimators for treatment (W=1) and control (W=0) cohorts.
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,
        estimator_type: str = "lightgbm",
        random_state: int = 42,
        **estimator_kwargs
    ):
        self.estimator_type = estimator_type.lower()
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs

        if estimator is not None:
            self.model_1 = clone(estimator)
            self.model_0 = clone(estimator)
        elif self.estimator_type == "lightgbm":
            params = {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 5,
                "num_leaves": 31,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "verbose": -1,
                "n_jobs": -1
            }
            params.update(estimator_kwargs)
            self.model_1 = lgb.LGBMClassifier(**params)
            # Use different seed for control model for statistical independence
            params0 = dict(params)
            params0["random_state"] = self.random_state + 1
            self.model_0 = lgb.LGBMClassifier(**params0)
        elif self.estimator_type == "logistic_regression":
            params = {"max_iter": 500, "random_state": self.random_state}
            params.update(estimator_kwargs)
            self.model_1 = LogisticRegression(**params)
            self.model_0 = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported estimator_type: {estimator_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "TLearner":
        mask_1 = (treatment == 1)
        mask_0 = (treatment == 0)

        if np.sum(mask_1) == 0 or np.sum(mask_0) == 0:
            raise ValueError("Both treatment and control cohorts must have non-empty samples.")

        self.model_1.fit(X[mask_1], y[mask_1])
        self.model_0.fit(X[mask_0], y[mask_0])
        return self

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(self.model_1, "predict_proba"):
            y1_prob = self.model_1.predict_proba(X)[:, 1]
            y0_prob = self.model_0.predict_proba(X)[:, 1]
        else:
            y1_prob = self.model_1.predict(X)
            y0_prob = self.model_0.predict(X)

        return y1_prob, y0_prob

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        y1, y0 = self.predict_counterfactuals(X)
        return y1 - y0
