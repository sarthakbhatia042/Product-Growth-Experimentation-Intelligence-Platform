"""
S-Learner (Single Model Uplift Estimator).
Estimates CATE using a single supervised model conditioned on features X and treatment W:
    tau_hat(X) = mu_hat(X, 1) - mu_hat(X, 0)
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from .base import BaseUpliftLearner


class SLearner(BaseUpliftLearner):
    """
    S-Learner (Single Model) Uplift Estimator.
    Supports LightGBM (default) or Logistic Regression base estimators.
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,
        estimator_type: str = "lightgbm",
        include_interactions: bool = True,
        random_state: int = 42,
        **estimator_kwargs
    ):
        self.estimator_type = estimator_type.lower()
        self.include_interactions = include_interactions
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs

        if estimator is not None:
            self.model = estimator
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
            self.model = lgb.LGBMClassifier(**params)
        elif self.estimator_type == "logistic_regression":
            params = {"max_iter": 500, "random_state": self.random_state}
            params.update(estimator_kwargs)
            self.model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported estimator_type: {estimator_type}")

    def _augment_features(self, X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        w = treatment.reshape(-1, 1).astype(float)
        if self.include_interactions:
            # [X, W, X * W]
            interactions = X * w
            return np.hstack([X, w, interactions])
        else:
            return np.hstack([X, w])

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "SLearner":
        X_aug = self._augment_features(X, treatment)
        self.model.fit(X_aug, y)
        return self

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        ones = np.ones(n, dtype=int)
        zeros = np.zeros(n, dtype=int)

        X_treat = self._augment_features(X, ones)
        X_ctrl = self._augment_features(X, zeros)

        if hasattr(self.model, "predict_proba"):
            y1_prob = self.model.predict_proba(X_treat)[:, 1]
            y0_prob = self.model.predict_proba(X_ctrl)[:, 1]
        else:
            y1_prob = self.model.predict(X_treat)
            y0_prob = self.model.predict(X_ctrl)

        return y1_prob, y0_prob

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        y1, y0 = self.predict_counterfactuals(X)
        return y1 - y0
