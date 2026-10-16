"""
Base class definitions for Causal Machine Learning and Uplift Estimators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


class BaseUpliftLearner(ABC):
    """
    Abstract base class for Causal Machine Learning Uplift Estimators.
    All learners must implement fit(X, y, treatment) and predict_uplift(X).
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "BaseUpliftLearner":
        """
        Fits the causal learner on pre-treatment covariates X, binary outcome y,
        and binary treatment indicator.
        """
        pass

    @abstractmethod
    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the Individual / Conditional Average Treatment Effect (CATE):
        tau(X) = E[Y(1) - Y(0) | X].
        """
        pass

    @abstractmethod
    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts potential outcomes: (E[Y(1) | X], E[Y(0) | X]).
        """
        pass


class BaselineATELearner(BaseUpliftLearner):
    """
    Naive baseline causal model that predicts a constant treatment effect
    equal to the overall Average Treatment Effect (ATE):
    tau_hat(X) = E[Y | T=1] - E[Y | T=0] for all X.
    """

    def __init__(self):
        self.ate_: float = 0.0
        self.p_treated_: float = 0.0
        self.p_control_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, treatment: np.ndarray) -> "BaselineATELearner":
        treated_mask = (treatment == 1)
        control_mask = (treatment == 0)

        self.p_treated_ = float(np.mean(y[treated_mask])) if np.sum(treated_mask) > 0 else 0.0
        self.p_control_ = float(np.mean(y[control_mask])) if np.sum(control_mask) > 0 else 0.0
        self.ate_ = self.p_treated_ - self.p_control_
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(X.shape[0],), fill_value=self.ate_, dtype=float)

    def predict_counterfactuals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y1 = np.full(shape=(X.shape[0],), fill_value=self.p_treated_, dtype=float)
        y0 = np.full(shape=(X.shape[0],), fill_value=self.p_control_, dtype=float)
        return y1, y0
