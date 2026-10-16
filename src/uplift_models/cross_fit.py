"""
Cross-Fitting (K-Fold Out-of-Fold Uplift Estimation).
Implements Neyman-orthogonality style K-fold cross-fitting to generate
leakage-free out-of-sample uplift predictions.
"""

from typing import Dict, Any, List, Type
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .base import BaseUpliftLearner


def cross_fit_uplift_predictions(
    learner_cls: Type[BaseUpliftLearner],
    X: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    n_splits: int = 3,
    random_state: int = 42,
    **learner_kwargs
) -> np.ndarray:
    """
    Computes out-of-fold uplift predictions using K-Fold stratified cross-fitting.
    Prevents overfitting and leakage by scoring each sample using a model
    trained without that sample.

    Parameters
    ----------
    learner_cls : Type[BaseUpliftLearner]
        Class of the uplift learner (e.g. SLearner, TLearner, XLearner).
    X : np.ndarray
        Covariates matrix.
    y : np.ndarray
        Binary outcome vector.
    treatment : np.ndarray
        Binary treatment vector.
    n_splits : int
        Number of cross-fitting folds (default: 3).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Out-of-fold predicted uplift array matching X.shape[0].
    """
    oof_predictions = np.zeros(X.shape[0], dtype=float)

    # Stratify by compound treatment and outcome to maintain balance across folds
    strat_key = treatment.astype(str) + "_" + y.astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_key)):
        X_train, y_train, w_train = X[train_idx], y[train_idx], treatment[train_idx]
        X_val = X[val_idx]

        model = learner_cls(random_state=random_state + fold, **learner_kwargs)
        model.fit(X_train, y_train, w_train)

        oof_predictions[val_idx] = model.predict_uplift(X_val)

    return oof_predictions
