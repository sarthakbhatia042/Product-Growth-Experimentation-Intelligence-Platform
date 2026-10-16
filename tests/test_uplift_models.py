"""
Unit tests for Causal ML Uplift Models (Baseline ATE, S-Learner, T-Learner, X-Learner, Cross-Fitting).
"""

import pytest
import numpy as np
from src.uplift_models.base import BaselineATELearner
from src.uplift_models.s_learner import SLearner
from src.uplift_models.t_learner import TLearner
from src.uplift_models.x_learner import XLearner
from src.uplift_models.cross_fit import cross_fit_uplift_predictions


@pytest.fixture
def synthetic_toy_data():
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.normal(0, 1, size=(n, 4))
    # True CATE has positive lift on x0 > 0
    w = rng.binomial(1, 0.5, size=n)
    p0 = 1.0 / (1.0 + np.exp(-(-2.0 + 0.5 * X[:, 0])))
    cate = np.where(X[:, 0] > 0, 0.08, 0.01)
    p1 = np.clip(p0 + cate, 0.01, 0.99)

    y = np.where(w == 1, rng.binomial(1, p1), rng.binomial(1, p0))
    return X, y, w


def test_baseline_ate_learner(synthetic_toy_data):
    X, y, w = synthetic_toy_data
    model = BaselineATELearner()
    model.fit(X, y, w)

    preds = model.predict_uplift(X)
    assert len(preds) == len(X)
    # Baseline predicts a single constant value for all units
    assert np.all(preds == preds[0])
    y1, y0 = model.predict_counterfactuals(X)
    assert len(y1) == len(X)
    assert len(y0) == len(X)


def test_s_learner(synthetic_toy_data):
    X, y, w = synthetic_toy_data
    model = SLearner(estimator_type="lightgbm", n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y, w)

    uplift = model.predict_uplift(X)
    assert uplift.shape == (X.shape[0],)
    # Check that predictions are heterogeneous (not constant)
    assert np.std(uplift) > 0.0

    y1, y0 = model.predict_counterfactuals(X)
    assert np.all(y1 >= 0.0) and np.all(y1 <= 1.0)
    assert np.all(y0 >= 0.0) and np.all(y0 <= 1.0)


def test_t_learner(synthetic_toy_data):
    X, y, w = synthetic_toy_data
    model = TLearner(estimator_type="lightgbm", n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y, w)

    uplift = model.predict_uplift(X)
    assert uplift.shape == (X.shape[0],)
    assert np.std(uplift) > 0.0


def test_x_learner(synthetic_toy_data):
    X, y, w = synthetic_toy_data
    model = XLearner(estimator_type="lightgbm", n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y, w)

    uplift = model.predict_uplift(X)
    assert uplift.shape == (X.shape[0],)
    assert np.std(uplift) > 0.0


def test_cross_fit_uplift(synthetic_toy_data):
    X, y, w = synthetic_toy_data
    oof_preds = cross_fit_uplift_predictions(
        TLearner,
        X, y, w,
        n_splits=2,
        estimator_type="logistic_regression"
    )
    assert oof_preds.shape == (X.shape[0],)
    assert np.all(np.isfinite(oof_preds))
