"""Simulation and statistical verification package."""
from .synthetic_validator import (
    generate_synthetic_cate_dataset,
    verify_estimator_properties,
    verify_false_positive_rate_under_null,
    verify_confidence_interval_coverage,
)

__all__ = [
    "generate_synthetic_cate_dataset",
    "verify_estimator_properties",
    "verify_false_positive_rate_under_null",
    "verify_confidence_interval_coverage",
]
