"""Experimentation package for A/B testing, SRM, and sequential testing."""
from .engine import analyze_binary_experiment, estimate_cace, calculate_relative_lift_ci, adjust_multiple_testing, LiftMetrics, CACEMetrics
from .srm import check_sample_ratio_mismatch, SRMTestResult
from .sequential import simulate_peeking_risk, calculate_alpha_spending_boundaries

__all__ = [
    "analyze_binary_experiment",
    "estimate_cace",
    "calculate_relative_lift_ci",
    "adjust_multiple_testing",
    "LiftMetrics",
    "CACEMetrics",
    "check_sample_ratio_mismatch",
    "SRMTestResult",
    "simulate_peeking_risk",
    "calculate_alpha_spending_boundaries"
]
