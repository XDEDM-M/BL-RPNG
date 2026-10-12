from __future__ import annotations

from typing import Any


def storage_preflight(
    *, current_bytes: int, estimated_new_bytes: int, maximum_gb: float = 300.0
) -> dict[str, Any]:
    maximum = int(maximum_gb * 2**30)
    projected = int(current_bytes) + int(estimated_new_bytes)
    return {
        "passed": projected <= maximum,
        "current_bytes": int(current_bytes),
        "estimated_new_bytes": int(estimated_new_bytes),
        "projected_bytes": projected,
        "maximum_bytes": maximum,
    }


def solver_gate(
    *,
    convergence_fraction: float,
    valid_correction_fraction: float,
    boundary_error: float,
    divergence_error: float | None = None,
    timing_cv: float | None = None,
    minimum_convergence: float = 0.99,
    minimum_valid_correction: float = 0.95,
    boundary_tolerance: float = 1e-6,
) -> dict[str, Any]:
    checks = {
        "convergence": convergence_fraction >= minimum_convergence,
        "valid_correction": valid_correction_fraction >= minimum_valid_correction,
        "boundary": boundary_error <= boundary_tolerance,
        "divergence": divergence_error is None or divergence_error <= 1e-6,
        "timing_cv": timing_cv is None or timing_cv <= 0.10,
    }
    return {"passed": all(checks.values()), "checks": checks}


def nonlinear_method_gate(
    *,
    oracle_ratio_ci95: tuple[float, float],
    scratch_improvement: float,
    marginal_speedup_ci95: tuple[float, float],
    seed_wins: int,
    seeds_completed: int,
) -> dict[str, Any]:
    required_wins = 1 if seeds_completed == 1 else 4
    checks = {
        "oracle_noninferiority": oracle_ratio_ci95[1] <= 1.05,
        "scratch_improvement": scratch_improvement >= 0.75,
        "marginal_speedup": marginal_speedup_ci95[0] >= 10.0,
        "seed_wins": seed_wins >= required_wins,
    }
    return {"passed": all(checks.values()), "checks": checks}


def poisson_negative_control_gate(path_improvement: float) -> dict[str, Any]:
    """Path should not create a material gain for a one-step linear solve."""
    return {
        "passed": abs(float(path_improvement)) <= 0.05,
        "absolute_path_improvement": abs(float(path_improvement)),
        "maximum_expected": 0.05,
    }
