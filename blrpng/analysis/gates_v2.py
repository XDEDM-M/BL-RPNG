from __future__ import annotations

from typing import Any


def calibration_gate(
    *,
    oracle_ratio_ci95: tuple[float, float],
    marginal_speedup_ci95: tuple[float, float],
    scratch_improvement: float,
    ood_frequency_degradation: float | None = None,
    ood_amplitude_degradation: float | None = None,
    oracle_margin: float = 0.05,
    minimum_speedup: float = 10.0,
    minimum_scratch_improvement: float = 0.75,
) -> dict[str, Any]:
    checks = {
        "oracle_noninferiority": float(oracle_ratio_ci95[1]) <= 1.0 + oracle_margin,
        "marginal_speedup": float(marginal_speedup_ci95[0]) >= minimum_speedup,
        "scratch_improvement": float(scratch_improvement) >= minimum_scratch_improvement,
        "ood_not_both_degraded": not (
            ood_frequency_degradation is not None
            and ood_amplitude_degradation is not None
            and ood_frequency_degradation > 0.05
            and ood_amplitude_degradation > 0.05
        ),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "oracle_ratio_ci95": list(oracle_ratio_ci95),
        "marginal_speedup_ci95": list(marginal_speedup_ci95),
        "scratch_improvement": scratch_improvement,
    }
