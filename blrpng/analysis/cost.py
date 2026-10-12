from __future__ import annotations

from typing import Any

import numpy as np


def target_cost(
    *,
    assembly_seconds: float,
    solve_seconds: float,
    adaptation_seconds: float,
    inference_seconds_per_query: float,
    queries: int,
) -> float:
    return float(assembly_seconds + solve_seconds + adaptation_seconds + queries * inference_seconds_per_query)


def break_even_queries(
    *,
    learned_fixed: float,
    learned_per_query: float,
    reference_fixed: float,
    reference_per_query: float,
) -> int | None:
    slope = reference_per_query - learned_per_query
    if slope <= 0:
        return None
    return max(int(np.ceil((learned_fixed - reference_fixed) / slope)), 0)


def amortized_cost(pretraining_seconds: float, target_seconds: float, target_tasks: int) -> float:
    if target_tasks <= 0:
        raise ValueError("target_tasks must be positive.")
    return float(pretraining_seconds / target_tasks + target_seconds)


def cost_summary(
    method: dict[str, float],
    oracle: dict[str, float],
    *,
    queries: int,
    pretraining_seconds: float = 0.0,
    amortization_tasks: int = 1,
) -> dict[str, Any]:
    candidate = target_cost(
        assembly_seconds=method["assembly_seconds"], solve_seconds=method["solve_seconds"],
        adaptation_seconds=method["adaptation_seconds"],
        inference_seconds_per_query=method["inference_seconds_per_query"], queries=queries,
    )
    reference = target_cost(
        assembly_seconds=oracle["assembly_seconds"], solve_seconds=oracle["solve_seconds"],
        adaptation_seconds=oracle["adaptation_seconds"],
        inference_seconds_per_query=oracle["inference_seconds_per_query"], queries=queries,
    )
    return {
        "queries": queries, "marginal_method_seconds": candidate,
        "marginal_oracle_seconds": reference,
        "marginal_speedup": reference / max(candidate, 1e-15),
        "amortized_method_seconds": amortized_cost(pretraining_seconds, candidate, amortization_tasks),
        "break_even_queries": break_even_queries(
            learned_fixed=method["assembly_seconds"] + method["solve_seconds"] + method["adaptation_seconds"],
            learned_per_query=method["inference_seconds_per_query"],
            reference_fixed=oracle["assembly_seconds"] + oracle["solve_seconds"] + oracle["adaptation_seconds"],
            reference_per_query=oracle["inference_seconds_per_query"],
        ),
    }
