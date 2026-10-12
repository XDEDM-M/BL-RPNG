from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def data_efficiency_auc(
    records: list[dict[str, Any]],
    *,
    method_key: str = "method",
    budget_key: str = "boundary_budget",
    metric_key: str = "relative_l2",
) -> dict[str, float]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[str(row[method_key])][int(row[budget_key])].append(float(row[metric_key]))
    result = {}
    for method, by_budget in grouped.items():
        budgets = np.asarray(sorted(by_budget), dtype=np.float64)
        errors = np.asarray(
            [np.mean(by_budget[int(budget)]) for budget in budgets],
            dtype=np.float64,
        )
        x = np.log2(budgets)
        span = max(float(x[-1] - x[0]), 1.0)
        result[method] = float(np.trapezoid(errors, x=x) / span)
    return result


def best_final_stability(history: list[dict[str, Any]]) -> dict[str, float]:
    values = np.asarray(
        [float(row["validation"]["rollout_rel_l2"]) for row in history],
        dtype=np.float64,
    )
    tail = values[-min(5, len(values)) :]
    return {
        "best": float(values.min()),
        "final": float(values[-1]),
        "final_over_best": float(values[-1] / max(values.min(), 1e-15)),
        "last_five_std": float(tail.std()),
    }
