from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _index(records: list[dict[str, Any]], method: str, metric: str):
    indexed = {}
    for row in records:
        if row["method"] != method:
            continue
        key = (
            int(row["seed"]),
            int(row["boundary_id"]),
            int(row["initial_condition_id"]),
        )
        if key in indexed:
            raise ValueError(f"Duplicate statistical unit for {method}: {key}")
        indexed[key] = float(row[metric])
    return indexed


def paired_hierarchical_bootstrap(
    records: list[dict[str, Any]],
    *,
    method: str,
    baseline: str,
    metric: str = "relative_l2",
    replicates: int = 50_000,
    seed: int = 20260730,
) -> dict[str, Any]:
    """Paired seed→boundary→initial-condition bootstrap of relative gain."""
    candidate = _index(records, method, metric)
    reference = _index(records, baseline, metric)
    keys = sorted(set(candidate) & set(reference))
    if set(candidate) != set(reference):
        raise ValueError("Methods do not contain identical paired statistical units.")
    hierarchy: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for seed_id, boundary_id, initial_id in keys:
        hierarchy[seed_id][boundary_id].append(initial_id)
    seed_ids = np.asarray(sorted(hierarchy))
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        differences = []
        baselines = []
        for seed_id in sampled_seeds:
            boundary_ids = np.asarray(sorted(hierarchy[int(seed_id)]))
            sampled_boundaries = rng.choice(
                boundary_ids, size=len(boundary_ids), replace=True
            )
            for boundary_id in sampled_boundaries:
                initial_ids = np.asarray(hierarchy[int(seed_id)][int(boundary_id)])
                sampled_initials = rng.choice(
                    initial_ids, size=len(initial_ids), replace=True
                )
                for initial_id in sampled_initials:
                    key = (int(seed_id), int(boundary_id), int(initial_id))
                    differences.append(reference[key] - candidate[key])
                    baselines.append(reference[key])
        estimates[replicate] = np.mean(differences) / max(np.mean(baselines), 1e-15)
    point = (
        np.mean([reference[key] - candidate[key] for key in keys])
        / max(np.mean([reference[key] for key in keys]), 1e-15)
    )
    per_seed_wins = {}
    for seed_id in seed_ids:
        selected = [key for key in keys if key[0] == int(seed_id)]
        per_seed_wins[str(int(seed_id))] = bool(
            np.mean([candidate[key] for key in selected])
            < np.mean([reference[key] for key in selected])
        )
    return {
        "method": method,
        "baseline": baseline,
        "metric": metric,
        "paired_units": len(keys),
        "replicates": replicates,
        "relative_improvement": float(point),
        "ci95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "seed_wins": per_seed_wins,
        "num_seed_wins": int(sum(per_seed_wins.values())),
    }


def paired_ratio_bootstrap(
    records: list[dict[str, Any]],
    *,
    method: str,
    baseline: str,
    metric: str = "relative_l2",
    replicates: int = 50_000,
    seed: int = 20260813,
) -> dict[str, Any]:
    """Paired hierarchical bootstrap for method/baseline mean error ratio."""
    candidate = _index(records, method, metric)
    reference = _index(records, baseline, metric)
    if set(candidate) != set(reference):
        raise ValueError("Methods do not contain identical paired statistical units.")
    hierarchy: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for seed_id, boundary_id, initial_id in candidate:
        hierarchy[seed_id][boundary_id].append(initial_id)
    seed_ids = np.asarray(sorted(hierarchy))
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        method_values, baseline_values = [], []
        for seed_id in rng.choice(seed_ids, size=len(seed_ids), replace=True):
            boundary_ids = np.asarray(sorted(hierarchy[int(seed_id)]))
            for boundary_id in rng.choice(boundary_ids, size=len(boundary_ids), replace=True):
                initials = np.asarray(hierarchy[int(seed_id)][int(boundary_id)])
                for initial_id in rng.choice(initials, size=len(initials), replace=True):
                    key = (int(seed_id), int(boundary_id), int(initial_id))
                    method_values.append(candidate[key]); baseline_values.append(reference[key])
        estimates[replicate] = np.mean(method_values) / max(np.mean(baseline_values), 1e-15)
    point = np.mean(list(candidate.values())) / max(np.mean(list(reference.values())), 1e-15)
    return {
        "method": method, "baseline": baseline, "metric": metric,
        "paired_units": len(candidate), "replicates": replicates,
        "ratio": float(point),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
    }


def paired_equivalence_test(
    records: list[dict[str, Any]],
    *,
    method: str,
    baseline: str,
    metric: str = "relative_l2",
    noninferiority_margin: float = 1.05,
    equivalence_lower: float = 0.95,
    replicates: int = 50_000,
    seed: int = 20260820,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired hierarchical ratio bootstrap with non-inferiority and TOST gates.

    Errors are lower-is-better, so the primary null is that the method/oracle
    error ratio is at least ``noninferiority_margin``.  The optional TOST
    interval is deliberately symmetric around one and is reported separately:
    it must not be confused with the one-sided non-inferiority claim.
    """
    if not 1.0 < noninferiority_margin:
        raise ValueError("The non-inferiority upper margin must exceed one.")
    if not 0.0 < equivalence_lower < 1.0 < noninferiority_margin:
        raise ValueError("TOST bounds must straddle one.")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5).")

    candidate = _index(records, method, metric)
    reference = _index(records, baseline, metric)
    if set(candidate) != set(reference):
        raise ValueError("Methods do not contain identical paired statistical units.")
    hierarchy: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for seed_id, boundary_id, initial_id in candidate:
        hierarchy[seed_id][boundary_id].append(initial_id)
    seed_ids = np.asarray(sorted(hierarchy))
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        method_values: list[float] = []
        baseline_values: list[float] = []
        for seed_id in rng.choice(seed_ids, size=len(seed_ids), replace=True):
            boundary_ids = np.asarray(sorted(hierarchy[int(seed_id)]))
            for boundary_id in rng.choice(boundary_ids, size=len(boundary_ids), replace=True):
                initial_ids = np.asarray(hierarchy[int(seed_id)][int(boundary_id)])
                for initial_id in rng.choice(initial_ids, size=len(initial_ids), replace=True):
                    key = (int(seed_id), int(boundary_id), int(initial_id))
                    method_values.append(candidate[key])
                    baseline_values.append(reference[key])
        estimates[replicate] = np.mean(method_values) / max(np.mean(baseline_values), 1e-15)

    point = np.mean(list(candidate.values())) / max(np.mean(list(reference.values())), 1e-15)
    confidence = 1.0 - alpha
    one_sided_upper = float(np.quantile(estimates, confidence))
    one_sided_lower = float(np.quantile(estimates, alpha))
    two_sided = [
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    ]
    # Empirical tail probabilities are descriptive bootstrap p-values.  The
    # confidence-bound decisions below are the frozen experiment gates.
    p_upper = float((np.count_nonzero(estimates >= noninferiority_margin) + 1) / (replicates + 1))
    p_lower = float((np.count_nonzero(estimates <= equivalence_lower) + 1) / (replicates + 1))
    noninferior = one_sided_upper <= noninferiority_margin
    tost_equivalent = one_sided_lower >= equivalence_lower and one_sided_upper <= noninferiority_margin
    return {
        "schema": "blrpng_paired_equivalence_v1",
        "method": method,
        "baseline": baseline,
        "metric": metric,
        "paired_units": len(candidate),
        "replicates": replicates,
        "alpha": alpha,
        "ratio": float(point),
        "two_sided_ci": two_sided,
        "one_sided_ci": [one_sided_lower, one_sided_upper],
        "noninferiority": {
            "upper_margin": noninferiority_margin,
            "upper_confidence_bound": one_sided_upper,
            "bootstrap_tail_probability": p_upper,
            "passed": bool(noninferior),
        },
        "tost_equivalence": {
            "lower_margin": equivalence_lower,
            "upper_margin": noninferiority_margin,
            "lower_confidence_bound": one_sided_lower,
            "upper_confidence_bound": one_sided_upper,
            "lower_bootstrap_tail_probability": p_lower,
            "upper_bootstrap_tail_probability": p_upper,
            "passed": bool(tost_equivalent),
        },
    }


def select_oracle_plateau(
    errors: dict[int, float], *, improvement_threshold: float = 0.01
) -> dict[str, Any]:
    budgets = sorted(int(value) for value in errors)
    if not budgets:
        raise ValueError("Oracle learning curve is empty.")
    selected = budgets[-1]
    improvements = {}
    for previous, current in zip(budgets, budgets[1:]):
        improvement = (float(errors[previous]) - float(errors[current])) / max(float(errors[previous]), 1e-15)
        improvements[f"{previous}_to_{current}"] = improvement
        if improvement < improvement_threshold:
            selected = previous
            break
    return {"selected_budget": selected, "errors": errors, "relative_improvements": improvements}
