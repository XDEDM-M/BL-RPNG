from __future__ import annotations

import numpy as np


RESIDUAL_FLOOR = 1e-30
LOG_DENOMINATOR_FLOOR = 1e-12
NONZERO_STEP_NORM_THRESHOLD = 1e-13


def replay_progress_selection(
    residuals: list[float] | np.ndarray,
    step_norms: list[float] | np.ndarray,
    threshold: float,
    *,
    residual_floor: float = RESIDUAL_FLOOR,
    log_denominator_floor: float = LOG_DENOMINATOR_FLOOR,
    nonzero_step_norm_threshold: float = NONZERO_STEP_NORM_THRESHOLD,
) -> tuple[int, float, bool]:
    """Replay the progress selector from a compact recorded-trace audit.

    ``step_norms[k]`` is the norm between raw recorded trace entries ``k-1``
    and ``k``; entry zero is ignored.  Raw traces may contain repeated states
    (notably a repeated natural initial guess in SfePy), so only nonzero steps
    are candidates.  Residual padding preserves the historical selector's
    behavior for callers that supplied fewer residuals than states.
    """
    norms = np.asarray(step_norms, dtype=np.float64).reshape(-1)
    if norms.size < 2:
        return -1, 0.0, False
    residual = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if residual.size < norms.size:
        # Keep the legacy edge-padding behavior of select_progress_update().
        residual = np.pad(residual, (0, norms.size - residual.size), mode="edge")
    residual = np.maximum(residual[: norms.size], float(residual_floor))
    denominator = max(
        float(np.log(residual[0]) - np.log(residual[-1])),
        float(log_denominator_floor),
    )
    progress = np.clip(
        (np.log(residual[0]) - np.log(residual)) / denominator,
        0.0,
        1.0,
    )
    candidates = [
        index
        for index in range(1, norms.size)
        if norms[index] > float(nonzero_step_norm_threshold)
    ]
    if not candidates:
        return -1, 0.0, False
    selected = next(
        (index for index in candidates if progress[index] >= threshold),
        candidates[-1],
    )
    return selected, float(progress[selected]), True


def select_progress_update(
    states: list[np.ndarray],
    residuals: list[float],
    threshold: float,
    *,
    return_index: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, bool] | tuple[np.ndarray, np.ndarray, float, bool, int]:
    """Select the first Newton path state crossing log-residual progress.

    The returned correction is the cumulative displacement from the initial
    Newton state to the selected state.  A neural probe conditioned only on
    the time-step input cannot identify a local update based on an unobserved
    intermediate iterate; the cumulative path displacement is conditionally
    well-defined and is the appropriate representation target.
    """
    if len(states) < 2:
        result = (states[0], np.zeros_like(states[0]), 0.0, False)
        return (*result, -1) if return_index else result
    step_norms = np.asarray(
        [0.0] + [
            float(np.linalg.norm(states[index] - states[index - 1]))
            for index in range(1, len(states))
        ],
        dtype=np.float64,
    )
    selected, selected_progress, valid = replay_progress_selection(
        residuals, step_norms, threshold
    )
    if not valid:
        result = (states[0], np.zeros_like(states[0]), 0.0, False)
        return (*result, -1) if return_index else result
    result = (
        states[0],
        states[selected] - states[0],
        selected_progress,
        True,
    )
    return (*result, selected) if return_index else result
