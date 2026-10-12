from __future__ import annotations

import numpy as np


def select_progress_update(
    states: list[np.ndarray],
    residuals: list[float],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Select the first Newton path state crossing log-residual progress.

    The returned correction is the cumulative displacement from the initial
    Newton state to the selected state.  A neural probe conditioned only on
    the time-step input cannot identify a local update based on an unobserved
    intermediate iterate; the cumulative path displacement is conditionally
    well-defined and is the appropriate representation target.
    """
    if len(states) < 2:
        return states[0], np.zeros_like(states[0]), 0.0, False
    residual = np.asarray(residuals, dtype=np.float64)
    if residual.size < len(states):
        residual = np.pad(residual, (0, len(states) - residual.size), mode="edge")
    residual = np.maximum(residual[: len(states)], 1e-30)
    denominator = max(
        float(np.log(residual[0]) - np.log(residual[-1])),
        1e-12,
    )
    progress = np.clip(
        (np.log(residual[0]) - np.log(residual)) / denominator,
        0.0,
        1.0,
    )
    candidates = [
        index
        for index in range(1, len(states))
        if np.linalg.norm(states[index] - states[index - 1]) > 1e-13
    ]
    if not candidates:
        return states[0], np.zeros_like(states[0]), 0.0, False
    selected = next(
        (index for index in candidates if progress[index] >= threshold),
        candidates[-1],
    )
    return (
        states[0],
        states[selected] - states[0],
        float(progress[selected]),
        True,
    )
