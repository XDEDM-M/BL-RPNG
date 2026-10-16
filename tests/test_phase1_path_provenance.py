from __future__ import annotations

import unittest

import numpy as np
import torch

from blrpng.data.shards import RolloutShardDataset, SHARD_SCHEMA, validate_shard
from blrpng.solvers.progress import (
    LOG_DENOMINATOR_FLOOR,
    NONZERO_STEP_NORM_THRESHOLD,
    RESIDUAL_FLOOR,
    replay_progress_selection,
    select_progress_update,
)


def _evidence_payload() -> dict:
    steps, fields, size = 2, 1, 5
    residuals = torch.tensor(
        [[10.0, 10.0, 1.0, 0.01, 0.0, 0.0],
         [10.0, 10.0, 9.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    step_norms = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    counts = torch.tensor([4, 4], dtype=torch.int16)
    indices = {0.0: [], 0.25: [], 0.75: []}
    progresses = {0.0: [], 0.25: [], 0.75: []}
    validity = {0.0: [], 0.25: [], 0.75: []}
    for threshold in indices:
        for step in range(steps):
            selected, progress, valid = replay_progress_selection(
                residuals[step, : int(counts[step])].numpy(),
                step_norms[step, : int(counts[step])].numpy(),
                threshold,
            )
            indices[threshold].append(selected)
            progresses[threshold].append([progress])
            validity[threshold].append([valid])
    mask = torch.zeros((fields, size, size), dtype=torch.bool)
    mask[:, (0, -1), :] = True
    mask[:, :, (0, -1)] = True
    return {
        "schema": SHARD_SCHEMA,
        "trajectory": torch.zeros((steps + 1, fields, size, size)),
        "boundary_values": torch.zeros((fields, size, size)),
        "boundary_lift": torch.zeros((fields, size, size)),
        "constraint_mask": mask,
        "newton_correction": torch.zeros((steps, fields, size, size)),
        "newton_correction_first": torch.zeros((steps, fields, size, size)),
        "newton_correction_p075": torch.zeros((steps, fields, size, size)),
        "residual_progress": torch.tensor(progresses[0.25], dtype=torch.float32),
        "residual_progress_first": torch.tensor(progresses[0.0], dtype=torch.float32),
        "residual_progress_p075": torch.tensor(progresses[0.75], dtype=torch.float32),
        "valid_correction_mask": torch.tensor(validity[0.25], dtype=torch.bool),
        "valid_correction_mask_first": torch.tensor(validity[0.0], dtype=torch.bool),
        "valid_correction_mask_p075": torch.tensor(validity[0.75], dtype=torch.bool),
        "selected_newton_state_index": torch.tensor(indices[0.25], dtype=torch.int16),
        "selected_newton_state_index_first": torch.tensor(indices[0.0], dtype=torch.int16),
        "selected_newton_state_index_p075": torch.tensor(indices[0.75], dtype=torch.int16),
        "accepted_newton_residuals": residuals,
        "accepted_newton_step_norms": step_norms,
        "accepted_newton_counts": counts,
        "ids": {
            "case_id": 1,
            "pair_id": 1,
            "boundary_id": 1,
            "initial_condition_id": 1,
            "rollout_id": 1,
            "target_task_id": 0,
        },
        "meta": {
            "equation": "allen_cahn",
            "role": "support",
            "resolution": size,
            "fidelity": 0,
            "nonlinear_iterations": [3, 3],
            "newton_path_selection": {
                "schema": "blrpng_newton_selection_evidence_v1",
                "thresholds": [0.25, 0.75],
                "first_selection_threshold": 0.0,
                "residual_floor": RESIDUAL_FLOOR,
                "log_denominator_floor": LOG_DENOMINATOR_FLOOR,
                "nonzero_step_norm_threshold": NONZERO_STEP_NORM_THRESHOLD,
                "fallback": "last_nonzero_candidate",
                "q0_semantics": "natural_previous_physical_state",
                "trace_count_includes_q0_and_appended_final": True,
                "trace_may_contain_repeated_q0": True,
                "full_node_states_retained": False,
                "evidence_scope": "selection_replay_not_field_trajectory_reconstruction",
            },
        },
    }


class Phase1PathProvenanceTests(unittest.TestCase):
    def test_replay_preserves_selector_and_residual_padding(self) -> None:
        states = [
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([1.5, 0.5]),
        ]
        residuals = [10.0, 10.0, 1.0]
        step_norms = [
            0.0,
            0.0,
            float(np.linalg.norm(states[2] - states[1])),
            float(np.linalg.norm(states[3] - states[2])),
        ]
        selected, progress, valid = replay_progress_selection(
            residuals, step_norms, 0.25
        )
        self.assertEqual((selected, valid), (2, True))
        old_signature = select_progress_update(states, residuals, 0.25)
        indexed = select_progress_update(states, residuals, 0.25, return_index=True)
        self.assertEqual(len(old_signature), 4)
        self.assertEqual(indexed[4], selected)
        self.assertEqual(indexed[2], progress)
        np.testing.assert_array_equal(indexed[1], states[selected] - states[0])

    def test_replay_fallback_and_no_candidate(self) -> None:
        fallback = replay_progress_selection(
            [10.0, 10.0, 9.0, 1.0], [0.0, 0.0, 1.0, 0.0], 0.75
        )
        self.assertEqual(fallback[0], 2)
        self.assertLess(fallback[1], 0.75)
        self.assertTrue(fallback[2])
        self.assertEqual(
            replay_progress_selection([1.0, 1.0], [0.0, 0.0], 0.25),
            (-1, 0.0, False),
        )

    def test_replay_residual_and_denominator_floors(self) -> None:
        floored = replay_progress_selection([1.0, 0.0], [0.0, 1.0], 0.25)
        self.assertEqual((floored[0], floored[2]), (1, True))
        self.assertEqual(floored[1], 1.0)
        flat = replay_progress_selection([1.0, 1.0], [0.0, 1.0], 0.25)
        self.assertEqual((flat[0], flat[2]), (1, True))
        self.assertEqual(flat[1], 0.0)

    def test_synthetic_shard_replays_first_and_progress_thresholds(self) -> None:
        result = validate_shard(_evidence_payload())
        self.assertEqual(result["newton_selection_replay_pass_count"], 6)
        payload = _evidence_payload()
        self.assertEqual(payload["selected_newton_state_index_first"].tolist(), [2, 2])

    def test_synthetic_shard_rejects_corrupt_evidence(self) -> None:
        mutations = [
            (lambda p: p["accepted_newton_residuals"].__setitem__((0, 2), 9.0), "replay mismatch"),
            (lambda p: p["accepted_newton_step_norms"].__setitem__((0, 2), 0.0), "replay mismatch"),
            (lambda p: p["accepted_newton_counts"].__setitem__(0, 3), "padding"),
            (lambda p: p["accepted_newton_residuals"].__setitem__((0, 5), 1.0), "padding"),
            (lambda p: p["selected_newton_state_index"].__setitem__(0, 3), "replay mismatch"),
            (lambda p: p["selected_newton_state_index_first"].__setitem__(0, 3), "replay mismatch"),
            (lambda p: p["residual_progress"].__setitem__((0, 0), 0.9), "progress replay mismatch"),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                payload = _evidence_payload()
                mutate(payload)
                with self.assertRaisesRegex(ValueError, message):
                    validate_shard(payload)

    def test_evidence_all_or_none_and_old_shard_compatibility(self) -> None:
        partial = _evidence_payload()
        del partial["accepted_newton_counts"]
        with self.assertRaisesRegex(ValueError, "all-or-none"):
            validate_shard(partial)

        legacy_residual_only = _evidence_payload()
        for key in (
            "newton_correction_first",
            "residual_progress_first",
            "valid_correction_mask_first",
            "selected_newton_state_index_first",
        ):
            legacy_residual_only.pop(key)
        legacy_residual_only.pop("accepted_newton_step_norms")
        legacy_residual_only["meta"].pop("newton_path_selection")
        self.assertEqual(
            validate_shard(legacy_residual_only)["newton_selection_replay_pass_count"],
            0,
        )

        historical = _evidence_payload()
        for key in (
            "accepted_newton_residuals",
            "accepted_newton_step_norms",
            "accepted_newton_counts",
            "newton_correction_first",
            "newton_correction_p075",
            "residual_progress_first",
            "residual_progress_p075",
            "valid_correction_mask_first",
            "valid_correction_mask_p075",
            "selected_newton_state_index_first",
        ):
            historical.pop(key)
        historical["meta"].pop("newton_path_selection")
        result = validate_shard(historical)
        self.assertEqual(result["newton_selection_replay_pass_count"], 0)

    def test_first_group_is_optional_but_atomic_and_has_zero_trace(self) -> None:
        without_first = _evidence_payload()
        for key in (
            "newton_correction_first",
            "residual_progress_first",
            "valid_correction_mask_first",
            "selected_newton_state_index_first",
        ):
            without_first.pop(key)
        without_first["meta"]["newton_path_selection"].pop(
            "first_selection_threshold"
        )
        self.assertEqual(
            validate_shard(without_first)["newton_selection_replay_pass_count"], 4
        )

        partial = _evidence_payload()
        partial.pop("residual_progress_first")
        with self.assertRaisesRegex(ValueError, "all-or-none"):
            validate_shard(partial)

        nonzero_trace = _evidence_payload()
        nonzero_trace["newton_correction_first"][0, 0, 0, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "zero boundary trace"):
            validate_shard(nonzero_trace)

    def test_dataset_exposes_first_labels_only_when_stored(self) -> None:
        def sample(payload: dict) -> dict[str, torch.Tensor]:
            dataset = object.__new__(RolloutShardDataset)
            dataset.samples = [(0, 0)]
            dataset._load = lambda _index: payload
            return dataset[0]

        current = sample(_evidence_payload())
        for key in (
            "newton_correction_first",
            "residual_progress_first",
            "valid_correction_mask_first",
            "selected_newton_state_index_first",
        ):
            self.assertIn(key, current)

        historical = _evidence_payload()
        for key in (
            "newton_correction_first",
            "residual_progress_first",
            "valid_correction_mask_first",
            "selected_newton_state_index_first",
        ):
            historical.pop(key)
        historical["meta"]["newton_path_selection"].pop(
            "first_selection_threshold"
        )
        old_sample = sample(historical)
        self.assertNotIn("newton_correction_first", old_sample)
        self.assertNotIn("valid_correction_mask_first", old_sample)


if __name__ == "__main__":
    unittest.main()
