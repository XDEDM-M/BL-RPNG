from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from blrpng.data.rollout_windows import RolloutWindowDataset
from blrpng.data.shards import INDEX_SCHEMA, SHARD_SCHEMA, RolloutShardDataset
from blrpng.paths import PROJECT_ROOT, sha256_file, sha256_json


class Phase1RolloutWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_root = PROJECT_ROOT / "artifacts" / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            dir=temporary_root, prefix="rollout_windows_"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _payload(*, steps: int, rollout_id: int, offset: float = 0.0) -> dict:
        resolution = 5
        trajectory = torch.zeros(steps + 1, 1, resolution, resolution)
        for step in range(steps + 1):
            trajectory[step, :, 1:-1, 1:-1] = offset + float(step)
        zeros = torch.zeros(1, resolution, resolution)
        return {
            "schema": SHARD_SCHEMA,
            "trajectory": trajectory,
            "boundary_values": zeros.clone(),
            "boundary_lift": zeros.clone(),
            "constraint_mask": torch.ones_like(zeros, dtype=torch.bool),
            "newton_correction": torch.zeros(steps, 1, resolution, resolution),
            "residual_progress": torch.zeros(steps),
            "valid_correction_mask": torch.ones(steps, 1, dtype=torch.bool),
            "ids": {
                "case_id": rollout_id,
                "pair_id": rollout_id,
                "boundary_id": 7,
                "initial_condition_id": 1000 + rollout_id,
                "rollout_id": rollout_id,
                "target_task_id": 3,
                "support_seed": 19,
                "support_rank": rollout_id,
            },
            "meta": {
                "fidelity": 0,
                "solver_converged": True,
                "nonlinear_iterations": [2] * steps,
            },
        }

    def _dataset(self, payloads: list[dict]) -> RolloutShardDataset:
        rows = []
        for index, payload in enumerate(payloads):
            path = self.root / f"rollout_{index:02d}.pt"
            torch.save(payload, path)
            rows.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                    "steps": int(payload["trajectory"].shape[0] - 1),
                    "resolution": int(payload["trajectory"].shape[-1]),
                    "boundary_id": int(payload["ids"]["boundary_id"]),
                    "initial_condition_id": int(payload["ids"]["initial_condition_id"]),
                    "target_task_id": int(payload["ids"]["target_task_id"]),
                }
            )
        unsigned = {"schema": INDEX_SCHEMA, "metadata": {"role": "unit_test"}, "shards": rows}
        index = {**unsigned, "sha256": sha256_json(unsigned)}
        index_path = self.root / "rollouts.index.json"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        return RolloutShardDataset(index_path)

    def test_horizon_one_preserves_every_original_field_and_order(self):
        base = self._dataset([
            self._payload(steps=4, rollout_id=10),
            self._payload(steps=3, rollout_id=11, offset=100.0),
        ])
        windows = RolloutWindowDataset(base, horizon=1)
        self.assertEqual(len(windows), len(base))
        self.assertEqual(windows.samples, base.samples)
        for index in range(len(base)):
            original = base[index]
            window = windows[index]
            self.assertEqual(set(window), set(original) | {"future_targets"})
            for key, value in original.items():
                self.assertTrue(torch.equal(window[key], value), key)
            self.assertEqual(tuple(window["future_targets"].shape), (1, 1, 5, 5))
            self.assertTrue(torch.equal(window["future_targets"][0], original["y"]))

    def test_horizon_three_keeps_only_same_rollout_valid_starts(self):
        base = self._dataset([
            self._payload(steps=4, rollout_id=10),
            self._payload(steps=3, rollout_id=11, offset=100.0),
        ])
        windows = RolloutWindowDataset(base, horizon=3)
        self.assertEqual(windows.samples, [(0, 0), (0, 1), (1, 0)])
        self.assertEqual(len(windows), 3)
        first, second, third = windows[0], windows[1], windows[2]
        self.assertEqual(int(first["rollout_id"]), 10)
        self.assertEqual(int(second["rollout_id"]), 10)
        self.assertEqual(int(third["rollout_id"]), 11)
        self.assertEqual(int(first["local_step_id"]), 0)
        self.assertEqual(int(second["local_step_id"]), 1)
        self.assertEqual(int(third["local_step_id"]), 0)
        self.assertTrue(torch.equal(first["future_targets"][0], first["y"]))
        self.assertEqual(
            [float(value[0, 2, 2]) for value in first["future_targets"]],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            [float(value[0, 2, 2]) for value in third["future_targets"]],
            [101.0, 102.0, 103.0],
        )
        for key in ("boundary_lift", "boundary_mask", "boundary_id", "support_seed"):
            self.assertIn(key, first)

    def test_dataloader_collates_future_targets_as_batch_horizon_channels(self):
        windows = RolloutWindowDataset(
            self._dataset([self._payload(steps=4, rollout_id=10)]), horizon=3
        )
        batch = next(iter(DataLoader(windows, batch_size=2, shuffle=False)))
        self.assertEqual(tuple(batch["x"].shape), (2, 3, 5, 5))
        self.assertEqual(tuple(batch["y"].shape), (2, 1, 5, 5))
        self.assertEqual(tuple(batch["future_targets"].shape), (2, 3, 1, 5, 5))

    def test_time_dependent_conditions_fail_explicitly_for_multistep(self):
        payload = self._payload(steps=4, rollout_id=10)
        payload["condition_fields"] = torch.zeros(4, 2, 5, 5)
        windows = RolloutWindowDataset(self._dataset([payload]), horizon=2)
        with self.assertRaisesRegex(NotImplementedError, "time-dependent condition_fields"):
            _ = windows[0]

    def test_invalid_or_unavailable_horizon_fails_closed(self):
        base = self._dataset([self._payload(steps=2, rollout_id=10)])
        for invalid in (0, -1, 1.5, True):
            with self.subTest(horizon=invalid):
                with self.assertRaises(ValueError):
                    RolloutWindowDataset(base, horizon=invalid)
        with self.assertRaisesRegex(ValueError, "no rollout contains"):
            RolloutWindowDataset(base, horizon=3)


if __name__ == "__main__":
    unittest.main()
