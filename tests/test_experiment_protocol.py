from __future__ import annotations

import unittest
from unittest.mock import patch

from blrpng.analysis.bootstrap import paired_equivalence_test, paired_hierarchical_bootstrap
from blrpng.config import load_config
from blrpng.data.generate import _case_chunks, generate_from_config
from blrpng.cli.verify_data import _formal_paths
from blrpng.paths import exclusive_artifact_lock
from blrpng.training.matrix import matrix_payload
from scripts.summarize_phase1_progress import terminal_gpu_hours


class ExperimentProtocolTests(unittest.TestCase):
    def test_progress_accounting_counts_duplicate_hrdev_executions(self):
        rows = [
            {"trial_id": "task00", "status": "DONE", "round": "R2", "elapsed_gpu_hours": "0.1"},
            {"trial_id": "task00", "status": "EXCLUDED_RESOURCE_CONTENTION", "round": "R2", "elapsed_gpu_hours": "0.1"},
            {"trial_id": "eval-task00", "status": "DONE", "round": "HRDEV", "elapsed_gpu_hours": "0.001"},
            {"trial_id": "eval-task00", "status": "DONE", "round": "HRDEV", "elapsed_gpu_hours": "0.002"},
        ]
        self.assertAlmostEqual(terminal_gpu_hours(rows), 0.103)

    def test_formal_matrix_has_120_matched_tasks(self):
        payload = matrix_payload(load_config("configs/allen_cahn_formal.yaml"))
        self.assertEqual(payload["causal_task_count"], 120)
        groups = {}
        for task in payload["causal_tasks"]:
            key = (task["seed"], task["boundary_budget"])
            groups.setdefault(key, set()).add(task["data_order_group"])
        self.assertTrue(all(len(values) == 1 for values in groups.values()))

    def test_formal_generation_uses_new_training_step_key(self):
        config = load_config("configs/allen_cahn_formal.yaml")
        observed_steps = []

        def fake_worker(**kwargs):
            observed_steps.append(kwargs["steps"])
            return kwargs["output_path"]

        with patch("blrpng.data.generate.create_manifest") as create, patch(
            "blrpng.data.generate._run_worker", side_effect=fake_worker
        ):
            create.return_value = "artifacts/manifests/allen_cahn_formal.json"
            generate_from_config(config, stage="a")
        self.assertEqual(observed_steps, [10, 10])

    def test_formal_verifier_covers_all_frozen_splits(self):
        paths = _formal_paths("allen_cahn_formal")
        self.assertEqual(len(paths), 10)
        self.assertIn("b_train_coarse", paths)
        self.assertIn("blind_unseen", paths)

    def test_parallel_case_chunks_are_contiguous_and_complete(self):
        chunks = _case_chunks(10, 3)
        self.assertEqual(chunks, [(0, 4), (4, 3), (7, 3)])
        covered = [index for start, count in chunks for index in range(start, start + count)]
        self.assertEqual(covered, list(range(10)))

    def test_experiment_lock_rejects_duplicate_writer(self):
        with exclusive_artifact_lock("test_duplicate_writer"):
            with self.assertRaisesRegex(RuntimeError, "already locked"):
                with exclusive_artifact_lock("test_duplicate_writer"):
                    pass

    def test_paired_hierarchical_bootstrap(self):
        records = []
        for seed in (0, 1):
            for boundary in (10, 11):
                for initial in (20, 21):
                    common = {
                        "seed": seed,
                        "boundary_id": boundary,
                        "initial_condition_id": initial,
                    }
                    records.append(
                        {**common, "method": "baseline", "relative_l2": 1.0}
                    )
                    records.append(
                        {**common, "method": "method", "relative_l2": 0.8}
                    )
        report = paired_hierarchical_bootstrap(
            records,
            method="method",
            baseline="baseline",
            replicates=500,
            seed=7,
        )
        self.assertAlmostEqual(report["relative_improvement"], 0.2)
        self.assertGreater(report["ci95"][0], 0.0)
        self.assertEqual(report["num_seed_wins"], 2)

    def test_paired_noninferiority_and_tost(self):
        records = []
        for seed in (0, 1):
            for boundary in (10, 11):
                for initial in (20, 21, 22):
                    common = {
                        "seed": seed,
                        "boundary_id": boundary,
                        "initial_condition_id": initial,
                    }
                    records.append({**common, "method": "oracle", "relative_l2": 1.0})
                    records.append({**common, "method": "candidate", "relative_l2": 1.01})
        report = paired_equivalence_test(
            records,
            method="candidate",
            baseline="oracle",
            replicates=500,
            seed=9,
        )
        self.assertAlmostEqual(report["ratio"], 1.01)
        self.assertTrue(report["noninferiority"]["passed"])
        self.assertTrue(report["tost_equivalence"]["passed"])


if __name__ == "__main__":
    unittest.main()
