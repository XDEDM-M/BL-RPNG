from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from blrpng.analysis.ns_grid_projection import grid_divergence, project_no_slip_divergence
from blrpng.config import load_config
from blrpng.data.navier_stokes_protocol import (
    build_navier_stokes_protocol,
    validate_navier_stokes_protocol,
)
from blrpng.data.navier_stokes_sfepy import (
    _quarantine_empty_shard,
    run_sfepy_navier_stokes_case,
)
from blrpng.data.shards import validate_shard
from blrpng.models.fno import DirichletOperator, FNOConfig
from blrpng.paths import ensure_artifact_path
from blrpng.training.physics import navier_stokes_defect


class SfePyNavierStokesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/navier_stokes_sfepy_seed42.yaml")

    def test_interrupted_empty_shard_is_quarantined_without_touching_nonempty(self):
        temporary_root = ensure_artifact_path("artifacts/tmp")
        with tempfile.TemporaryDirectory(dir=temporary_root, prefix="ns_quarantine_test_") as directory:
            empty = Path(directory) / "interrupted.pt"
            empty.touch()
            quarantined = _quarantine_empty_shard(empty)
            self.assertIsNotNone(quarantined)
            self.assertFalse(empty.exists())
            self.assertEqual(quarantined.stat().st_size, 0)
            manifest = quarantined.with_suffix(quarantined.suffix + ".json")
            self.assertEqual(json.loads(manifest.read_text())["reason"], "zero_byte_shard_after_interrupted_process")
            nonempty = Path(directory) / "valid.pt"
            nonempty.write_bytes(b"existing user data")
            self.assertIsNone(_quarantine_empty_shard(nonempty))
            self.assertEqual(nonempty.read_bytes(), b"existing user data")
            manifest.unlink()
            quarantined.unlink()

    def test_grid_projection_preserves_wall_velocity_and_removes_divergence(self):
        axis = np.linspace(0.0, 1.0, 32)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        velocity = np.stack((np.sin(np.pi * xx) * np.sin(np.pi * yy), np.zeros_like(xx)))
        before = velocity.copy()
        projected, diagnostic = project_no_slip_divergence(velocity)
        self.assertGreater(diagnostic["divergence_rms_before"], 0.1)
        self.assertLess(diagnostic["divergence_rms_after"], 1e-6)
        self.assertLess(float(np.sqrt(np.mean(grid_divergence(projected) ** 2))), 1e-6)
        self.assertTrue(np.array_equal(projected[:, 0, :], before[:, 0, :]))
        self.assertTrue(np.array_equal(projected[:, -1, :], before[:, -1, :]))
        self.assertTrue(np.array_equal(projected[:, :, 0], before[:, :, 0]))
        self.assertTrue(np.array_equal(projected[:, :, -1], before[:, :, -1]))

    def test_steady_continuation_has_real_early_newton_target(self):
        executable = Path(self.config["resources"]["sfepy_python"])
        if not executable.exists():
            self.skipTest("The configured isolated SfePy environment is unavailable.")
        config = copy.deepcopy(self.config)
        config["solver"]["path_selection_mode"] = "first_accepted"
        config["resources"]["sfepy_case_timeout_seconds"] = 60.0
        case = copy.deepcopy(build_navier_stokes_protocol(config)["targets"]["0"]["support"][0])
        case.update({
            "case_id": 9_900_210,
            "pair_id": 9_900_210,
            "rollout_id": 9_900_210,
            "role": "automated_steady_test",
            "reynolds_schedule": [50, 100, 200],
            "newton_predictor_blend": 0.5,
            "grad_div_gamma": 100.0,
            "forcing": {"active": False},
        })
        path = "artifacts/data/navier_stokes_sfepy_test/steady_rollout_9900210.pt"
        run_sfepy_navier_stokes_case(
            case,
            output_shard=path,
            output_resolution=16,
            mesh_resolution=9,
            steps=3,
            dt=1.0,
            config=config,
            fidelity=0,
            time_integrator="STEADY_CONTINUATION",
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_shard(payload)
        self.assertEqual(payload["meta"]["trajectory_axis"], "reynolds_continuation_not_physical_time")
        self.assertEqual(payload["meta"]["condition_names"][1], "continuation_progress")
        self.assertTrue(payload["meta"]["solver_converged"])
        self.assertEqual(tuple(payload["predictor_endpoint_increment"].shape), (3, 3, 16, 16))
        self.assertGreater(payload["meta"]["selected_velocity_predictor_endpoint_discrepancy"][2], 0.1)
        self.assertLess(max(payload["meta"]["relative_residuals"]), 1e-8)

    def test_protocol_is_frozen_nested_and_leak_free(self):
        first = build_navier_stokes_protocol(copy.deepcopy(self.config))
        second = build_navier_stokes_protocol(copy.deepcopy(self.config))
        self.assertEqual(first["sha256"], second["sha256"])
        result = validate_navier_stokes_protocol(first)
        self.assertTrue(result["passed"])
        self.assertEqual(result["target_tasks"], 24)
        for task_id, task in first["targets"].items():
            self.assertEqual(len(task["support"]), 8)
            self.assertEqual(len(task["validation"]), 4)
            self.assertEqual(len(task["query"]), 8)
            expected_extra = 24 if int(task_id) in first["oracle_task_ids"] else 0
            self.assertEqual(len(task["oracle_extra"]), expected_extra)

    def test_registered_model_sizes_and_multiresolution_forward(self):
        expected = {
            "small": (4, 32, 12, 1_187_247),
            "medium": (6, 32, 16, 3_155_765),
            "large": (6, 48, 16, 7_099_077),
        }
        for _name, (layers, width, modes, parameters) in expected.items():
            model = DirichletOperator(
                FNOConfig(
                    in_channels=12,
                    out_channels=3,
                    condition_channels=4,
                    layers=layers,
                    width=width,
                    modes1=modes,
                    modes2=modes,
                ),
                field_scales=(1.0, 1.0, 0.25),
                gauge_channels=(2,),
            )
            self.assertEqual(sum(value.numel() for value in model.parameters()), parameters)
            # One lightweight candidate is sufficient to test the common operator path.
            if width == 32 and layers == 4:
                for resolution in (64, 128):
                    state = torch.zeros(1, 3, resolution, resolution)
                    lift = torch.zeros_like(state)
                    mask = torch.zeros_like(state, dtype=torch.bool)
                    mask[:, :2, (0, -1), :] = True
                    conditions = torch.zeros(1, 4, resolution, resolution)
                    axis = torch.linspace(0.0, 1.0, resolution)
                    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
                    raw_input = torch.cat(
                        (state, torch.stack((xx, yy))[None], conditions), dim=1
                    )
                    prediction, probes = model(raw_input, lift, mask)
                    self.assertEqual(tuple(prediction.shape), tuple(state.shape))
                    self.assertIn("layer1:hook0", probes)
                    self.assertLess(float(prediction[:, 2].mean().abs().detach()), 1e-6)

    def test_defect_uses_dt_viscosity_and_body_force_channels(self):
        previous = torch.zeros(2, 3, 12, 12, dtype=torch.float64)
        current = torch.zeros_like(previous)
        current[:, 0] = 0.1
        conditions = torch.zeros(2, 4, 12, 12, dtype=torch.float64)
        conditions[:, 0] = 0.01
        conditions[0, 1] = 0.01
        conditions[1, 1] = 0.02
        conditions[:, 2] = 0.25
        defect = navier_stokes_defect(
            previous, current, {"dt": 99.0, "condition_fields": conditions}
        )
        self.assertFalse(torch.allclose(defect[0], defect[1]))
        self.assertTrue(torch.allclose(defect[0, 0], torch.full_like(defect[0, 0], 9.75)))
        self.assertTrue(torch.allclose(defect[:, 1], torch.zeros_like(defect[:, 1])))

    def test_sfepy_worker_bridge_smoke_and_accepted_state_semantics(self):
        executable = Path(self.config["resources"]["sfepy_python"])
        if not executable.exists():
            self.skipTest("The configured isolated SfePy environment is unavailable.")
        protocol = build_navier_stokes_protocol(copy.deepcopy(self.config))
        case = copy.deepcopy(protocol["development_train"][0])
        case.update(
            {
                "case_id": 9_900_001,
                "pair_id": 9_900_001,
                "rollout_id": 9_900_001,
                "role": "automated_test",
                "reynolds": 100.0,
            }
        )
        path = "artifacts/data/navier_stokes_sfepy_test/rollout_9900001.pt"
        run_sfepy_navier_stokes_case(
            case,
            output_shard=path,
            output_resolution=16,
            mesh_resolution=9,
            steps=1,
            dt=0.01,
            config=self.config,
            fidelity=0,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validation = validate_shard(payload)
        self.assertLess(validation["trace_error_max"], 1e-6)
        self.assertTrue(payload["meta"]["solver_converged"])
        self.assertLess(payload["meta"]["pressure_gauge_error"], 1e-7)
        self.assertLess(max(payload["meta"]["weak_continuity"]), 1e-6)
        count = int(payload["accepted_newton_counts"][0])
        residuals = payload["accepted_newton_residuals"][0, :count]
        self.assertGreaterEqual(count, 2)
        self.assertTrue(bool(torch.all(residuals[1:] <= residuals[:-1])))
        self.assertEqual(tuple(payload["endpoint_increment"].shape), (1, 3, 16, 16))


if __name__ == "__main__":
    unittest.main()
