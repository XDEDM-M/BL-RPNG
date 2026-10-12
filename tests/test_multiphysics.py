from __future__ import annotations

import unittest
import copy

import numpy as np
import torch

from blrpng.analysis.stage_gates import poisson_negative_control_gate, solver_gate, storage_preflight
from blrpng.data.capacity import require_prerequisite_gate
from blrpng.config import load_config
from blrpng.data.shards import RolloutShardDataset
from blrpng.evaluation.multiphysics import evaluate_multiphysics_index
from blrpng.solvers.gray_scott import generate_gray_scott_rollout
from blrpng.solvers.hyperelastic import generate_hyperelastic_rollout
from blrpng.solvers.navier_stokes_mac import generate_navier_stokes_rollout
from blrpng.solvers.poisson import generate_poisson_rollout, poisson_boundary
from blrpng.data.poisson_protocol import build_poisson_protocol, validate_poisson_protocol
from blrpng.data.gray_scott_protocol import build_gray_scott_protocol, validate_gray_scott_protocol
from blrpng.training.physics import (
    gray_scott_defect,
    hyperelastic_defect,
    navier_stokes_defect,
    poisson_defect,
)
from blrpng.models.fno import DirichletOperator, FNOConfig
from blrpng.training.engine import _train_batch, train
from blrpng.utils import state_dict_sha256


class MultiphysicsTests(unittest.TestCase):
    def test_poisson_negative_control_dual_grid(self):
        coarse = generate_poisson_rollout(resolution=16, seed=7).payload()
        fine = generate_poisson_rollout(resolution=32, seed=7).payload()
        self.assertTrue(coarse["meta"]["solver_converged"])
        self.assertTrue(fine["meta"]["solver_converged"])
        self.assertEqual(float(coarse["newton_correction"][:, :, (0, -1)].abs().max()), 0.0)
        self.assertTrue(poisson_negative_control_gate(0.02)["passed"])

    def test_poisson_fixed_target_boundary_and_independent_forcing(self):
        first = generate_poisson_rollout(
            resolution=32, seed=1, boundary_seed=17, forcing_seed=31,
            family="fourier", regime="target", target_task_id=2,
        )
        second = generate_poisson_rollout(
            resolution=32, seed=2, boundary_seed=17, forcing_seed=32,
            family="fourier", regime="target", target_task_id=2,
        )
        self.assertTrue(torch.equal(first.boundary_values, second.boundary_values))
        self.assertFalse(torch.equal(first.condition_fields, second.condition_fields))
        source = poisson_boundary(64, 17, "fourier", regime="source")
        target = poisson_boundary(64, 17, "fourier", regime="target")
        self.assertFalse(np.array_equal(source, target))

    def test_poisson_protocol_has_no_split_leakage(self):
        config = copy.deepcopy(load_config("configs/poisson_negative_control.yaml"))
        protocol = build_poisson_protocol(config)
        result = validate_poisson_protocol(protocol)
        self.assertTrue(result["passed"])
        self.assertEqual(result["target_tasks"], 30)

    def test_gray_scott_v4_coupled_boundary(self):
        payload = generate_gray_scott_rollout(
            resolution=16, steps=1, seed=3, correlation=-0.25
        ).payload()
        self.assertEqual(tuple(payload["trajectory"].shape), (2, 2, 16, 16))
        self.assertEqual(tuple(payload["condition_fields"].shape), (3, 16, 16))
        self.assertTrue(payload["meta"]["solver_converged"])

    def test_gray_scott_fixed_boundary_and_protocol_isolation(self):
        first = generate_gray_scott_rollout(
            resolution=16, steps=1, seed=1, boundary_seed=9, initial_seed=10,
            target_task_id=3, boundary_regime="high_frequency",
        )
        second = generate_gray_scott_rollout(
            resolution=16, steps=1, seed=2, boundary_seed=9, initial_seed=11,
            target_task_id=3, boundary_regime="high_frequency",
        )
        self.assertTrue(torch.equal(first.boundary_values, second.boundary_values))
        self.assertFalse(torch.equal(first.trajectory[0], second.trajectory[0]))
        protocol = build_gray_scott_protocol(
            copy.deepcopy(load_config("configs/gray_scott_confirmation.yaml"))
        )
        result = validate_gray_scott_protocol(protocol)
        self.assertTrue(result["passed"])
        self.assertEqual(result["target_tasks"], 16)

    def test_hyperelastic_newton_and_zero_trace(self):
        payload = generate_hyperelastic_rollout(resolution=9, seed=5).payload()
        self.assertTrue(payload["meta"]["solver_converged"])
        mask = payload["constraint_mask"]
        self.assertEqual(float(payload["newton_correction"][:, mask].abs().max()), 0.0)
        self.assertGreater(payload["meta"]["strain_energy"], 0.0)

    def test_mac_newton_krylov_gate(self):
        payload = generate_navier_stokes_rollout(
            resolution=9, steps=1, seed=5, reynolds=100.0
        ).payload()
        self.assertTrue(payload["meta"]["solver_converged"])
        self.assertLessEqual(payload["meta"]["divergence_max"], 1e-6)
        self.assertLessEqual(payload["meta"]["pressure_gauge_error"], 1e-6)
        self.assertTrue(solver_gate(
            convergence_fraction=1.0, valid_correction_fraction=1.0,
            boundary_error=0.0, divergence_error=payload["meta"]["divergence_max"],
        )["passed"])

    def test_multiphysics_defect_jvps(self):
        torch.manual_seed(2)
        for function, channels, params in (
            (hyperelastic_defect, 2, {"mu": 1.0, "lambda": 2.0}),
            (navier_stokes_defect, 3, {"dt": 0.01, "reynolds": 100.0}),
        ):
            previous = 0.01 * torch.randn(1, channels, 12, 12, dtype=torch.float64)
            state = 0.01 * torch.randn(1, channels, 12, 12, dtype=torch.float64)
            direction = torch.randn_like(state)
            _, jvp = torch.autograd.functional.jvp(
                lambda value: function(previous, value, params), state, direction
            )
            epsilon = 1e-6
            finite = (
                function(previous, state + epsilon * direction, params)
                - function(previous, state - epsilon * direction, params)
            ) / (2.0 * epsilon)
            relative = (jvp - finite).norm() / finite.norm().clamp_min(1e-12)
            self.assertLess(float(relative), 2e-5)

    def test_defects_use_per_sample_condition_fields(self):
        previous = torch.zeros(2, 2, 12, 12)
        current = torch.full_like(previous, 0.2)
        conditions = torch.zeros(2, 3, 12, 12)
        conditions[0, 0] = 0.02; conditions[1, 0] = 0.08
        conditions[:, 1] = 0.06
        gray = gray_scott_defect(previous, current, {
            "dt": 1.0, "diffusion_u": 2e-5, "diffusion_v": 1e-5,
            "feed": 0.04, "kill": 0.06, "condition_fields": conditions,
        })
        self.assertFalse(torch.allclose(gray[0], gray[1]))
        forcing = torch.stack((torch.zeros(12, 12), torch.ones(12, 12)))[:, None]
        poisson = poisson_defect(
            torch.zeros(2, 1, 12, 12), torch.zeros(2, 1, 12, 12),
            {"condition_fields": forcing},
        )
        self.assertFalse(torch.allclose(poisson[0], poisson[1]))

    def test_condition_fields_stream_loader(self):
        dataset = RolloutShardDataset("artifacts/data/poisson_smoke.index.json")
        sample = dataset[0]
        self.assertEqual(sample["x"].shape[0], 4)

    def test_frozen_multiphysics_evaluator(self):
        model = DirichletOperator(FNOConfig(
            in_channels=5, out_channels=1, condition_channels=1,
            modes1=4, modes2=4, width=8, layers=2,
        ))
        report = evaluate_multiphysics_index(
            model, "artifacts/data/poisson_smoke.index.json",
            device=torch.device("cpu"), equation="poisson",
            equation_parameters={}, horizon=1,
        )
        self.assertIn("near_boundary_relative_l2", report)
        self.assertEqual(report["field_names"], ["solution"])

    def test_v4_stream_training_epoch(self):
        config = copy.deepcopy(load_config("configs/poisson_negative_control.yaml"))
        config["model"].update({"modes1": 4, "modes2": 4, "width": 8, "layers": 2})
        config["training"].update({
            "epochs": 1, "batch_size": 2, "lr_scheduler": "multistep",
            "lr_drop_epoch": 1, "recovery_interval": 1,
            "adapter": {"strategy": "full", "rank": 2, "alpha": 2.0, "train_last_blocks": 0},
        })
        result = train(
            config, objective="final_only",
            output_checkpoint="artifacts/checkpoints/test_v4_stream.pt",
            train_data_path="artifacts/data/poisson_smoke.index.json",
            validation_data_paths=["artifacts/data/poisson_smoke.index.json"],
        )
        checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
        repeated = train(
            config, objective="final_only",
            output_checkpoint="artifacts/checkpoints/test_v4_stream_repeat.pt",
            train_data_path="artifacts/data/poisson_smoke.index.json",
            validation_data_paths=["artifacts/data/poisson_smoke.index.json"],
        )
        repeated_checkpoint = torch.load(
            repeated["checkpoint"], map_location="cpu", weights_only=False
        )
        self.assertEqual(len(checkpoint["history"]), 1)
        self.assertEqual(
            state_dict_sha256(checkpoint["raw_weights"]),
            state_dict_sha256(repeated_checkpoint["raw_weights"]),
        )
        self.assertEqual(checkpoint["manifest_sha256"], RolloutShardDataset(
            "artifacts/data/poisson_smoke.index.json"
        ).index["sha256"])

    def test_storage_preflight(self):
        self.assertTrue(storage_preflight(
            current_bytes=100 * 2**30, estimated_new_bytes=50 * 2**30
        )["passed"])
        self.assertFalse(storage_preflight(
            current_bytes=290 * 2**30, estimated_new_bytes=20 * 2**30
        )["passed"])
        with self.assertRaises(PermissionError):
            require_prerequisite_gate(
                "artifacts/manifests/allen_cahn_v2_seed42_gate.json"
            )

    def test_all_pdes_one_batch_nlt_and_256_inference(self):
        paths = (
            "artifacts/data/poisson_smoke.index.json",
            "artifacts/data/gray_scott_v4_smoke.index.json",
            "artifacts/data/hyperelastic_smoke.index.json",
            "artifacts/data/navier_stokes_smoke.index.json",
        )
        for path in paths:
            dataset = RolloutShardDataset(path)
            sample = dataset[0]
            fields = int(sample["y"].shape[0])
            conditions = int(sample["x"].shape[0] - fields - 2)
            model = DirichletOperator(
                FNOConfig(
                    in_channels=2 * fields + 2 + conditions,
                    out_channels=fields, modes1=4, modes2=4,
                    width=8, layers=2, condition_channels=conditions,
                ),
                field_scales=tuple([1.0] * fields),
                gauge_channels=(2,) if fields == 3 else (),
            )
            model.configure_adaptation("nlt")
            batch = {
                key: value.unsqueeze(0)
                for key, value in sample.items()
                if key in {"x", "y", "boundary_lift", "boundary_mask",
                           "newton_correction", "valid_correction_mask"}
            }
            batch["field_scales"] = torch.ones(1, fields)
            loss, _ = _train_batch(
                model=model, batch=batch, objective="blrpng",
                training_config={
                    "path_layer": 1, "path_gradient_ratio": 0.15,
                    "path_log_magnitude_weight": 0.1, "path_epsilon": 1e-8,
                },
            )
            loss.backward()
            n = 256
            state = torch.nn.functional.interpolate(
                sample["x"][:fields].unsqueeze(0), size=(n, n),
                mode="bilinear", align_corners=True,
            )
            lift = torch.nn.functional.interpolate(
                sample["boundary_lift"].unsqueeze(0), size=(n, n),
                mode="bilinear", align_corners=True,
            )
            mask = torch.zeros(1, fields, n, n, dtype=torch.bool)
            constrained = sample["boundary_mask"].flatten(1).any(dim=1)
            for channel in range(fields):
                if constrained[channel]:
                    mask[:, channel, (0, -1), :] = True
                    mask[:, channel, :, (0, -1)] = True
            axis = torch.linspace(0.0, 1.0, n)
            xx, yy = torch.meshgrid(axis, axis, indexing="ij")
            auxiliary = [xx[None, None], yy[None, None]]
            if conditions:
                auxiliary.append(torch.nn.functional.interpolate(
                    sample["x"][fields + 2 :].unsqueeze(0), size=(n, n),
                    mode="bilinear", align_corners=True,
                ))
            prediction, _ = model(torch.cat((state, *auxiliary), dim=1), lift, mask)
            self.assertEqual(tuple(prediction.shape), (1, fields, n, n))


if __name__ == "__main__":
    unittest.main()
