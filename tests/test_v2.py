from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from blrpng.analysis.bootstrap import select_oracle_plateau
from blrpng.data.blind_boundaries import (
    FAMILIES,
    evaluate_case_boundary,
    evaluate_blind_boundary,
    sample_boundary_descriptor,
)
from blrpng.analysis.cost import break_even_queries, cost_summary
from blrpng.analysis.gates_v2 import calibration_gate
from blrpng.data.shards import materialize_index, convert_v3_to_v4, validate_shard
from blrpng.models.fno import DirichletOperator, FNOConfig
from blrpng.evaluation.rollout import projected_rollout, projected_rollout_index
from blrpng.training.physics import (
    allen_cahn_defect,
    gray_scott_defect,
    navier_stokes_defect,
    restriction_commutator_loss,
)
from blrpng.training.losses import balanced_blrpng_loss
from blrpng.training.engine import _exact_refined_lift, _train_batch
from blrpng.models.uno import DirichletUNO


class V2Tests(unittest.TestCase):
    def test_blind_boundary_families_are_frozen_and_corner_compatible(self):
        corner_x = np.asarray([0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
        corner_y = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
        for index, family in enumerate(FAMILIES):
            first = sample_boundary_descriptor(family, boundary_id=100 + index, seed=17)
            second = sample_boundary_descriptor(family, boundary_id=100 + index, seed=17)
            self.assertEqual(first, second)
            values = evaluate_blind_boundary(corner_x, corner_y, first)
            self.assertAlmostEqual(float(values[0]), float(values[-1]), places=12)
            line = np.linspace(0.0, 1.0, 2049, dtype=np.float64)
            sampled = evaluate_blind_boundary(line, np.zeros_like(line), first)
            self.assertLessEqual(float(abs(sampled).max()), float(first["amplitude"]) + 1e-10)
            interior = evaluate_case_boundary(
                np.asarray([0.25, 0.5]), np.asarray([0.4, 0.5]),
                {"boundary_descriptor": first},
            )
            self.assertTrue(np.isfinite(interior).all())

    def test_oracle_cost_and_gate_protocol(self):
        plateau = select_oracle_plateau({4: .10, 8: .08, 16: .0795, 32: .0794})
        self.assertEqual(plateau["selected_budget"], 8)
        self.assertEqual(break_even_queries(
            learned_fixed=20, learned_per_query=.1,
            reference_fixed=100, reference_per_query=2.,
        ), 0)
        summary = cost_summary(
            {"assembly_seconds": 1, "solve_seconds": 5, "adaptation_seconds": 2, "inference_seconds_per_query": .01},
            {"assembly_seconds": 5, "solve_seconds": 100, "adaptation_seconds": 20, "inference_seconds_per_query": .01},
            queries=10,
        )
        self.assertGreater(summary["marginal_speedup"], 10)
        gate = calibration_gate(
            oracle_ratio_ci95=(.98, 1.04), marginal_speedup_ci95=(12., 18.),
            scratch_improvement=.8,
        )
        self.assertTrue(gate["passed"])

    def test_multifield_constraints_and_pressure_gauge(self):
        config = FNOConfig(in_channels=8, out_channels=3, modes1=4, modes2=4, width=8, layers=2)
        model = DirichletOperator(config, field_scales=(1.0, 2.0, 0.5), gauge_channels=(2,))
        n = 16
        state = torch.randn(2, 3, n, n)
        axis = torch.linspace(0, 1, n)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        x = torch.cat((state, xx[None, None].expand(2, -1, -1, -1), yy[None, None].expand(2, -1, -1, -1)), dim=1)
        lift = torch.zeros_like(state)
        lift[:, 0, 0] = 0.4
        mask = torch.zeros_like(state, dtype=torch.bool)
        mask[:, :2, (0, -1), :] = True
        mask[:, :2, :, (0, -1)] = True
        prediction, _ = model(x, lift, mask)
        self.assertEqual(float((prediction[:, :2][mask[:, :2]] - lift[:, :2][mask[:, :2]]).abs().max().detach()), 0.0)
        self.assertLess(float(prediction[:, 2].mean(dim=(-2, -1)).abs().max().detach()), 1e-6)

    def test_lora_and_nlt_parameter_efficiency(self):
        for strategy in ("lora", "nlt", "nlt_mix", "nlt_lora"):
            model = DirichletOperator(FNOConfig(modes1=4, modes2=4, width=8, layers=2))
            state = torch.randn(1, 1, 16, 16)
            grid = torch.zeros(1, 2, 16, 16)
            lift = torch.zeros_like(state)
            mask = torch.zeros_like(state, dtype=torch.bool)
            before, _ = model(torch.cat((state, grid), 1), lift, mask)
            info = model.configure_adaptation(strategy, rank=2)
            self.assertLess(info["trainable_parameters"], info["total_parameters"])
            output, _ = model(torch.cat((state, grid), 1), lift, mask)
            if strategy in {"nlt_mix", "nlt_lora"}:
                self.assertTrue(torch.equal(before, output))
            output.mean().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))

    def test_boundary_augmented_and_lift_conditioned_baselines(self):
        n = 16
        state = torch.randn(1, 1, n, n)
        grid = torch.zeros(1, 2, n, n)
        lift = torch.randn_like(state)
        mask = torch.zeros_like(state, dtype=torch.bool)
        mask[:, :, (0, -1), :] = True
        mask[:, :, :, (0, -1)] = True
        for representation in ("boundary_augmented", "lift_conditioned"):
            model = DirichletOperator(
                FNOConfig(in_channels=5, out_channels=1, modes1=4, modes2=4, width=8, layers=2),
                representation=representation,
            )
            prediction, _ = model(torch.cat((state, grid), 1), lift, mask)
            self.assertTrue(torch.equal(prediction[mask], lift[mask]))

    def test_nlt_path_shared_parameters_exclude_probe_and_activate_ratio(self):
        model = DirichletOperator(
            FNOConfig(modes1=4, modes2=4, width=8, layers=2)
        )
        model.configure_adaptation("nlt")
        shared = model.path_active_parameters(1)
        shared_ids = {id(parameter) for parameter in shared}
        self.assertTrue(all(id(parameter) in shared_ids for parameter in model.fno.nlt_scales))
        self.assertTrue(all(id(parameter) in shared_ids for parameter in model.fno.nlt_biases))
        self.assertTrue(all(
            id(parameter) not in shared_ids
            for probe in model.fno.probes for parameter in probe.parameters()
        ))
        final = sum(parameter.square().sum() for parameter in shared)
        path = sum((parameter - 0.5).square().sum() for parameter in shared)
        balanced = balanced_blrpng_loss(
            final_loss=final,
            path_loss=path,
            path_parameters=shared,
            target_ratio=0.15,
        )
        self.assertAlmostEqual(balanced.gradient_ratio, 0.15, places=6)
        self.assertGreater(balanced.final_gradient_norm, 0.0)
        self.assertGreater(balanced.path_gradient_norm, 0.0)
        self.assertEqual(balanced.shared_parameter_count, 32)

    def test_nlt_path_objective_has_nonzero_shared_gradients(self):
        torch.manual_seed(4)
        model = DirichletOperator(
            FNOConfig(modes1=4, modes2=4, width=8, layers=2)
        )
        model.configure_adaptation("nlt")
        n = 16
        state = torch.randn(2, 1, n, n)
        grid = torch.zeros(2, 2, n, n)
        correction = torch.randn_like(state)
        mask = torch.zeros_like(state, dtype=torch.bool)
        mask[:, :, (0, -1), :] = True
        mask[:, :, :, (0, -1)] = True
        correction.masked_fill_(mask, 0.0)
        loss, metrics = _train_batch(
            model=model,
            batch={
                "x": torch.cat((state, grid), dim=1),
                "y": torch.randn_like(state),
                "boundary_lift": torch.zeros_like(state),
                "boundary_mask": mask,
                "newton_correction": correction,
                "valid_correction_mask": torch.ones(2, 1, dtype=torch.bool),
                "field_scales": torch.ones(2, 1),
            },
            objective="blrpng",
            training_config={
                "path_layer": 1,
                "path_gradient_ratio": 0.15,
                "path_log_magnitude_weight": 0.1,
                "path_epsilon": 1e-8,
            },
        )
        self.assertAlmostEqual(metrics["gradient_ratio"], 0.15, places=6)
        self.assertGreater(metrics["path_weight"], 0.0)
        self.assertGreater(metrics["final_shared_gradient_norm"], 0.0)
        loss.backward()

    def test_optional_condition_fields_are_resolution_independent(self):
        model = DirichletOperator(
            FNOConfig(
                in_channels=6, out_channels=1, modes1=4, modes2=4,
                width=8, layers=2, condition_channels=2,
            )
        )
        for n in (16, 32):
            state = torch.randn(1, 1, n, n)
            grid_and_conditions = torch.randn(1, 4, n, n)
            lift = torch.zeros_like(state)
            mask = torch.zeros_like(state, dtype=torch.bool)
            prediction, _ = model(
                torch.cat((state, grid_and_conditions), dim=1), lift, mask
            )
            self.assertEqual(tuple(prediction.shape), (1, 1, n, n))

    def test_gray_scott_reaction_features_ignore_generator_metadata(self):
        model = DirichletOperator(
            FNOConfig(
                in_channels=12, out_channels=2, modes1=4, modes2=4,
                width=8, layers=2, condition_channels=6,
            ),
            input_feature_mode="gray_scott_reaction",
            raw_condition_channels=3,
        )
        n = 16
        state = torch.rand(1, 2, n, n)
        grid = torch.zeros(1, 2, n, n)
        conditions = torch.zeros(1, 3, n, n)
        conditions[:, 0], conditions[:, 1], conditions[:, 2] = .04, .06, -.65
        lift = torch.zeros_like(state)
        mask = torch.zeros_like(state, dtype=torch.bool)
        first, _ = model(torch.cat((state, grid, conditions), 1), lift, mask)
        conditions[:, 2] = .8
        second, _ = model(torch.cat((state, grid, conditions), 1), lift, mask)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(model.model_config_dict()["input_feature_mode"], "gray_scott_reaction")

    def test_exact_multiscale_lift_and_two_step_commutator(self):
        n = 16
        lift = torch.zeros(1, 2, n, n)
        lift[:, :, 0, :] = torch.linspace(0, 1, n)
        refined = _exact_refined_lift(lift, 32)
        lap = (refined[:, :, 2:, 1:-1] + refined[:, :, :-2, 1:-1]
               + refined[:, :, 1:-1, 2:] + refined[:, :, 1:-1, :-2]
               - 4 * refined[:, :, 1:-1, 1:-1])
        self.assertLess(float(lap.abs().max()), 2e-6)
        model = DirichletOperator(
            FNOConfig(
                in_channels=12, out_channels=2, modes1=4, modes2=4,
                width=8, layers=2, condition_channels=6,
            ),
            input_feature_mode="gray_scott_reaction", raw_condition_channels=3,
        )
        state = torch.rand(1, 2, n, n)
        conditions = torch.zeros(1, 3, n, n)
        conditions[:, 0], conditions[:, 1] = .04, .06
        mask = torch.zeros_like(state, dtype=torch.bool)
        mask[:, :, (0, -1), :] = True; mask[:, :, :, (0, -1)] = True
        loss, metrics = _train_batch(
            model=model,
            batch={
                "x": torch.cat((state, torch.zeros(1, 2, n, n), conditions), 1),
                "y": state.clone(), "boundary_lift": lift, "boundary_mask": mask,
                "newton_correction": torch.zeros_like(state),
                "valid_correction_mask": torch.ones(1, 1, dtype=torch.bool),
                "field_scales": torch.tensor([[1.0, .25]]),
            },
            objective="commutator_only",
            training_config={
                "calibration_resolutions": [32, 64],
                "commutator_rollout_steps": 2, "commutator_weight": .05,
            },
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("commutator_16_to_32", metrics)
        self.assertIn("commutator_32_to_64", metrics)
        self.assertIn("commutator_step2_32_to_64", metrics)

    def test_uno_secondary_backbone_lifting_nlt_and_resolution(self):
        model = DirichletUNO(FNOConfig(
            in_channels=4, out_channels=1, modes1=4, modes2=4,
            width=8, layers=5,
        ))
        model.configure_adaptation("nlt")
        for n in (32, 64):
            state = torch.randn(1, 1, n, n)
            grid = torch.zeros(1, 2, n, n)
            lift = torch.zeros_like(state)
            mask = torch.zeros_like(state, dtype=torch.bool)
            mask[:, :, (0, -1), :] = True; mask[:, :, :, (0, -1)] = True
            prediction, probes = model(torch.cat((state, grid), 1), lift, mask)
            self.assertEqual(tuple(prediction.shape), tuple(state.shape))
            self.assertEqual(len(probes), 5)
            self.assertEqual(float(prediction[mask].abs().max().detach()), 0.0)

    def test_commutator_and_residuals_are_differentiable(self):
        coarse = torch.ones(2, 1, 16, 16, requires_grad=True)
        fine = torch.ones(2, 1, 32, 32, requires_grad=True)
        loss = restriction_commutator_loss(fine, coarse)
        self.assertLess(float(loss.detach()), 1e-6)
        loss.backward()
        previous = torch.randn(1, 1, 16, 16)
        current = torch.randn(1, 1, 16, 16, requires_grad=True)
        allen_cahn_defect(previous, current, {"dt": .2, "eps": .03, "gamma": 5.}).square().mean().backward()
        self.assertIsNotNone(current.grad)
        gray_scott_defect(torch.randn(1, 2, 16, 16), torch.randn(1, 2, 16, 16), {
            "dt": 1., "diffusion_u": 2e-5, "diffusion_v": 1e-5, "feed": .04, "kill": .06,
        })
        ns = navier_stokes_defect(torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16), {"dt": .01, "reynolds": 100.})
        self.assertEqual(ns.shape[1], 3)

    def test_v3_to_v4_lazy_equivalence(self):
        index = Path("artifacts/tmp/test_v4.index.json")
        convert_v3_to_v4("artifacts/data/smoke_train.pt", index)
        materialized = materialize_index(index)
        legacy = torch.load("artifacts/data/smoke_train.pt", map_location="cpu", weights_only=False)
        self.assertTrue(torch.allclose(materialized.tensors["x"], legacy["x"], atol=1e-7, rtol=0.0))
        self.assertTrue(torch.equal(materialized.tensors["y"], legacy["y"]))
        self.assertTrue(torch.equal(materialized.tensors["newton_correction"], legacy["newton_correction"]))
        model = DirichletOperator(FNOConfig(modes1=4, modes2=4, width=8, layers=2))
        memory = projected_rollout(model, materialized, device=torch.device("cpu"), horizon=1)
        stream = projected_rollout_index(model, str(index), device=torch.device("cpu"), horizon=1)
        self.assertTrue(torch.allclose(torch.tensor(memory["per_rollout_rel_l2"]),
                                       torch.tensor(stream["per_rollout_rel_l2"]), atol=1e-7, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
