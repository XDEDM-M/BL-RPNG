from __future__ import annotations

import unittest

import numpy as np
import torch

from blrpng.models.fno import FNOConfig, DirichletOperator
from blrpng.solvers.progress import select_progress_update
from blrpng.training.losses import (
    balanced_blrpng_loss,
    correction_direction_magnitude_loss,
)


class MethodTests(unittest.TestCase):
    def test_progress_selection(self):
        states = [np.asarray([0.0]), np.asarray([1.0]), np.asarray([2.0]), np.asarray([3.0])]
        residuals = [100.0, 80.0, 5.0, 1.0]
        pre, correction, progress, valid = select_progress_update(states, residuals, 0.25)
        self.assertTrue(valid)
        self.assertGreaterEqual(progress, 0.25)
        np.testing.assert_allclose(pre, states[0])
        np.testing.assert_allclose(pre + correction, states[2])

    def test_scale_invariant_correction_loss(self):
        torch.manual_seed(0)
        prediction = torch.randn(3, 1, 16, 16)
        target = torch.randn(3, 1, 16, 16)
        valid = torch.ones(3, 1, dtype=torch.bool)
        first, _ = correction_direction_magnitude_loss(prediction, target, valid)
        second, _ = correction_direction_magnitude_loss(
            7.0 * prediction, 7.0 * target, valid
        )
        self.assertAlmostEqual(float(first), float(second), places=6)

    def test_gradient_ratio(self):
        parameter = torch.nn.Parameter(torch.tensor([1.2, -0.4]))
        final = (parameter**2).sum()
        path = ((parameter - 1.0) ** 2).sum()
        balanced = balanced_blrpng_loss(
            final_loss=final,
            path_loss=path,
            path_parameters=[parameter],
            target_ratio=0.15,
        )
        self.assertAlmostEqual(balanced.gradient_ratio, 0.15, places=6)

    def test_64_and_128_forward_and_boundary(self):
        config = FNOConfig(modes1=24, modes2=24, width=8, layers=2)
        model = DirichletOperator(config)
        for resolution in (64, 128):
            state = torch.randn(1, 1, resolution, resolution)
            axis = torch.linspace(0, 1, resolution)
            x, y = torch.meshgrid(axis, axis, indexing="ij")
            raw = torch.cat((state, x[None, None], y[None, None]), dim=1)
            lift = torch.zeros_like(state)
            mask = torch.zeros_like(state, dtype=torch.bool)
            mask[:, :, (0, -1), :] = True
            mask[:, :, :, (0, -1)] = True
            prediction, probes = model(raw, lift, mask)
            self.assertEqual(prediction.shape, state.shape)
            self.assertEqual(len(probes), 2)
            self.assertEqual(float(prediction[mask].abs().max().detach()), 0.0)
            self.assertEqual(
                tuple(model.fno.blocks[0].spectral.weight_positive.shape[-2:]),
                (24, 24),
            )


if __name__ == "__main__":
    unittest.main()
