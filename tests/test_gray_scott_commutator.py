import unittest
import torch
from blrpng.training.gray_scott_commutator import field_commutator, prolong_lifted, refined_geometry


class GrayScottCommutatorTests(unittest.TestCase):
    def test_constant_prediction_commutes_with_finite_zero_gradient(self):
        fine = torch.ones(1,2,32,32, requires_grad=True)
        coarse = torch.ones(1,2,16,16, requires_grad=True)
        mask = torch.zeros_like(coarse, dtype=torch.bool)
        loss = field_commutator(fine, coarse, torch.ones_like(coarse), mask)
        self.assertLess(float(loss.detach()), 1e-7)
        loss.backward()
        self.assertTrue(torch.isfinite(fine.grad).all())
        self.assertTrue(torch.isfinite(coarse.grad).all())

    def test_symmetric_gradients_and_field_weighting(self):
        fine = torch.zeros(1,2,32,32, requires_grad=True)
        coarse = torch.ones(1,2,16,16, requires_grad=True)
        target = torch.ones_like(coarse)
        mask = torch.zeros_like(coarse, dtype=torch.bool)
        loss = field_commutator(fine, coarse, target, mask)
        loss.backward()
        self.assertGreater(float(fine.grad.abs().sum()),0)
        self.assertGreater(float(coarse.grad.abs().sum()),0)
        self.assertAlmostEqual(float(loss.detach()),1.,places=6)

    def test_refinement_requires_no_solution_and_preserves_trace(self):
        p = {'meta': {'frequency_range':[1,4], 'boundary_seed':42,
                      'correlation':.5, 'amplitude':.1},
             'condition_fields': torch.zeros(3,16,16)}
        geo = refined_geometry(p,32,torch.device('cpu'))
        lift, mask, _, _ = geo
        coarse = torch.randn(1,2,16,16,requires_grad=True)
        result = prolong_lifted(coarse,torch.zeros_like(coarse),geo)
        self.assertEqual(float((result-lift).masked_select(mask).detach().abs().max()),0)
        result.sum().backward()
        self.assertTrue(torch.isfinite(coarse.grad).all())


if __name__ == '__main__': unittest.main()
