from __future__ import annotations

import unittest

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from blrpng.data.lifting import (
    discrete_laplace_interior,
    harmonic_lift,
    rectangular_boundary_mask,
)


def boundary_trace(x, y):
    return 0.2 + 0.1 * x - 0.05 * y + 0.12 * np.sin(2 * np.pi * x)


class HarmonicLiftTests(unittest.TestCase):
    def test_boundary_and_laplace_residual(self):
        axis = np.linspace(0.0, 1.0, 64)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        mask = rectangular_boundary_mask(64, 64)
        values = np.where(mask, boundary_trace(x, y), 0.0)
        lift = harmonic_lift(values)
        self.assertLess(np.max(np.abs(lift[mask] - values[mask])), 1e-12)
        self.assertLess(np.max(np.abs(discrete_laplace_interior(lift))), 1e-11)

    def test_cross_resolution_consistency(self):
        fields = []
        for resolution in (64, 128):
            axis = np.linspace(0.0, 1.0, resolution)
            x, y = np.meshgrid(axis, axis, indexing="ij")
            mask = rectangular_boundary_mask(resolution, resolution)
            fields.append(harmonic_lift(np.where(mask, boundary_trace(x, y), 0.0)))
        fine_axis = np.linspace(0.0, 1.0, 128)
        coarse_axis = np.linspace(0.0, 1.0, 64)
        x, y = np.meshgrid(fine_axis, fine_axis, indexing="ij")
        upsampled = RegularGridInterpolator((coarse_axis, coarse_axis), fields[0])(
            np.column_stack((x.reshape(-1), y.reshape(-1)))
        ).reshape(128, 128)
        self.assertLess(np.sqrt(np.mean((upsampled - fields[1]) ** 2)), 2e-3)


if __name__ == "__main__":
    unittest.main()
