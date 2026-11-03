"""Numerical and geometric checks of the hand-built homography solver."""

import unittest

import numpy as np

from homography import build_linear_system, computeH, project_points


class HomographyTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[25, 40], [1200, 35], [1150, 1600], [40, 1550],
                                [200, 400], [850, 300], [900, 1250], [300, 1300]], dtype=float)
        self.H = np.array([[1.1, 0.04, -220], [-0.03, 0.95, 35], [0.0002, -0.0001, 1]])
        # Construct targets independently from the helper being tested.
        x, y = self.points.T
        w = 0.0002*x - 0.0001*y + 1
        self.targets = np.column_stack(((1.1*x + 0.04*y - 220)/w,
                                         (-0.03*x + 0.95*y + 35)/w))

    def test_exact_four_and_overdetermined_recovery(self):
        for count in (4, 8):
            with self.subTest(count=count):
                recovered = computeH(self.points[:count], self.targets[:count])
                np.testing.assert_allclose(recovered, self.H, atol=1e-8, rtol=1e-9)

    def test_equations_and_coordinate_direction(self):
        A, b = build_linear_system(self.points, self.targets)
        self.assertEqual(A.shape, (16, 8))
        np.testing.assert_allclose(A @ self.H.ravel()[:8], b, atol=1e-10)
        np.testing.assert_allclose(project_points(self.points, self.H), self.targets, atol=1e-10)
        np.testing.assert_allclose(project_points(self.targets, np.linalg.inv(self.H)), self.points, atol=1e-9)

    def test_noisy_fit_is_original_pixel_least_squares(self):
        noisy = self.targets + np.random.default_rng(671).normal(0, 0.6, self.targets.shape)
        recovered = computeH(self.points, noisy)
        A, b = build_linear_system(self.points, noisy)
        expected = np.linalg.lstsq(A, b, rcond=None)[0]
        np.testing.assert_allclose(recovered.ravel()[:8], expected, atol=1e-7, rtol=1e-7)
        # Compare on independent interior points, not only fitted samples.
        held_out = np.array([[500, 700], [750, 950], [550, 1100]])
        error = np.linalg.norm(project_points(held_out, recovered) - project_points(held_out, self.H), axis=1)
        self.assertLess(error.max(), 2.0)

    def test_degenerate_and_invalid_inputs_are_rejected(self):
        bad_inputs = [
            (self.points[:3], self.targets[:3]),
            (self.points, self.targets[:4]),
            (np.ones((4, 2)), np.ones((4, 2))),
            (np.array([[0, 0], [1, 1], [2, 2], [3, 3]]), self.targets[:4]),
            (np.array([[0, 0], [0, 0], [1, 0], [0, 1]]), np.array([[0, 0], [0, 0], [1, 0], [0, 1]])),
            (np.full((4, 2), np.nan), self.targets[:4]),
            ([1, 2, 3], [1, 2, 3]),
        ]
        for p, q in bad_inputs:
            with self.subTest(p=p), self.assertRaises(ValueError):
                computeH(p, q)

    def test_projective_scale_and_points_at_infinity(self):
        np.testing.assert_allclose(project_points(self.points, 1e-20*self.H), self.targets, atol=1e-9)
        H = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -25]], dtype=float)
        with self.assertRaises(ValueError):
            project_points([[25, 40]], H)


if __name__ == "__main__":
    unittest.main()
