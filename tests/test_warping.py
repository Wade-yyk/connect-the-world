import unittest

import numpy as np

from homography import computeH
from warping import warped_bounds, warpImageBilinear, warpImageNearestNeighbor


class WarpingTests(unittest.TestCase):
    def test_identity_preserves_every_pixel_and_border(self):
        for shape in ((4, 5), (4, 5, 3), (1, 1), (1, 5), (5, 1)):
            image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
            for warp in (warpImageNearestNeighbor, warpImageBilinear):
                with self.subTest(shape=shape, method=warp.__name__):
                    output, mask = warp(image, np.eye(3), return_mask=True)
                    np.testing.assert_array_equal(output, image)
                    self.assertTrue(mask.all())

    def test_hand_calculated_fractional_sample_and_invalid_region(self):
        image = np.array([[0., 10.], [20., 30.]])
        H = np.array([[1., 0., -0.25], [0., 1., -0.75], [0., 0., 1.]])
        nearest, mask_nn = warpImageNearestNeighbor(image, H, (2, 2), return_mask=True)
        bilinear, mask_bil = warpImageBilinear(image, H, (2, 2), return_mask=True)
        self.assertEqual(nearest[0, 0], 20.)
        self.assertAlmostEqual(bilinear[0, 0], 17.5)
        np.testing.assert_array_equal(mask_nn, [[True, False], [False, False]])
        np.testing.assert_array_equal(mask_bil, mask_nn)
        self.assertEqual(bilinear[1, 1], 0.)

    def test_negative_translation_uses_complete_bounds(self):
        image = np.arange(20).reshape(4, 5)
        H = np.array([[1., 0., -7.], [0., 1., 3.], [0., 0., 1.]])
        self.assertEqual(warped_bounds(image.shape, H), ((4, 5), (-7, 3)))
        np.testing.assert_array_equal(warpImageBilinear(image, H), image)
        output, mask = warpImageNearestNeighbor(image, H, (10, 12), origin=(-9, 1), return_mask=True)
        np.testing.assert_array_equal(output[2:6, 2:7], image)
        self.assertEqual(mask.sum(), image.size)

    def test_projective_rectification_on_analytic_plane(self):
        y, x = np.indices((20, 30), dtype=float)
        plane = 3*x + 5*y + 7
        source = np.array([[3., 2.], [25., 4.], [22., 17.], [5., 16.]])
        target = np.array([[0., 0.], [12., 0.], [12., 9.], [0., 9.]])
        H = computeH(source, target)
        output, mask = warpImageBilinear(plane, H, (10, 13), return_mask=True)
        inverse = np.linalg.inv(H)
        # Independent scalar geometry and analytic plane values at each pixel.
        for row in range(10):
            for col in range(13):
                mapped = inverse @ np.array([col, row, 1.])
                sx, sy = mapped[:2] / mapped[2]
                self.assertTrue(mask[row, col])
                self.assertAlmostEqual(output[row, col], 3*sx + 5*sy + 7, places=9)

    def test_black_pixels_are_valid_and_projective_scale_is_irrelevant(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        output, mask = warpImageBilinear(image, -1e-20*np.eye(3), return_mask=True)
        self.assertTrue(mask.all())
        np.testing.assert_array_equal(output, image)

    def test_invalid_homographies_and_unbounded_canvas_are_rejected(self):
        image = np.ones((3, 4))
        for H in (np.zeros((3, 3)), np.full((3, 3), np.nan), np.eye(2)):
            with self.subTest(H=H), self.assertRaises(ValueError):
                warpImageBilinear(image, H)
        horizon = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., -1.5]])
        with self.assertRaises(ValueError):
            warped_bounds(image.shape, horizon)
        with self.assertRaises(ValueError):
            warpImageNearestNeighbor(image, np.eye(3), (0, 5))


if __name__ == "__main__":
    unittest.main()
