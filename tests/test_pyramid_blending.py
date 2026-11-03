import unittest

import numpy as np

from mosaicing import stitch_mosaic
from pyramid_blending import blend_laplacian, gaussian_blur, laplacian_pyramid, reconstruct


class PyramidTests(unittest.TestCase):
    def test_reconstruction_of_random_images_including_odd_and_thin_sizes(self):
        rng = np.random.default_rng(671)
        for shape in ((31, 47, 3), (32, 48, 3), (1, 8, 3), (7, 1, 3)):
            image = rng.uniform(0, 255, shape).astype(np.float32)
            np.testing.assert_allclose(reconstruct(laplacian_pyramid(image, 6)), image, atol=5e-5)
            constant = np.full(shape, 73, dtype=np.float32)
            np.testing.assert_allclose(gaussian_blur(constant), constant, atol=1e-5)

    def test_uniform_weights_equal_an_analytical_weighted_mean(self):
        rng = np.random.default_rng(12)
        a = rng.uniform(0, 255, (31, 27, 3)).astype(np.float32)
        b = rng.uniform(0, 255, a.shape).astype(np.float32)
        result = blend_laplacian([(a, np.full(a.shape[:2], 0.25)),
                                  (b, np.full(a.shape[:2], 0.75))])
        np.testing.assert_allclose(result, 0.25*a + 0.75*b, atol=6e-5)

    def test_selection_and_identical_colors_preserve_single_image_and_black(self):
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, (19, 23, 3), dtype=np.uint8)
        result = blend_laplacian([(image, np.ones(image.shape[:2]))])
        np.testing.assert_allclose(result, image, atol=4e-5)
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        mask[:, :11] = 1
        constant = np.full_like(image, 70)
        for color in (constant, np.zeros_like(image)):
            result = blend_laplacian([(color, mask), (color, 1-mask)])
            np.testing.assert_allclose(result, color, atol=4e-5)

    def test_stitch_comparison_reuses_geometry_and_preserves_black_and_holes(self):
        image = np.zeros((9, 7, 3), dtype=np.uint8)
        translation = np.array([[1., 0., 12.], [0., 1., 0.], [0., 0., 1.]])
        result = stitch_mosaic([image, image], [np.eye(3), translation], pyramid_levels=4)
        self.assertEqual(result[0].shape, result[4].shape)
        self.assertEqual(result[4].sum(), 0)
        np.testing.assert_array_equal(result[2][0], [1]*7 + [0]*5 + [1]*7)

    def test_invalid_weights_and_canvas_are_rejected(self):
        image = np.ones((7, 9, 3))
        for layers in ([], [(image, -np.ones((7, 9)))], [(image, np.ones((3, 3)))],
                       [(image, np.ones((7, 9))), (image[:3], np.ones((3, 9)))]):
            with self.assertRaises(ValueError):
                blend_laplacian(layers)


if __name__ == "__main__":
    unittest.main()
