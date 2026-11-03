import unittest

import numpy as np

from descriptors import extract_descriptors


class DescriptorTests(unittest.TestCase):
    def test_analytic_plane_sampling_normalization_and_coordinate_order(self):
        y, x = np.indices((160, 180), dtype=float)
        plane = (x+3*y+10)/1000
        result = extract_descriptors(plane, [[80, 75]])
        offsets = np.array([-18, -13, -8, -3, 2, 7, 12, 17])
        expected = ((80+offsets[None, :])+3*(75+offsets[:, None])+10)/1000
        np.testing.assert_allclose(result['samples8'][0], expected, atol=2e-8)
        normalized = (expected-expected.mean())/expected.std()
        np.testing.assert_allclose(result['descriptors'][0].reshape(8, 8), normalized, atol=2e-7)
        self.assertEqual(result['descriptors'].shape, (1, 64))

    def test_positive_gain_and_bias_do_not_change_the_descriptor(self):
        rng = np.random.default_rng(671)
        image = rng.uniform(0.2, 0.6, (100, 110))
        points = [[40, 40], [65, 60]]
        a = extract_descriptors(image, points)
        b = extract_descriptors(1.4*image+0.05, points)
        np.testing.assert_allclose(a['descriptors'], b['descriptors'], atol=1e-6)
        np.testing.assert_allclose(a['descriptors'].mean(axis=1), 0, atol=1e-7)
        np.testing.assert_allclose(a['descriptors'].std(axis=1), 1, atol=1e-7)

    def test_invalid_borders_and_flat_windows_are_rejected_without_nan(self):
        image = np.ones((80, 90))*0.5
        result = extract_descriptors(image, [[0, 30], [45, 40], [89, 40]])
        np.testing.assert_array_equal(result['border_rejected'], [0, 2])
        np.testing.assert_array_equal(result['contrast_rejected'], [1])
        self.assertEqual(result['descriptors'].shape, (0, 64))
        self.assertTrue(np.isfinite(result['descriptors']).all())

    def test_source_indices_and_rgb_grayscale_consistency(self):
        rng = np.random.default_rng(9)
        image = rng.uniform(0.2, 0.7, (80, 90))
        points = [[0, 0], [30, 30], [70, 60], [89, 79]]
        a = extract_descriptors(image, points)
        b = extract_descriptors(np.repeat(image[..., None], 3, axis=2), points)
        np.testing.assert_array_equal(a['source_indices'], [1, 2])
        np.testing.assert_array_equal(a['points'], [[30, 30], [70, 60]])
        np.testing.assert_allclose(a['descriptors'], b['descriptors'], atol=1e-6)
        restored = a['descriptors'].reshape(-1, 8, 8)*a['stds'][:, None, None]+a['means'][:, None, None]
        np.testing.assert_allclose(restored, a['samples8'], atol=3e-8)

    def test_empty_and_invalid_inputs(self):
        image = np.ones((80, 90))
        result = extract_descriptors(image, np.empty((0, 2)))
        self.assertEqual(result['points'].shape, (0, 2))
        for points in ([[20.5, 30]], [[20, np.nan]], [20, 30]):
            with self.assertRaises(ValueError):
                extract_descriptors(image, points)
        with self.assertRaises(ValueError):
            extract_descriptors(np.full((80, 90), np.inf), [[40, 40]])


if __name__ == '__main__':
    unittest.main()
