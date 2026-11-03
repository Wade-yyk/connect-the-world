import unittest

import numpy as np

from feature_matching import match_descriptors


class FeatureMatchingTests(unittest.TestCase):
    def test_hand_calculated_distances_ratio_and_ambiguity(self):
        result = match_descriptors([[0.], [4.9], [10.]], [[0.], [5.], [10.]])
        np.testing.assert_array_equal(result['nearest_indices'], [0, 1, 2])
        np.testing.assert_allclose(result['distance1'], [0, 0.1, 0], atol=1e-12)
        np.testing.assert_allclose(result['distance2'], [5, 4.9, 5], atol=1e-12)
        np.testing.assert_allclose(result['ratios'], [0, 0.1/4.9, 0], atol=1e-12)
        tied = match_descriptors([[5.], [0.]], [[0.], [0.], [10.]])
        self.assertFalse(tied['accepted'].any())

    def test_euclidean_threshold_is_not_applied_to_squared_ratio(self):
        result = match_descriptors([[0.]], [[3.], [4.]], ratio_threshold=0.7)
        self.assertAlmostEqual(result['ratios'][0], 0.75)
        self.assertFalse(result['accepted'][0])
        # Squared ratio 9/16 < 0.7 would incorrectly accept this query.
        self.assertLess(9/16, 0.7)
        boundary = match_descriptors([[0.]], [[7.], [10.]], ratio_threshold=0.7)
        self.assertFalse(boundary['accepted'][0])

    def test_blocked_result_matches_independent_direct_difference_search(self):
        rng = np.random.default_rng(671)
        a, b = rng.normal(size=(17, 64)), rng.normal(size=(23, 64))
        b[:5] = a[:5]+rng.normal(scale=0.1, size=(5,64))
        result = match_descriptors(a, b, block_size=4)
        expected_matches = []
        for i, descriptor in enumerate(a):
            distances = [float(np.linalg.norm(descriptor-target)) for target in b]
            order = sorted(range(len(b)), key=lambda j: (distances[j], j))
            ratio = distances[order[0]]/distances[order[1]]
            self.assertEqual(result['nearest_indices'][i], order[0])
            self.assertEqual(result['second_indices'][i], order[1])
            self.assertAlmostEqual(result['ratios'][i], ratio, places=12)
            if ratio < 0.7:
                expected_matches.append([i, order[0]])
        np.testing.assert_array_equal(result['matches'], expected_matches)

    def test_single_perfect_match_and_insufficient_neighbors(self):
        result = match_descriptors([[0., 0.]], [[0., 0.], [1., 1.]])
        np.testing.assert_array_equal(result['matches'], [[0, 0]])
        for targets in (np.empty((0, 2)), np.zeros((1, 2))):
            result = match_descriptors([[0., 0.]], targets)
            self.assertEqual(result['matches'].shape, (0, 2))
        empty = match_descriptors(np.empty((0, 2)), np.zeros((3, 2)))
        self.assertEqual(empty['matches'].shape, (0, 2))

    def test_one_way_matching_does_not_silently_drop_duplicate_targets(self):
        result = match_descriptors([[0.], [0.1]], [[0.], [10.]])
        np.testing.assert_array_equal(result['matches'], [[0, 0], [1, 0]])

    def test_invalid_descriptors_and_parameters(self):
        for a, b in (([[np.nan]], [[0.], [1.]]), ([[0., 1.]], [[0.], [1.]]), ([1, 2], [[0.], [1.]])):
            with self.assertRaises(ValueError):
                match_descriptors(a, b)
        with self.assertRaises(ValueError):
            match_descriptors([[0.]], [[0.], [1.]], ratio_threshold=1)


if __name__ == '__main__':
    unittest.main()
