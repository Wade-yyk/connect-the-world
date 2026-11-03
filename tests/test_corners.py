import unittest

import numpy as np

from corners import anms, detect_harris


class CornerTests(unittest.TestCase):
    def test_anms_hand_calculated_radii_and_ranking(self):
        points = np.array([[0., 0.], [3., 0.], [20., 0.], [5., 0.]])
        selected, radii = anms(points, [10., 8., 7., 9.5], 3, block_size=2)
        np.testing.assert_array_equal(radii, [np.inf, 2, 15, np.inf])
        np.testing.assert_array_equal(selected, [0, 3, 2])
        _, boundary = anms([[0, 0], [1, 0]], [10., 9.])
        self.assertTrue(np.isinf(boundary).all())

    def test_anms_matches_independent_scalar_search(self):
        rng = np.random.default_rng(671)
        points = rng.uniform(0, 100, (37, 2))
        scores = rng.uniform(0.1, 20, 37)
        selected, radii = anms(points, scores, 9, block_size=7)
        expected = []
        for i in range(len(points)):
            distances = [float(np.hypot(*(points[i]-points[j]))) for j in range(len(points))
                         if scores[i] < 0.9*scores[j]]
            expected.append(min(distances, default=np.inf))
        np.testing.assert_allclose(radii, expected)
        expected_order = sorted(range(len(points)), key=lambda i: (-expected[i], -scores[i], points[i, 1], points[i, 0]))
        np.testing.assert_array_equal(selected, expected_order[:9])

    def test_harris_finds_rectangle_corners_and_rejects_flat_and_straight_edge(self):
        image = np.zeros((100, 120))
        image[25:75, 30:90] = 1
        response, points, scores = detect_harris(image, border=10)
        self.assertEqual(len(points), 4)
        for expected in ((30, 25), (89, 25), (30, 74), (89, 74)):
            self.assertLessEqual(np.linalg.norm(points-expected, axis=1).min(), 3)
        self.assertTrue(np.all(scores > 0))
        np.testing.assert_allclose(scores, response[points[:, 1], points[:, 0]])
        edge = np.zeros_like(image)
        edge[:, 60:] = 1
        for simple in (edge, np.ones_like(image)):
            _, points, _ = detect_harris(simple)
            self.assertEqual(len(points), 0)

    def test_coordinate_border_and_intensity_scale(self):
        image = np.zeros((100, 120))
        image[25:75, 30:90] = 1
        _, a, _ = detect_harris(image)
        _, b, _ = detect_harris((image*255).astype(np.uint8))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(a >= 20))
        self.assertTrue(np.all(a[:, 0] < 100))
        self.assertTrue(np.all(a[:, 1] < 80))

    def test_empty_candidates_and_invalid_input(self):
        indices, radii = anms(np.empty((0, 2)), [], 500)
        self.assertEqual(len(indices), 0)
        self.assertEqual(len(radii), 0)
        for points, scores in (([[0, 0]], [0]), ([[0, np.nan]], [1]), ([[0, 0]], [1, 2])):
            with self.assertRaises(ValueError):
                anms(points, scores)
        with self.assertRaises(ValueError):
            detect_harris(np.ones((1, 10)))
        with self.assertRaises(ValueError):
            detect_harris(np.full((10, 10), np.nan))


if __name__ == "__main__":
    unittest.main()
