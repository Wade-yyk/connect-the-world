import unittest

import numpy as np

from mosaicing import largest_valid_rectangle, stitch_mosaic


class MosaicTests(unittest.TestCase):
    def test_crop_matches_exhaustive_maximum_and_avoids_holes(self):
        rng = np.random.default_rng(671)
        masks = [np.ones((3, 5), dtype=bool), np.eye(4, dtype=bool),
                 np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)]
        masks.extend(rng.random((4, 5)) > 0.3 for _ in range(30))
        for mask in masks:
            left, top, right, bottom = largest_valid_rectangle(mask)
            self.assertTrue(mask[top:bottom, left:right].all())
            maximum = 0
            for y0 in range(mask.shape[0]):
                for y1 in range(y0 + 1, mask.shape[0] + 1):
                    for x0 in range(mask.shape[1]):
                        for x1 in range(x0 + 1, mask.shape[1] + 1):
                            if mask[y0:y1, x0:x1].all():
                                maximum = max(maximum, (y1-y0)*(x1-x0))
            self.assertEqual((right-left)*(bottom-top), maximum)

    def test_crop_rejects_empty_masks(self):
        for mask in (np.zeros((3, 5)), np.zeros((0, 2)), np.ones(3)):
            with self.assertRaises(ValueError):
                largest_valid_rectangle(mask)

    def test_negative_canvas_and_hand_calculated_feather_blend(self):
        black = np.zeros((5, 5, 3), dtype=np.uint8)
        white = np.full_like(black, 240)
        shifted = np.array([[1., 0., -2.], [0., 1., 0.], [0., 0., 1.]])
        result, overwrite, coverage, origin = stitch_mosaic([black, white], [shifted, np.eye(3)])
        self.assertEqual(origin, (-2, 0))
        self.assertEqual(result.shape, (5, 7, 3))
        # At reference x=0,y=2: black center weight=1, white edge=1/3.
        np.testing.assert_array_equal(result[2, 2], [60, 60, 60])
        np.testing.assert_array_equal(result[2, 3], [120, 120, 120])
        np.testing.assert_array_equal(result[2, 4], [180, 180, 180])
        np.testing.assert_array_equal(overwrite[2, 2], [240, 240, 240])
        np.testing.assert_array_equal(coverage[2], [1, 1, 2, 2, 2, 1, 1])

    def test_black_pixels_and_uncovered_holes_are_distinct(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        shifted = np.array([[1., 0., 4.], [0., 1., 0.], [0., 0., 1.]])
        result, _, coverage, _ = stitch_mosaic([image, image], [np.eye(3), shifted])
        self.assertEqual(result.sum(), 0)
        np.testing.assert_array_equal(coverage[0], [1, 1, 0, 0, 1, 1])

    def test_single_image_preserves_borders_and_blending_is_order_independent(self):
        image = np.arange(75, dtype=np.uint8).reshape(5, 5, 3)
        result, _, coverage, _ = stitch_mosaic([image], [np.eye(3)])
        np.testing.assert_array_equal(result, image)
        self.assertTrue(coverage.all())
        other = 255 - image
        a = stitch_mosaic([image, other], [np.eye(3), np.eye(3)])[0]
        b = stitch_mosaic([other, image], [np.eye(3), np.eye(3)])[0]
        np.testing.assert_array_equal(a, b)

    def test_invalid_inputs(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for images, matrices in (([], []), ([image], []), ([image[..., 0]], [np.eye(3)]),
                                 ([image], [np.zeros((3, 3))])):
            with self.subTest(images=len(images)), self.assertRaises(ValueError):
                stitch_mosaic(images, matrices)


if __name__ == "__main__":
    unittest.main()
