import unittest

import numpy as np

from homography import project_points
from ransac import ransac_homography, reprojection_errors


class RansacTests(unittest.TestCase):
    def test_recovers_known_projective_model_despite_noise_and_outliers(self):
        rng = np.random.default_rng(6)
        source = rng.uniform([0,0], [1000,800], (70,2))
        H = np.array([[1.03,0.04,-120], [-0.02,0.97,30], [0.00008,-0.00003,1.]])
        target = project_points(source,H)
        target[:45] += rng.normal(0,0.2,(45,2))
        target[45:] = rng.uniform([-200,-100],[1000,900],(25,2))
        result = ransac_homography(source,target,threshold_px=1.5,max_iterations=2000,min_iterations=100)
        self.assertEqual(int(result['inliers'].sum()),45)
        self.assertTrue(result['inliers'][:45].all())
        self.assertFalse(result['inliers'][45:].any())
        held_out = rng.uniform([0,0],[1000,800],(30,2))
        error = np.linalg.norm(project_points(held_out,result['H'])-project_points(held_out,H),axis=1)
        self.assertLess(np.sqrt(np.mean(error**2)),0.3)
        np.testing.assert_array_equal(result['inliers'],result['errors'] < 1.5)
        self.assertTrue(result['consensus_stable'])
        np.testing.assert_array_equal(result['fit_mask'],result['inliers'])

    def test_four_exact_pairs_and_seed_reproducibility(self):
        source = np.array([[0.,0.],[100.,0.],[100.,80.],[0.,80.]])
        H = np.array([[1.,0.02,7.],[0.01,1.,-4.],[0.0001,0.0002,1.]])
        target = project_points(source,H)
        a = ransac_homography(source,target,min_iterations=10,max_iterations=100,seed=7)
        b = ransac_homography(source,target,min_iterations=10,max_iterations=100,seed=7)
        np.testing.assert_allclose(a['H'],H,atol=1e-10)
        np.testing.assert_array_equal(a['H'],b['H'])
        self.assertEqual(a['trials'],b['trials'])
        self.assertTrue(a['inliers'].all())

    def test_infinite_projection_does_not_discard_other_points(self):
        H = np.array([[1.,0.,0.],[0.,1.,0.],[1.,0.,-1.]])
        source = np.array([[1.,2.],[2.,2.]])
        errors = reprojection_errors(source,np.array([[0.,0.],[2.,2.]]),H)
        self.assertTrue(np.isinf(errors[0]))
        self.assertEqual(errors[1],0)

    def test_degenerate_samples_and_invalid_inputs(self):
        line = np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
        with self.assertRaises(ValueError):
            ransac_homography(line,line)
        three_collinear = np.array([[0.,0.],[10.,0.],[20.,0.],[0.,10.]])
        with self.assertRaises(ValueError):
            ransac_homography(three_collinear,three_collinear,max_iterations=20,min_iterations=10)
        with self.assertRaises(ValueError):
            ransac_homography(np.zeros((3,2)),np.zeros((3,2)))
        rectangle = np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
        with self.assertRaises(ValueError):
            ransac_homography(rectangle,rectangle,threshold_px=-1)


if __name__ == '__main__':
    unittest.main()
