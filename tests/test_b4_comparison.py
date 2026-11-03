import unittest

import numpy as np

from b4_autostitch import common_crop, transforms_to_reference


class B4ComparisonTests(unittest.TestCase):
    def test_composition_and_inverse_direction(self):
        paths=['a.jpg','b.jpg','c.jpg','d.jpg']
        ab=np.array([[1.,0.,3.],[0.,1.,2.],[0.,0.,1.]])
        bc=np.array([[1.,0.1,4.],[0.,1.,-6.],[0.,0.,1.]])
        cd=np.array([[1.,0.,-8.],[0.2,1.,5.],[0.,0.,1.]])
        edges={('a.jpg','b.jpg'):ab,('b.jpg','c.jpg'):bc,('c.jpg','d.jpg'):cd}
        result=transforms_to_reference(paths,'b.jpg',edges)
        expected=[ab,np.eye(3),np.linalg.inv(bc),np.linalg.inv(bc)@np.linalg.inv(cd)]
        for H,E in zip(result,expected):
            np.testing.assert_allclose(H/H[2,2],E/E[2,2])
        result=transforms_to_reference(paths,'d.jpg',edges)
        np.testing.assert_allclose(result[0]/result[0][2,2],cd@bc@ab)

    def test_comparison_uses_identical_world_pixels_and_both_valid_masks(self):
        def colors(height,width,origin):
            y,x=np.indices((height,width))
            return np.stack((x+origin[0]+10,y+origin[1]+10,np.full_like(x,42)),axis=2).astype(np.uint8)
        manual=np.dstack((colors(8,10,(-2,-1)),np.full((8,10),255,dtype=np.uint8)))
        automatic=colors(8,10,(0,0))
        valid=np.ones((8,10),dtype=bool)
        a,b,reference_box,manual_box,auto_box=common_crop(manual,(-2,-1),automatic,valid,(0,0))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(reference_box,[0,0,8,7])
        self.assertEqual(manual_box,[2,1,10,8])
        self.assertEqual(auto_box,[0,0,8,7])
        valid[0,0]=False
        a,b,box,_,_=common_crop(manual,(-2,-1),automatic,valid,(0,0))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(box,[1,0,8,7])


if __name__ == '__main__':
    unittest.main()
