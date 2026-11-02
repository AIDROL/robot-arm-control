import unittest
import cv2
import numpy as np
from simulation.vision_overlay import annotate
from simulation.global_camera import DICTIONARY


class VisionOverlayTests(unittest.TestCase):
    def fixture(self):
        rgb=np.full((480,640,3),255,np.uint8)
        rgb[140:260,240:360]=cv2.aruco.generateImageMarker(DICTIONARY,2,120)[:,:,None]
        return rgb,np.array([[600.,0,319.5],[0,600.,239.5],[0,0,1.]])

    def test_detection_is_image_derived_and_input_is_unchanged(self):
        rgb,K=self.fixture();before=rgb.copy()
        overlay,records=annotate(rgb,K,'test')
        self.assertEqual([r['id'] for r in records],[2])
        self.assertEqual(np.array(records[0]['corners_px']).shape,(4,2))
        self.assertTrue(records[0]['pose_valid'])
        self.assertLess(records[0]['reprojection_error_px'],1.)
        np.testing.assert_array_equal(rgb,before)
        self.assertEqual(overlay.shape,(360,640,3))

    def test_missing_marker_never_gets_a_box(self):
        _,records=annotate(np.full((480,640,3),255,np.uint8),np.eye(3),'test')
        self.assertEqual(records,[])

    def test_depth_reuses_rgb_corner_coordinates(self):
        rgb,K=self.fixture();_,records=annotate(rgb,K,'rgb')
        depth=np.zeros_like(rgb)
        overlay,reused=annotate(rgb,K,'depth',detections=records,background=depth)
        self.assertIs(reused,records)
        x,y=np.rint(np.array(records[0]['corners_px'][0])*[1.,.75]).astype(int)
        self.assertTrue(overlay[y-3:y+4,x-3:x+4].any())
        self.assertFalse(depth.any())


if __name__=='__main__':unittest.main()
