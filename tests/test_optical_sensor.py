import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from simulation.optical_sensor import observe, publish, read, usable


class OpticalSensorTests(unittest.TestCase):
    def signal(self):
        a=np.zeros((64,64));a[29:36,33:40]=10
        return a

    def test_metrics_come_from_pixels_and_preserve_transverse_offset(self):
        image,o=observe(self.signal(),.006,.001,3.)
        self.assertTrue(o['valid'])
        self.assertAlmostEqual(o['centroid_yz_m'][0],4*.006/64)
        self.assertAlmostEqual(o['centroid_yz_m'][1],0.)
        self.assertAlmostEqual(o['power_W'],49*10*(.006/64)**2,delta=1e-8)
        self.assertEqual(image.dtype,np.uint16)
        self.assertTrue(usable(o,3.1)[0]);self.assertFalse(usable(o,4.)[0])

    def test_dark_nan_saturation_and_edge_are_not_usable(self):
        for a,reason in [(np.zeros((64,64)),'NO_SIGNAL'),
                         (np.full((64,64),np.nan),'INVALID_INTENSITY'),
                         (self.signal()*200,'SATURATED'),
                         (np.ones((64,64))*10,'BEAM_AT_SENSOR_EDGE')]:
            _,o=observe(a,.006,.001,3.)
            self.assertFalse(o['valid']);self.assertIn(reason,o['reasons'])

    def test_publish_pairs_image_and_observation_and_rejects_corruption(self):
        with TemporaryDirectory() as temp:
            spec={'window_m':.006,'input_power_W':.001}
            publish(temp,self.signal(),spec,3.)
            obs,frame=read(temp)
            self.assertTrue(obs['valid']);self.assertEqual(frame.shape,(780,640,3))
            (Path(temp)/'preview.png').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):read(temp)


if __name__=='__main__':unittest.main()
