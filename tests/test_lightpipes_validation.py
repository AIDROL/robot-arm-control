import json
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np
from simulation.lightpipes_validation import evaluate, export_actual

ROOT=Path(__file__).resolve().parents[1]


class LightPipesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design=json.loads((ROOT/'configs/optical_setups/three_mounts.json').read_text())
        cls.spec=json.loads((ROOT/'configs/optical_setups/three_mounts_optics.json').read_text())

    def test_nominal_pinhole_transmission_matches_gaussian_abcd(self):
        result,_=evaluate(self.design,self.spec,1024)
        # Independent Gaussian q-parameter prediction at the pinhole.
        wl=self.spec['wavelength_m'];q=1j*np.pi*self.spec['waist_radius_m']**2/wl
        q += .020
        q=1/(1/q-1/.180)+.180
        w=np.sqrt(-wl/(np.pi*np.imag(1/q)))
        expected=1-np.exp(-2*(.000075/w)**2)
        transmission=result['stages'][1]['local_transmission']
        self.assertAlmostEqual(transmission,expected,delta=.002)
        self.assertTrue(result['pass'])
        self.assertLessEqual(result['power_W'],self.spec['input_power_W']*1.000001)

    def test_transverse_pinhole_error_reduces_power_and_fails(self):
        changed=deepcopy(self.design)
        changed['components'][1]['optical_center_m'][1]+=.001
        result,_=evaluate(changed,self.spec,1024)
        self.assertFalse(result['pass'])
        self.assertLess(result['transmitted_fraction'],.01)

    def test_large_tilt_is_rejected_instead_of_silently_ignored(self):
        changed=deepcopy(self.design)
        changed['components'][0]['direction_xyz']=[-.99,np.sqrt(1-.99**2),0]
        with self.assertRaisesRegex(ValueError,'orientation range'):evaluate(changed,self.spec,512)


if __name__=='__main__':unittest.main()
