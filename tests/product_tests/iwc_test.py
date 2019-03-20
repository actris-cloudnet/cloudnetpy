from unittest import TestCase
from cloudnetpy.products.iwc import IwcSource
import numpy as np
from numpy import testing

fname = '/home/korpinen/Documents/ACTRIS/cloudnet_data/categorize_test_file_new.nc'
#TODO: Mieti, miten saa tiedostopolun kutsulla mukaan

class TestIwcSource(TestCase):
    """
    Test correct functionality of functions in class IwcSource
    """
    def setUp(self):
        self.freq = [0, 1]
        self.liq_att = {0: 1.0, 1: 4.5}
        self.coeff = {0: (0.878, 0.000242, -0.0186, 0.0699, -1.63),
                      1: (0.669, 0.000580, -0.00706, 0.0923, -0.992)}
        self.factor = {0: -0.24988, 1: -1.43056}
        self.iwcS = IwcSource(fname)


    def test__wl_band(self):
        errors = []
        ind = 0
        for i in range(len(self.freq)):
            if not self.iwcS.wl_band == self.freq[i]:
                errors.append("Wavelength band not correct")
            else:
                ind += 1
        if ind > 0:
            errors = []
        assert not errors, "errors occured:\n{}".format("\n".join(errors))


    def test__get_approximative_specific_liquid_atten(self):
        self.assertEqual(self.iwcS.spec_liq_atten, self.liq_att[self.iwcS.wl_band])


    def test__get_iwc_coeffs(self):
        self.assertEqual(self.iwcS.coeffs[0:],self.coeff[self.iwcS.wl_band])


    def test__get_z_factor(self):
        self.assertEqual(round(self.iwcS.z_factor, 5),
                         self.factor[self.iwcS.wl_band])


    def test__get_subzero_temperatures(self):
        self.assertLess(np.max(self.iwcS.temperature), 0)


    def test__get_mean_temperature(self):
        testing.assert_array_equal(self.iwcS.mean_temperature,
                        np.mean(self.iwcS.temperature, axis=0))



