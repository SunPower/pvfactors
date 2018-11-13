# -*- coding: utf-8 -*-

"""
Test the calculation of the irradiance terms of the mathematical formulation
"""

from pvfactors.pvarray import Array
import numpy as np


def test_irradiance_terms_perez_but_isotropic():
    """
    Check that the irradiance terms calculated for this configuration and
    using the isotropic diffuse sky dome is working as expected
    There is some direct shading on the back surface in this configuration
    """
    # Simple sky and array configuration
    dni = 1e3
    dhi = 1e2
    solar_zenith = 80.
    solar_azimuth = 0.
    tracker_theta = 70.
    surface_azimuth = 180.
    arguments = {
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'gcr': 0.4,
    }
    # Create vf array
    array = Array(**arguments)
    # Calculate irradiance terms
    array.update_view_factors(solar_zenith, solar_azimuth, tracker_theta,
                              surface_azimuth)
    array.update_irradiance_terms_perez(solar_zenith, solar_azimuth,
                                        tracker_theta, surface_azimuth,
                                        dni, dhi, 0., 0., 0.)

    # Check that the values are as expected
    expected_irradiance_terms = np.array(
        [0., 0., 0., 0.,
         173.64817767, 173.64817767, 173.64817767, 173.64817767,
         173.64817767, 0., 0., 866.02540378,
         866.02540378, 866.02540378,
         100.])
    tol = 1e-8
    assert np.allclose(array.irradiance_terms, expected_irradiance_terms,
                       atol=tol, rtol=0)
