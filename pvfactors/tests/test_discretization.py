# -*- coding: utf-8 -*-

"""
Check that the discretization of the pvrow surfaces is working
"""

from pvfactors.pvarray import Array
import numpy as np


def test_discretized_surfaces():
    """
    Functional test to check that the discretization is working
    """
    # Create vf pv array with discretization request
    arguments = {
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'gcr': 0.4,
        'tracker_theta': 20.,
        'cut': [(0, 5, 'front'), (1, 3, 'front')]
    }
    array = Array(**arguments)

    # Check that the number of discretized surfaces is correct
    front_pvrow_0_has_5_surfaces = (
        array.surface_registry.loc[
            (array.surface_registry.pvrow_index == 0)
            & (array.surface_registry.surface_side == 'front'),
            :].shape[0] == 5
    )
    front_pvrow_1_has_3_surfaces = (
        array.surface_registry.loc[
            (array.surface_registry.pvrow_index == 1)
            & (array.surface_registry.surface_side == 'front'),
            :].shape[0] == 3
    )
    assert front_pvrow_0_has_5_surfaces & front_pvrow_1_has_3_surfaces


def test_consistent_qinc():
    """
    Test that the values of the calculated incident irradiance on all the
    surfaces (discretized or not) stays consistent
    """
    arguments = {
        'n_pvrows': 5,
        'pvrow_height': 1.,
        'solar_zenith': 70,
        'solar_azimuth': 180.,
        'array_azimuth': 180.,
        'pvrow_width': 1.5,
        'gcr': 0.6,
        'tracker_theta': 30.,
        'cut': [(0, 5, 'front'), (4, 2, 'front')]
    }
    array = Array(**arguments)

    # Run a calculation for the given configuration
    dni = 1e3
    dhi = 1e2
    luminance_isotropic = dhi
    luminance_circumsolar = 0.
    poa_horizon = 0.
    poa_circumsolar = 0.

    solar_zenith = 20.
    solar_azimuth = 180.

    tracker_theta = 20.
    surface_azimuth = 180.

    array.calculate_radiosities_perez(solar_zenith, solar_azimuth,
                                      tracker_theta, surface_azimuth,
                                      dni, luminance_isotropic,
                                      luminance_circumsolar, poa_horizon,
                                      poa_circumsolar)

    # Compare to expected values
    expected_qinc = np.array([
        1103.10852238, 1097.25580244, 1096.27281294, 1095.61848916,
        1093.44645666, 47.62848148, 37.3519236, 36.41100695,
        36.49146269, 45.94191771, 993.12051239, 991.41490791,
        991.3692459, 991.83044071, 1039.63791536, 1039.23551228,
        1026.62401361, 49.74148561, 44.25032849, 43.80024727,
        44.00294823, 76.10124014, 69.32324555, 69.60603804,
        71.31511657, 93.12702949, 1103.09038367, 1103.07239864,
        1103.05456561, 1103.03688292, 1097.08812485])

    tol = 1e-8
    assert np.allclose(array.surface_registry.qinc, expected_qinc,
                       atol=0, rtol=tol, equal_nan=True)
