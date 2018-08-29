# -*- coding: utf-8 -*-

"""
Test the implementation of diffuse shading
"""

from pvfactors.pvarray import Array
import os

TEST_DIR = os.path.join(__file__)


def test_calculate_back_horizon_shading():

    arguments = {
        'array_azimuth': 90.0,
        'array_tilt': -30.0,
        'gcr': 0.5,
        'n_pvrows': 2,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'rho_ground': 0.2,
        'rho_pvrow_back': 0.03,
        'rho_pvrow_front': 0.01,
        'solar_azimuth': 90.0,
        'solar_zenith': 30.0,
        'circumsolar_angle': 50.,
        'horizon_band_angle': 6.5,
        'calculate_front_circ_horizon_shading': False,
        'circumsolar_model': 'gaussian'
    }

    # Create shapely PV array
    array = Array(**arguments)
