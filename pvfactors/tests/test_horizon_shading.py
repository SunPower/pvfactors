# -*- coding: utf-8 -*-

"""
Test the implementation of diffuse shading
"""

from pvfactors.pvarray import Array
import os
import numpy as np
import pandas as pd

TEST_DIR = os.path.join(__file__)

idx_slice = pd.IndexSlice


def test_calculate_back_horizon_shading():

    arguments = {
        'surface_azimuth': 0.0,
        'surface_tilt': 30.0,
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

    # Test the horizon band shading part
    solar_zenith = 45.
    solar_azimuth = 90.
    surface_tilt = 30.
    surface_azimuth = 0.
    dni = 0.
    luminance_isotropic = 0.
    luminance_circumsolar = 0.
    poa_horizon = 1.
    poa_circumsolar = 0.

    array.update_irradiance_terms_perez(
        solar_zenith, solar_azimuth, surface_tilt, surface_azimuth,
        dni, luminance_isotropic, luminance_circumsolar,
        poa_horizon, poa_circumsolar)

    expected_horizon_shading = np.array([90.25751984, 0.0974248])
    calculated_horizon_shading = array.surface_registry.query(
        'pvrow_index==0 and surface_side=="back"')[
            ['horizon_band_shading_pct', 'horizon_term']].values[0]
    calculated_no_horizon_shading = array.surface_registry.query(
        'pvrow_index==1 and surface_side=="back"')[
            ['horizon_band_shading_pct', 'horizon_term']].values[0]

    TOL = 1e-7
    np.testing.assert_allclose(
        expected_horizon_shading, calculated_horizon_shading,
        atol=0, rtol=TOL)
    np.testing.assert_allclose(
        np.array([0., 1.]), calculated_no_horizon_shading,
        atol=0, rtol=TOL)
