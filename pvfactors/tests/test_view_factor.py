# -*- coding: utf-8 -*-

"""
Test the calculation of the view factors
"""

from pvfactors.pvarray import Array
from pvfactors.timeseries import (calculate_radiosities_serially_perez,
                                  get_average_pvrow_outputs,
                                  breakup_df_inputs)
import numpy as np
import pandas as pd
import os
import logging

LOGGER = logging.getLogger()
HANDLER = logging.StreamHandler()
HANDLER.setLevel(logging.ERROR)
LOGGER.addHandler(HANDLER)

TEST_DIR = os.path.dirname(__file__)
TEST_DATA = os.path.join(TEST_DIR, 'test_files')
IDX_SLICE = pd.IndexSlice


def test_view_matrix():
    """
    Test that the view matrix provides the expected views between surfaces
    """
    # PV array parameters
    arguments = {
        'n_pvrows': 2,
        'pvrow_height': 1.5,
        'solar_zenith': 30,
        'solar_azimuth': 180.,
        'array_azimuth': 180.,
        'pvrow_width': 1.,
        'gcr': 0.3,
        'array_tilt': 20.
    }
    array = Array(**arguments)

    # Expected values of the view matrix
    expected_view_matrix = np.array([
        [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 8, 0, 8, 0, 10, 0, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 1],
        [9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 1],
        [0, 9, 0, 0, 0, 0, 0, 0, 0, 7, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 1],
        [0, 10, 6, 6, 6, 0, 6, 6, 6, 0, 0, 5],
        [0, 0, 6, 6, 6, 0, 6, 0, 6, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # Compare with expectations: make sure to remove the sky from the views
    n_shape = expected_view_matrix.shape[0]
    assert np.array_equal(array.view_matrix, expected_view_matrix)
    finite_surfaces_view_matrix = array.view_matrix[n_shape - 1:, n_shape - 1]
    # Make sure that the matrix is symmetric:
    # "if I can see you, you can see me"
    assert is_symmetric(finite_surfaces_view_matrix)


def test_view_factor_matrix():
    """
    Check that the calculation of view factors remains consistent
    """
    # PV array parameters
    arguments = {
        'n_pvrows': 2,
        'pvrow_height': 1.5,
        'solar_zenith': 30,
        'solar_azimuth': 180.,
        'array_azimuth': 180.,
        'pvrow_width': 1.0,
        'gcr': 0.4,
        'array_tilt': 30.
    }
    array = Array(**arguments)

    # Expected values
    expected_vf_matrix = np.array([
        [0., 0., 0., 0., 0., 0.06328772, 0., 0., 0., 0., 0., 0.93671228],
        [0., 0., 0., 0., 0., 0.0342757, 0., 0.01414893, 0., 0.05586108,
            0., 0.89571429],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.23662653,
            0.04159081, 0.72178267],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.04671626,
            0.23662653, 0.71665722],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.10675226,
            0.21878649, 0.67446125],
        [0.00064976, 0.0003519, 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0.99899834],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.00084338,
            0.0032498, 0.99590682],
        [0., 0.00565957, 0., 0., 0., 0., 0., 0., 0., 0.09305653,
            0., 0.9012839],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.28720098,
            0.00351237, 0.70928665],
        [0., 0.05586108, 0.27323278, 0.05394329, 0.14361376,
            0., 0.08101242, 0.23264133, 0.11107537, 0.,
            0., 0.04861999],
        [0., 0., 0.04802493, 0.27323278, 0.29433335,
            0., 0.31216481, 0., 0.00135841, 0.,
            0., 0.07088572],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    # Perform comparison using very small absolute tolerance
    tol = 1e-8
    assert np.allclose(array.vf_matrix, expected_vf_matrix, atol=tol, rtol=0)


def is_symmetric(matrix):
    """
    Helper function to check whether a matrix is symmetric or not

    :param matrix: inputted matrix
    :return: result of test
    """
    return (matrix.T == matrix).all()


def test_negativevf_and_flatcasenoon():

    pvarray_parameters = {
        'array_azimuth': 90,
        'array_tilt': 0.0,
        'gcr': 0.3,
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.0,
        'rho_back_pvrow': 0.03,
        'rho_front_pvrow': 0.01,
        'rho_ground': 0.2,
        'solar_azimuth': 90.0,
        'solar_zenith': 20.0
    }

    input_filename = 'file_test_negativevf_and_flatcasenoon.csv'
    df_inputs = pd.read_csv(os.path.join(TEST_DATA, input_filename),
                            index_col=0)
    df_inputs.index = pd.DatetimeIndex(df_inputs.index).tz_localize(
        'UTC').tz_convert('US/Arizona')

    # Break up inputs
    (timestamps, array_tilt, array_azimuth,
     solar_zenith, solar_azimuth, dni, dhi) = breakup_df_inputs(df_inputs)

    args = (pvarray_parameters, timestamps, solar_zenith, solar_azimuth,
            array_tilt, array_azimuth, dni, dhi)
    df_registries, _ = calculate_radiosities_serially_perez(args)
    df_outputs = get_average_pvrow_outputs(df_registries)

    vf_ipoa_front = df_outputs.loc[:, IDX_SLICE[1, 'front', 'qinc']]
    vf_ipoa_back = df_outputs.loc[:, IDX_SLICE[1, 'back', 'qinc']]

    # The model should calculate for all daytime points now since we fixed
    # the solar noon case (almost flat but not really), and we allowed
    # negative vf values early and late in the day
    expected_n_calculated_values = 13

    assert np.sum(vf_ipoa_front.notnull()) == expected_n_calculated_values
    assert np.sum(vf_ipoa_back.notnull()) == expected_n_calculated_values


def test_back_surface_luminance():
    """
    The model didn't calculate cases when the sun would hit the back surface
    because the perez model would return 0 circumsolar (not calculated for
    back surface). Fix was implemented, and this should check for it.
    """
    pvarray_parameters = {
        'array_azimuth': 90,
        'array_tilt': 0.0,
        'gcr': 0.3,
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.0,
        'rho_back_pvrow': 0.03,
        'rho_front_pvrow': 0.01,
        'rho_ground': 0.2,
        'solar_azimuth': 90.0,
        'solar_zenith': 20.0
    }

    input_filename = 'file_test_back_surface_luminance.csv'

    df_inputs = pd.read_csv(os.path.join(TEST_DATA, input_filename),
                            index_col=0)
    df_inputs.index = pd.DatetimeIndex(df_inputs.index).tz_localize(
        'UTC').tz_convert('US/Arizona')

    # Break up inputs
    (timestamps, array_tilt, array_azimuth,
     solar_zenith, solar_azimuth, dni, dhi) = breakup_df_inputs(df_inputs)

    args = (pvarray_parameters, timestamps, solar_zenith, solar_azimuth,
            array_tilt, array_azimuth, dni, dhi)
    df_registries, _ = calculate_radiosities_serially_perez(args)

    df_outputs = get_average_pvrow_outputs(df_registries)

    vf_ipoa_front = df_outputs.loc[:, IDX_SLICE[1, 'front', 'qinc']]
    vf_ipoa_back = df_outputs.loc[:, IDX_SLICE[1, 'back', 'qinc']]

    assert isinstance(vf_ipoa_front[0], float)
    assert isinstance(vf_ipoa_back[0], float)
