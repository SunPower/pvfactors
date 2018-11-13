# -*- coding: utf-8 -*-

"""
Test some of the functions in the tools module
"""
import pytest
from pvfactors.timeseries import (perez_diffuse_luminance,
                                  calculate_radiosities_serially_perez,
                                  calculate_radiosities_parallel_perez,
                                  get_average_pvrow_outputs,
                                  get_pvrow_segment_outputs,
                                  breakup_df_inputs,
                                  array_timeseries_calculate)
import numpy as np
import pandas as pd
import os

TEST_DIR = os.path.dirname(__file__)
TEST_DATA = os.path.join(TEST_DIR, 'test_files')
FILE_PATH = os.path.join(TEST_DATA, 'file_test_tools_inputs_clearday.csv')
idx_slice = pd.IndexSlice


@pytest.fixture(scope='function')
def mock_array_timeseries_calculate(mocker):
    return mocker.patch('pvfactors.timeseries.array_timeseries_calculate')


def test_array_calculate_timeseries():
    """
    Check that the timeseries results of the radiosity calculation using the
    isotropic diffuse sky approach stay consistent
    """
    # Simple sky and array configuration
    df_inputs = pd.DataFrame({
        'solar_zenith': [80., 20., 70.4407256],
        'solar_azimuth': [0., 180., 248.08690811],
        'tracker_theta': [70., 40., 42.4337927],
        'surface_azimuth': [180., 180., 270.],
        'dni': [1e3, 1e3, 1000.],
        'dhi': [1e2, 1e2, 100.]
    },
        columns=['solar_zenith', 'solar_azimuth', 'tracker_theta',
                 'surface_azimuth', 'dni', 'dhi'],
        index=[0, 1, 2]
    )
    arguments = {
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'gcr': 0.3,
    }

    # Break up inputs
    (timestamps, tracker_theta, surface_azimuth,
     solar_zenith, solar_azimuth, dni, dhi) = breakup_df_inputs(df_inputs)

    # Fill in the missing pieces
    luminance_isotropic = dhi
    luminance_circumsolar = np.zeros(len(timestamps))
    poa_horizon = np.zeros(len(timestamps))
    poa_circumsolar = np.zeros(len(timestamps))

    # Run timeseries calculation
    df_registries = array_timeseries_calculate(
        arguments, timestamps, solar_zenith, solar_azimuth,
        tracker_theta, surface_azimuth, dni, luminance_isotropic,
        luminance_circumsolar, poa_horizon, poa_circumsolar)

    # Calculate surface averages for pvrows
    df_outputs = get_average_pvrow_outputs(df_registries,
                                           values=['q0', 'qinc'],
                                           include_shading=False)

    # Check that the outputs are as expected
    expected_outputs_array = np.array([
        [31.60177482, 6.28906975, 3.58335581],
        [632.03549634, 125.78139505, 71.66711623],
        [2.27843869, 31.55401986, 28.05923971],
        [75.94795617, 1051.80066185, 935.30799022],
        [31.87339866, 6.3776871, 1.81431887],
        [637.46797317, 127.55374206, 36.28637745],
        [2.20476856, 31.21803306, 27.85790853],
        [73.49228524, 1040.60110204, 928.59695092],
        [46.7960208, 7.21518794, 2.16642175],
        [935.92041595, 144.30375888, 43.32843492],
        [2.29986192, 31.16722793, 27.77628919],
        [76.66206408, 1038.90759755, 925.87630648],
        [True, False, False]], dtype=object)
    tol = 1e-8
    np.testing.assert_allclose(expected_outputs_array[:-1, :].astype(float),
                               df_outputs.values.T,
                               atol=tol, rtol=0, equal_nan=True)


def test_perez_diffuse_luminance(df_perez_luminance):
    """
    Test that the calculation of luminance -- first step in using the vf model
    with Perez -- is functional
    """
    df_inputs = df_perez_luminance[['tracker_theta', 'surface_azimuth',
                                    'solar_zenith', 'solar_azimuth', 'dni',
                                    'dhi']]
    (timestamps, tracker_theta, surface_azimuth, solar_zenith, solar_azimuth,
     dni, dhi) = breakup_df_inputs(df_inputs)
    surface_tilt = np.abs(tracker_theta)
    df_outputs = perez_diffuse_luminance(timestamps, surface_tilt,
                                         surface_azimuth, solar_zenith,
                                         solar_azimuth, dni, dhi)

    col_order = df_outputs.columns
    expected_output = df_perez_luminance.rename(
        columns={'tracker_theta': 'surface_tilt'})
    expected_output['surface_tilt'] = np.abs(expected_output['surface_tilt'])
    tol = 1e-8
    np.testing.assert_allclose(df_outputs.values,
                               expected_output[col_order].values,
                               atol=0, rtol=tol)


def test_luminance_in_timeseries_calc(df_perez_luminance,
                                      mock_array_timeseries_calculate):
    """
    Test that the calculation of luminance -- first step in using the vf model
    with Perez -- is functional
    """
    df_inputs_clearday = pd.read_csv(FILE_PATH)
    df_inputs_clearday = df_inputs_clearday.set_index('datetime', drop=True)
    df_inputs_clearday.index = (pd.DatetimeIndex(df_inputs_clearday.index)
                                .tz_localize('UTC').tz_convert('Etc/GMT+7')
                                .tz_localize(None))

    # Break up inputs
    (timestamps, tracker_theta, surface_azimuth, solar_zenith, solar_azimuth,
     dni, dhi) = breakup_df_inputs(df_inputs_clearday)
    _, df_outputs = calculate_radiosities_serially_perez(
        (None, timestamps, solar_zenith, solar_azimuth,
         tracker_theta, surface_azimuth,
         dni, dhi))

    expected_output = df_perez_luminance.rename(
        columns={'tracker_theta': 'surface_tilt'})
    expected_output['surface_tilt'] = np.abs(expected_output['surface_tilt'])
    col_order = df_outputs.columns
    tol = 1e-8
    np.testing.assert_allclose(df_outputs.values,
                               expected_output[col_order].values,
                               atol=0, rtol=tol)


def test_save_all_outputs_calculate_perez():
    """
    Make sure that the serial and parallel calculations are able to save all
    the requested data on discretized segments (instead of averaging them by
    default). Check the consistency of the results.
    """
    # Load timeseries input data
    df_inputs_clearday = pd.read_csv(FILE_PATH)
    df_inputs_clearday = df_inputs_clearday.set_index('datetime', drop=True)
    df_inputs_clearday.index = (pd.DatetimeIndex(df_inputs_clearday.index)
                                .tz_localize('UTC').tz_convert('Etc/GMT+7')
                                .tz_localize(None))
    idx_subset = 10

    # PV array parameters for test
    arguments = {
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'gcr': 0.4,
        'rho_ground': 0.8,
        'rho_back_pvrow': 0.03,
        'rho_front_pvrow': 0.01,
        'cut': [(1, 3, 'front')]
    }

    # Break up inputs
    (timestamps, tracker_theta, surface_azimuth,
     solar_zenith, solar_azimuth, dni, dhi) = breakup_df_inputs(
         df_inputs_clearday.iloc[:idx_subset])

    args = (arguments, timestamps, solar_zenith, solar_azimuth,
            tracker_theta, surface_azimuth, dni, dhi)

    # Run the serial calculation
    df_registries_serial, _ = (
        calculate_radiosities_serially_perez(args))

    df_registries_parallel, _ = (
        calculate_radiosities_parallel_perez(*args))

    # Format the outputs
    df_outputs_segments_serial = get_pvrow_segment_outputs(
        df_registries_serial, values=['qinc'], include_shading=False)
    df_outputs_segments_parallel = get_pvrow_segment_outputs(
        df_registries_parallel, values=['qinc'], include_shading=False)

    # Load files with expected outputs
    expected_ipoa_dict_qinc = np.array([
        [842.54617681, 842.5566707, 842.43690951],
        [839.30179691, 839.30652961, 839.30906023],
        [839.17118956, 839.17513098, 839.17725568],
        [842.24679271, 842.26194393, 842.15463231]])

    # Perform the comparisons
    rtol = 1e-6
    atol = 0
    np.testing.assert_allclose(expected_ipoa_dict_qinc,
                               df_outputs_segments_serial.values,
                               atol=atol, rtol=rtol)
    np.testing.assert_allclose(expected_ipoa_dict_qinc,
                               df_outputs_segments_parallel.values,
                               atol=atol, rtol=rtol)


def test_get_average_pvrow_outputs(df_registries, df_outputs):
    """ Test that obtaining the correct format, using outputs from v011 """

    calc_df_outputs_num = get_average_pvrow_outputs(df_registries,
                                                    include_shading=False)
    calc_df_outputs_shading = get_average_pvrow_outputs(df_registries,
                                                        values=[],
                                                        include_shading=True)

    # print(calc_df_outputs_num)
    tol = 1e-8
    # Compare numerical values
    np.testing.assert_allclose(df_outputs.iloc[:, 1:].values,
                               calc_df_outputs_num.values,
                               atol=0, rtol=tol)

    array_is_shaded = calc_df_outputs_shading.sum(axis=1).values > 0
    # Compare bool on shading
    assert np.array_equal(df_outputs.iloc[:, 0].values,
                          array_is_shaded)


def test_get_average_pvrow_segments(df_registries, df_segments):
    """ Test that obtaining the correct format, using outputs from v011 """

    calc_df_segments = get_pvrow_segment_outputs(df_registries)

    # Get only numerical values and levels used by df_segments
    df_calc_num = (calc_df_segments
                   .select_dtypes(include=[np.number]))
    cols_calcs = df_calc_num.columns.droplevel(
        level=['pvrow_index', 'surface_side'])
    df_calc_num.columns = cols_calcs

    # Get col ordering of expected df_segments
    segment_index = df_segments.columns.get_level_values(
        'segment_index').astype(int)
    terms = df_segments.columns.get_level_values('irradiance_term')
    ordered_index = zip(segment_index, terms)
    # Re-order cols of calculated df segments
    df_calc_num = df_calc_num.loc[:, ordered_index].fillna(0.)
    # Compare arrays
    tol = 1e-8
    np.testing.assert_allclose(
        df_segments.values,
        df_calc_num.values,
        atol=0, rtol=tol)


def test_get_average_pvrow_outputs_nans(df_registries_with_nan, df_outputs):
    """ Test that obtaining the correct format, using outputs from v011 """

    calc_df_outputs_num = get_average_pvrow_outputs(
        df_registries_with_nan,
        include_shading=False)
    calc_df_outputs_shading = get_average_pvrow_outputs(
        df_registries_with_nan,
        values=[],
        include_shading=True)

    # Check the number of lines to make sure the nans were added
    assert calc_df_outputs_num.shape[0] == 3
    assert calc_df_outputs_shading.shape[0] == 3

    tol = 1e-8
    # Compare numerical values
    np.testing.assert_allclose(df_outputs.iloc[:, 1:].values,
                               calc_df_outputs_num.dropna(axis=0,
                                                          how='all').values,
                               atol=0, rtol=tol)

    array_is_shaded = calc_df_outputs_shading.dropna(
        axis=0, how='all').sum(axis=1).values > 0
    # Compare bool on shading
    assert np.array_equal(df_outputs.iloc[:, 0].values,
                          array_is_shaded)
