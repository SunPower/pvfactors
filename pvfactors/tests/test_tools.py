# -*- coding: utf-8 -*-

"""
Test some of the functions in the tools module
"""

from pvfactors.pvarray import Array
from pvfactors.tools import (calculate_radiosities_serially_simple,
                             perez_diffuse_luminance,
                             calculate_radiosities_serially_perez,
                             calculate_radiosities_parallel_perez,
                             get_average_pvrow_outputs,
                             get_pvrow_segment_outputs)
import numpy as np
import pandas as pd
import os

TEST_DIR = os.path.dirname(__file__)
TEST_DATA = os.path.join(TEST_DIR, 'test_files')
FILE_PATH = os.path.join(TEST_DATA, 'file_test_tools_inputs_clearday.csv')
idx_slice = pd.IndexSlice


def test_calculate_radiosities_serially_simple():
    """
    Check that the results of the radiosity calculation using the isotropic
    diffuse sky stay consistent
    """
    # Simple sky and array configuration
    df_inputs = pd.DataFrame(
        np.array(
            [[80., 0., 70., 180., 1e3, 1e2],
             [20., 180., 40., 180., 1e3, 1e2],
             [70.4407256, 248.08690811, 42.4337927, 270., 1000., 100.]]),
        columns=['solar_zenith', 'solar_azimuth', 'array_tilt',
                 'array_azimuth', 'dni', 'dhi'],
        index=[0, 1, 2]
    )
    arguments = {
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'gcr': 0.3,
    }
    array = Array(**arguments)

    # Calculate irradiance terms
    df_outputs, df_bifacial_gains = (
        calculate_radiosities_serially_simple(array, df_inputs))

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
    assert np.allclose(expected_outputs_array[:-1, :].astype(float),
                       df_outputs.values[:-1, :].astype(float),
                       atol=tol, rtol=0, equal_nan=True)


def test_perez_diffuse_luminance():
    """
    Test that the calculation of luminance -- first step in using the vf model
    with Perez -- is functional
    """
    df_inputs_clearday = pd.read_csv(FILE_PATH)
    df_inputs_clearday = df_inputs_clearday.set_index('datetime', drop=True)
    df_inputs_clearday.index = (pd.DatetimeIndex(df_inputs_clearday.index)
                                .tz_localize('UTC').tz_convert('Etc/GMT+7')
                                .tz_localize(None))

    df_outputs = perez_diffuse_luminance(df_inputs_clearday)


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

    # Adjustment in angles needed: need to keep azimuth constant and change
    # tilt angle only
    df_inputs_clearday.loc[
        (df_inputs_clearday.solar_azimuth <= 180.), 'array_azimuth'] = (
            df_inputs_clearday.loc[:, 'array_azimuth'][-1])
    df_inputs_clearday.loc[
        (df_inputs_clearday.solar_azimuth <= 180.), 'array_tilt'] *= (-1)

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
    args = (arguments, df_inputs_clearday.iloc[:idx_subset])

    # Run the serial calculation
    df_registries_serial, _ = (
        calculate_radiosities_serially_perez(args))

    df_registries_parallel, _ = (
        calculate_radiosities_parallel_perez(
            arguments, df_inputs_clearday.iloc[:idx_subset]
        ))

    # Format the outputs
    df_outputs_segments_serial = get_pvrow_segment_outputs(
        df_registries_serial, values=['qinc'], include_shading=False)
    df_outputs_segments_parallel = get_pvrow_segment_outputs(
        df_registries_parallel, values=['qinc'], include_shading=False)

    # Load files with expected outputs
    expected_ipoa_dict_qinc = np.array([
        [842.43691838, 842.54795737, 842.52912932],
        [839.30539601, 839.30285394, 839.29810984],
        [839.17118976, 839.17513111, 839.17725576],
        [842.24681064, 842.26195526, 842.15463995]])

    # Perform the comparisons
    rtol = 1e-7
    atol = 0
    assert np.allclose(expected_ipoa_dict_qinc,
                       df_outputs_segments_serial.values,
                       atol=atol, rtol=rtol)
    assert np.allclose(expected_ipoa_dict_qinc,
                       df_outputs_segments_parallel.values,
                       atol=atol, rtol=rtol)


def test_get_average_pvrow_outputs(df_registries, df_outputs):
    """ Test that obtaining the correct format, using outputs from v011 """

    calc_df_outputs = get_average_pvrow_outputs(df_registries)

    tol = 1e-8
    # Compare numerical values
    assert np.allclose(df_outputs.iloc[:, 1:].values,
                       calc_df_outputs.iloc[:, 1:].values,
                       atol=0, rtol=tol)
    # Compare bool on shading
    assert np.array_equal(df_outputs.iloc[:, 0].values,
                          calc_df_outputs.iloc[:, 0].values)


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
    assert np.allclose(
        df_segments.values,
        df_calc_num.values,
        atol=0, rtol=tol)
