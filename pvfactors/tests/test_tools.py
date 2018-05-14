# -*- coding: utf-8 -*-

"""
Test some of the functions in the tools module
"""

from pvfactors.pvarray import Array
from pvfactors.tools import (calculate_radiosities_serially_simple,
                             perez_diffuse_luminance,
                             calculate_radiosities_serially_perez,
                             calculate_radiosities_parallel_perez)
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
        columns=['solar_zenith', 'solar_azimuth', 'array_tilt', 'array_azimuth',
                 'dni', 'dhi'],
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
        [31.601748050014145, 6.289069752504206, 3.5833558115691035],
        [632.0349610002829, 125.78139505008411, 71.66711623138208],
        [2.2784386524603493, 31.554019855401, 28.05923970649779],
        [75.94795508201167, 1051.8006618467002, 935.3079902165931],
        [31.87339865348199, 6.377687102750911, 1.814318872353118],
        [637.4679730696398, 127.55374205501823, 36.286377447062364],
        [2.2047681015326277, 31.218033061227334, 27.857908527655677],
        [73.4922700510876, 1040.6011020409114, 928.596950921856],
        [46.79602079759552, 7.215187943800262, 2.1664217462458804],
        [935.9204159519105, 144.30375887600525, 43.328434924917616],
        [2.2998617834782267, 31.167227926438414, 27.776289194444438],
        [76.66205944927422, 1038.9075975479473, 925.8763064814813],
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
        df_inputs_clearday.ix[-1, 'array_azimuth'])
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

    # We want to save the results from the front side of tracker #2 (index 1)
    save_segments = (1, 'front')
    args = (arguments, df_inputs_clearday.iloc[:idx_subset], save_segments)

    # Run the serial calculation
    _, _, _, df_outputs_segments_serial = (
        calculate_radiosities_serially_perez(args))

    _, _, _, df_outputs_segments_parallel = (
        calculate_radiosities_parallel_perez(
            arguments, df_inputs_clearday.iloc[:idx_subset],
            save_segments=save_segments
        ))

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
                       df_outputs_segments_serial.loc[:, idx_slice['qinc', :]]
                       .values,
                       atol=atol, rtol=rtol)
    assert np.allclose(expected_ipoa_dict_qinc,
                       df_outputs_segments_parallel.loc[:,
                                                        idx_slice['qinc', :]]
                       .values,
                       atol=atol, rtol=rtol)
