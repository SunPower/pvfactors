# -*- coding: utf-8 -*-

import pytest
import os
import pandas as pd
import numpy as np
import datetime as dt

DIR_TEST = os.path.dirname(__file__)
DIR_TEST_DATA = os.path.join(DIR_TEST, 'test_files')


@pytest.fixture(scope='function')
def df_outputs():
    """ Example of df_outputs to be used for tests """
    fp = os.path.join(DIR_TEST_DATA, 'file_test_df_outputs.csv')
    df_outputs = pd.read_csv(fp, header=[0, 1, 2], index_col=0)
    df_outputs.index = pd.to_datetime(df_outputs.index)

    yield df_outputs


@pytest.fixture(scope='function')
def df_registries():
    """ Example of df_registries to be used for tests """
    fp = os.path.join(DIR_TEST_DATA, 'file_test_df_registries.csv')
    df_registries = pd.read_csv(fp, header=[0], parse_dates=['timestamps'])
    yield df_registries


@pytest.fixture(scope='function')
def df_segments():
    """ Example of df_segments to be used for tests """
    fp = os.path.join(DIR_TEST_DATA, 'file_test_df_segments.csv')
    df_segments = pd.read_csv(fp, header=[0, 1], index_col=0)
    df_segments.index = pd.to_datetime(df_segments.index)
    yield df_segments


@pytest.fixture(scope='function')
def df_perez_luminance():
    """ Example of df_segments to be used for tests """
    fp = os.path.join(DIR_TEST_DATA, 'file_test_df_perez_luminance.csv')
    df_perez_luminance = pd.read_csv(fp, header=[0], index_col=0)

    df_perez_luminance.index = (pd.DatetimeIndex(df_perez_luminance.index)
                                .tz_localize('UTC').tz_convert('Etc/GMT+7')
                                .tz_localize(None))
    yield df_perez_luminance


@pytest.fixture(scope='function')
def df_inputs_serial_calculation():
    # Import simulation inputs for calculation
    filename = "file_test_multiprocessing_inputs.csv"
    df_inputs_simulation = pd.read_csv(os.path.join(DIR_TEST_DATA, filename),
                                       index_col=0)
    df_inputs_simulation.index = pd.DatetimeIndex(df_inputs_simulation.index)
    idx_subset = 10
    df_inputs_simulation = df_inputs_simulation.iloc[0:idx_subset, :]

    yield df_inputs_simulation


@pytest.fixture(scope='function')
def pvarray_parameters_serial_calc():
    arguments = {
        'n_pvrows': 2,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'array_azimuth': 270,
        'array_tilt': -20.,
        'gcr': 0.3,
        'solar_zenith': 30.,
        'solar_azimuth': 90.,
        'rho_ground': 0.22,
        'rho_pvrow_front': 0.01,
        'rho_pvrow_back': 0.03
    }
    yield arguments


@pytest.fixture(scope='function')
def df_inputs_serial_calculation_with_skips(
        df_inputs_serial_calculation):
    """ Create inputs that will lead to at least 1 skip in serial calc """

    df_skips = pd.DataFrame(
        {'array_tilt': 0.,
         'array_azimuth': 0.,
         'solar_zenith': 100.,  # the sun is down
         'solar_azimuth': 0.,
         'dni': np.nan, 'dhi': np.nan},
        index=[dt.datetime(2018, 9, 14, 22)])

    df_inputs = pd.concat([df_inputs_serial_calculation, df_skips], axis=0,
                          sort=True)

    yield df_inputs


@pytest.fixture(scope='function')
def df_registries_with_nan(df_registries):
    """ Example of df_registries to be used for tests """

    df_nan = pd.DataFrame(
        np.nan, columns=df_registries.columns, index=[100])
    df_nan.timestamps = dt.datetime(2018, 9, 14, 22)
    df_registries_with_nan = pd.concat([df_registries, df_nan],
                                       axis=0, sort=False)

    yield df_registries_with_nan
