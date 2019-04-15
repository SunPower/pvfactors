# -*- coding: utf-8 -*-

from pvfactors.geometry import \
    ShadeCollection, PVSegment, PVSurface, PVRowSide, OrderedPVArray
import pytest
import os
import pandas as pd
import numpy as np
import datetime as dt
from collections import OrderedDict
import pvlib

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
        'axis_azimuth': 270,
        'tracker_theta': -20.,
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
        {'tracker_theta': 0.,
         'axis_azimuth': 0.,
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


######################################################################


@pytest.fixture(scope='function')
def pvsegments(shade_collections):
    seg_1 = PVSegment(
        illum_collection=shade_collections[0])
    seg_2 = PVSegment(
        shaded_collection=shade_collections[1])
    yield seg_1, seg_2


@pytest.fixture(scope='function')
def shade_collections():
    illum_col = ShadeCollection([PVSurface([(0, 0), (1, 0)], shaded=False)])
    shaded_col = ShadeCollection([PVSurface([(1, 0), (2, 0)], shaded=True)])
    yield illum_col, shaded_col


@pytest.fixture(scope='function')
def pvrow_side(pvsegments):
    side = PVRowSide(pvsegments)
    yield side


@pytest.fixture(scope='function')
def params():

    pvarray_parameters = {
        'n_pvrows': 3,
        'pvrow_height': 2.5,
        'pvrow_width': 2.,
        'surface_azimuth': 90.,  # east oriented modules
        'axis_azimuth': 0.,  # axis of rotation towards North
        'surface_tilt': 20.,
        'gcr': 0.4,
        'solar_zenith': 20.,
        'solar_azimuth': 90.,  # sun located in the east
        'rho_ground': 0.2,
        'rho_front_pvrow': 0.01,
        'rho_back_pvrow': 0.03
    }

    yield pvarray_parameters


@pytest.fixture(scope='function')
def discr_params():
    """Discretized parameters, should have 5 segments on front of first PV row,
    and 3 segments on back of second PV row"""
    params = {
        'n_pvrows': 3,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'surface_tilt': 20.,
        'surface_azimuth': 180.,
        'gcr': 0.4,
        'solar_zenith': 20.,
        'solar_azimuth': 90.,  # sun located in the east
        'axis_azimuth': 0.,  # axis of rotation towards North
        'rho_ground': 0.2,
        'rho_front_pvrow': 0.01,
        'rho_back_pvrow': 0.03,
        'cut': {0: {'front': 5}, 1: {'back': 3}}
    }
    yield params


@pytest.fixture(scope='function')
def params_direct_shading(params):
    params.update({'gcr': 0.6, 'surface_tilt': 60, 'solar_zenith': 60})
    yield params


@pytest.fixture(scope='function')
def ordered_pvarray(params):
    pvarray = OrderedPVArray.from_dict(params)
    yield pvarray


@pytest.fixture(scope='function')
def params_serial():
    arguments = {
        'n_pvrows': 2,
        'pvrow_height': 1.5,
        'pvrow_width': 1.,
        'axis_azimuth': 0.,
        'surface_tilt': 20.,
        'surface_azimuth': 90,
        'gcr': 0.3,
        'solar_zenith': 30.,
        'solar_azimuth': 90.,
        'rho_ground': 0.22,
        'rho_front_pvrow': 0.01,
        'rho_back_pvrow': 0.03
    }
    yield arguments


@pytest.fixture(scope='function')
def fn_report_example():
    def fn_report(report, pvarray):
        # Initialize the report
        if report is None:
            list_keys = ['qinc_front', 'qinc_back', 'iso_front', 'iso_back']
            report = OrderedDict({key: [] for key in list_keys})
        # Add elements to the report
        if pvarray is not None:
            pvrow = pvarray.pvrows[1]  # use center pvrow
            report['qinc_front'].append(
                pvrow.front.get_param_weighted('qinc'))
            report['qinc_back'].append(
                pvrow.back.get_param_weighted('qinc'))
            report['iso_front'].append(
                pvrow.front.get_param_weighted('isotropic'))
            report['iso_back'].append(
                pvrow.back.get_param_weighted('isotropic'))
        else:
            # No calculation was performed, because sun was down
            report['qinc_front'].append(np.nan)
            report['qinc_back'].append(np.nan)
            report['iso_front'].append(np.nan)
            report['iso_back'].append(np.nan)
        return report
    yield fn_report


def generate_tucson_clrsky_met_data():
    """Helper function to generate timeseries data, taken from pvlib
    documentation"""
    # Define site and timestamps
    latitude, longitude, tz, altitude = 32.2, -111, 'US/Arizona', 700
    times = pd.date_range(start='2019-01-01 01:00', end='2020-01-01',
                          freq='60Min', tz=tz)
    gcr = 0.3
    max_angle = 50
    # Calculate MET data
    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    apparent_zenith = solpos['apparent_zenith']
    azimuth = solpos['azimuth']
    airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
    pressure = pvlib.atmosphere.alt2pres(altitude)
    airmass = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, latitude,
                                                            longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    ineichen = pvlib.clearsky.ineichen(
        apparent_zenith, airmass, linke_turbidity, altitude, dni_extra)
    # Calculate single axis tracking data
    trk = pvlib.tracking.singleaxis(apparent_zenith, azimuth,
                                    max_angle=max_angle, gcr=gcr,
                                    backtrack=True)

    # Get outputs
    df_inputs = pd.concat(
        [ineichen[['dni', 'dhi']], solpos[['apparent_zenith', 'azimuth']],
         trk[['surface_tilt', 'surface_azimuth']]],
        axis=1).rename(columns={'apparent_zenith': 'solar_zenith',
                                'azimuth': 'solar_azimuth'})

    print(df_inputs.head())

    df_inputs.to_csv('test_df_inputs_MET_clearsky_tucson.csv')


@pytest.fixture(scope='function')
def df_inputs_clearsky_8760():
    tz = 'US/Arizona'
    fp = os.path.join(DIR_TEST_DATA, 'test_df_inputs_MET_clearsky_tucson.csv')
    df = pd.read_csv(fp, index_col=0)
    df.index = pd.DatetimeIndex(df.index).tz_localize('UTC').tz_convert(tz)
    yield df
