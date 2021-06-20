"""Contains the pytest fixtures for running tests"""

from pvfactors.geometry.base import ShadeCollection, PVSegment, PVSurface
from pvfactors.geometry.pvrow import PVRowSide
from pvfactors.geometry.pvarray import OrderedPVArray
import pytest
import os
import pandas as pd
import pvlib

DIR_TEST = os.path.dirname(__file__)
DIR_TEST_DATA = os.path.join(DIR_TEST, 'test_files')


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
def df_perez_luminance():
    """ Example of df_segments to be used for tests """
    fp = os.path.join(DIR_TEST_DATA, 'file_test_df_perez_luminance.csv')
    df_perez_luminance = pd.read_csv(fp, header=[0], index_col=0)

    df_perez_luminance.index = (pd.DatetimeIndex(df_perez_luminance.index)
                                .tz_localize('UTC').tz_convert('Etc/GMT+7')
                                .tz_localize(None))
    yield df_perez_luminance


@pytest.fixture(scope='function')
def shade_collections():
    illum_col = ShadeCollection([PVSurface([(0, 0), (1, 0)], shaded=False)])
    shaded_col = ShadeCollection([PVSurface([(1, 0), (2, 0)], shaded=True)])
    yield illum_col, shaded_col


@pytest.fixture(scope='function')
def pvsegments(shade_collections):
    seg_1 = PVSegment(
        illum_collection=shade_collections[0])
    seg_2 = PVSegment(
        shaded_collection=shade_collections[1])
    yield [seg_1, seg_2]


# @pytest.fixture(scope='function')
# def pvrow_side(pvsegments):
#     side = PVRowSide(pvsegments)
#     yield side


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
        'cut': {0: {'front': 5}, 1: {'back': 3, 'front': 2}}
    }
    yield params


@pytest.fixture(scope='function')
def params_direct_shading(params):
    params.update({'gcr': 0.6, 'surface_tilt': 60, 'solar_zenith': 60})
    yield params


@pytest.fixture(scope='function')
def ordered_pvarray(params):
    pvarray = OrderedPVArray.fit_from_dict_of_scalars(params)
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
    def fn_report(pvarray): return {
        'qinc_front': pvarray.ts_pvrows[1].front.get_param_weighted('qinc'),
        'qinc_back': pvarray.ts_pvrows[1].back.get_param_weighted('qinc'),
        'qabs_back': pvarray.ts_pvrows[1].back.get_param_weighted('qabs'),
        'iso_front': pvarray.ts_pvrows[1]
        .front.get_param_weighted('isotropic'),
        'iso_back': pvarray.ts_pvrows[1].back.get_param_weighted('isotropic')}
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

    df_inputs.to_csv('file_test_inputs_MET_clearsky_tucson.csv')


@pytest.fixture(scope='function')
def df_inputs_clearsky_8760():
    tz = 'US/Arizona'
    fp = os.path.join(DIR_TEST_DATA,
                      'file_test_inputs_MET_clearsky_tucson.csv')
    df = pd.read_csv(fp, index_col=0)
    df.index = pd.DatetimeIndex(df.index).tz_convert(tz)
    yield df


@pytest.fixture(scope='function')
def pvmodule_canadian():
    """Example of pvlib PV module name to use in tests"""
    yield 'Canadian_Solar_CS5P_220M___2009_'
