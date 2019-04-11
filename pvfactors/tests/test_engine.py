from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
from pvfactors.irradiance import IsotropicOrdered, HybridPerezOrdered
from pvfactors.irradiance.utils import breakup_df_inputs
import numpy as np
import datetime as dt
import pytest


def test_pvengine_float_inputs_iso(params):
    """Test that PV engine works for float inputs"""

    irradiance_model = IsotropicOrdered()
    eng = PVEngine(params, irradiance_model=irradiance_model)

    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'],
            params['solar_azimuth'],
            params['surface_tilt'],
            params['surface_azimuth'],
            params['rho_ground'])
    # Checks
    np.testing.assert_almost_equal(eng.irradiance.direct['front_pvrow'], DNI)

    # Run timestep
    pvarray, vf_matrix, q0, qinc = eng.run_timestep(0)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    assert vf_matrix.shape[0] == pvarray.n_surfaces + 1
    print(pvarray.pvrows[0].front.get_param_weighted('qinc'))
    print(pvarray.pvrows[1].front.get_param_weighted('qinc'))
    print(pvarray.pvrows[2].front.get_param_weighted('qinc'))


def test_pvengine_float_inputs_perez(params):
    """Test that PV engine works for float inputs"""

    irradiance_model = HybridPerezOrdered()
    eng = PVEngine(params, irradiance_model=irradiance_model)

    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'],
            params['solar_azimuth'],
            params['surface_tilt'],
            params['surface_azimuth'],
            params['rho_ground'])
    # Checks
    np.testing.assert_almost_equal(eng.irradiance.direct['front_pvrow'], DNI)

    # Run timestep
    pvarray, vf_matrix, q0, qinc = eng.run_timestep(0)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    assert vf_matrix.shape[0] == pvarray.n_surfaces + 1
    print(pvarray.pvrows[0].front.get_param_weighted('qinc'))
    print(pvarray.pvrows[1].front.get_param_weighted('qinc'))
    print(pvarray.pvrows[2].front.get_param_weighted('qinc'))


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


def test_pvengine_ts_inputs_perez(params_serial,
                                  df_inputs_serial_calculation):
    """Test that PV engine works for timeseries inputs"""

    # Break up inputs
    (timestamps, surface_tilt, surface_azimuth,
     solar_zenith, solar_azimuth, dni, dhi) = breakup_df_inputs(
         df_inputs_serial_calculation)
    albedo = params_serial['rho_ground']

    # Create engine
    irradiance_model = HybridPerezOrdered()
    eng = PVEngine(params_serial,
                   irradiance_model=irradiance_model)

    # Fit engine
    eng.fit(timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth, albedo)

    # Run timestep
    eng.run_all_timesteps()
