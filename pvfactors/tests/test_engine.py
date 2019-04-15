from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
from pvfactors.irradiance import IsotropicOrdered, HybridPerezOrdered
from pvfactors.irradiance.utils import breakup_df_inputs
import numpy as np
import datetime as dt


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
    pvarray = eng.run_timestep(0)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    np.testing.assert_almost_equal(
        pvarray.pvrows[0].front.get_param_weighted('qinc'), 1099.22245374)
    np.testing.assert_almost_equal(
        pvarray.pvrows[1].front.get_param_weighted('qinc'), 1099.6948573)
    np.testing.assert_almost_equal(
        pvarray.pvrows[2].front.get_param_weighted('qinc'), 1102.76149246)


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
    pvarray = eng.run_timestep(0)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    np.testing.assert_almost_equal(
        pvarray.pvrows[0].front.get_param_weighted('qinc'), 1122.38723433)
    np.testing.assert_almost_equal(
        pvarray.pvrows[1].front.get_param_weighted('qinc'), 1122.86505181)
    np.testing.assert_almost_equal(
        pvarray.pvrows[2].front.get_param_weighted('qinc'), 1124.93948059)


def test_pvengine_ts_inputs_perez(params_serial,
                                  df_inputs_serial_calculation,
                                  fn_report_example):
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

    # Run all timesteps
    report = eng.run_all_timesteps(fn_build_report=fn_report_example)

    # Check values
    np.testing.assert_array_almost_equal(
        report['qinc_front'], [1066.9716724773091, 1066.8000713864162])
    np.testing.assert_array_almost_equal(
        report['qinc_back'], [135.94887929717575, 136.07375526982389])
    np.testing.assert_array_almost_equal(
        report['iso_front'], [43.515922835469098, 43.600460681412407])
    np.testing.assert_array_almost_equal(
        report['iso_back'], [1.7555186253932977, 1.7596394859882367])
