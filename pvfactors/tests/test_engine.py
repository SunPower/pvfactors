from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
from pvfactors.irradiance import IsotropicOrdered, HybridPerezOrdered
from pvfactors.irradiance.utils import breakup_df_inputs
from pvfactors.report import example_fn_build_report_fast_mode
import numpy as np
import datetime as dt


def test_pvengine_float_inputs_iso(params):
    """Test that PV engine works for float inputs"""

    irradiance_model = IsotropicOrdered()
    pvarray = OrderedPVArray.init_from_dict(params)
    eng = PVEngine(pvarray, irradiance_model=irradiance_model)

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
    pvarray = OrderedPVArray.init_from_dict(params)
    eng = PVEngine(pvarray, irradiance_model=irradiance_model)

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
        pvarray.pvrows[0].front.get_param_weighted('qinc'), 1110.1164773159298)
    np.testing.assert_almost_equal(
        pvarray.pvrows[1].front.get_param_weighted('qinc'), 1110.595903991)
    np.testing.assert_almost_equal(
        pvarray.pvrows[2].front.get_param_weighted('qinc'), 1112.37717553)
    np.testing.assert_almost_equal(
        pvarray.pvrows[1].back.get_param_weighted('qinc'), 116.49050349491208)


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
    pvarray = OrderedPVArray.init_from_dict(
        params_serial, param_names=irradiance_model.params)
    eng = PVEngine(pvarray, irradiance_model=irradiance_model)

    # Fit engine
    eng.fit(timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth, albedo)

    # Run all timesteps
    report = eng.run_all_timesteps(fn_build_report=fn_report_example)

    # Check values
    np.testing.assert_array_almost_equal(
        report['qinc_front'], [1066.272392, 1065.979824])
    np.testing.assert_array_almost_equal(
        report['qinc_back'], [135.897106, 136.01297])
    np.testing.assert_array_almost_equal(
        report['iso_front'], [42.816637, 42.780206])
    np.testing.assert_array_almost_equal(
        report['iso_back'], [1.727308, 1.726535])


def test_fast_pvengine_float_inputs_perez(params):
    """Test that PV engine works for float inputs"""

    # Prepare some engine inputs
    irradiance_model = HybridPerezOrdered()
    pvarray = OrderedPVArray.init_from_dict(
        params, param_names=irradiance_model.params)
    fast_mode_pvrow_index = 1

    # Create engine object
    eng = PVEngine(pvarray, irradiance_model=irradiance_model,
                   fast_mode_pvrow_index=fast_mode_pvrow_index)

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
        pvarray.pvrows[1].back.get_param_weighted('qinc'), 123.7087347744459)


def test_run_fast_mode(params):
    """Test that PV engine works for timeseries fast mode and float inputs.
    Value is very close to loop-like fast mode"""

    # Prepare some engine inputs
    irradiance_model = HybridPerezOrdered()
    pvarray = OrderedPVArray.init_from_dict(
        params, param_names=irradiance_model.params)
    fast_mode_pvrow_index = 1
    fast_mode_segment_index = 0

    # Create engine object
    eng = PVEngine(pvarray, irradiance_model=irradiance_model,
                   fast_mode_pvrow_index=fast_mode_pvrow_index)

    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'], params['solar_azimuth'],
            params['surface_tilt'], params['surface_azimuth'],
            params['rho_ground'])
    # Checks
    np.testing.assert_almost_equal(eng.irradiance.direct['front_pvrow'], DNI)

    # Run fast mode
    qinc = eng.run_fast_back_pvrow(
        fn_build_report=lambda pvarray: (pvarray.ts_pvrows[1]
                                         .back.get_param_weighted('qinc')))
    # Check results
    np.testing.assert_allclose(qinc, 123.753462)

    # Without providing segment index
    eng.fast_mode_segment_index = None
    qinc = eng.run_fast_back_pvrow(
        fn_build_report=lambda pvarray: (pvarray.ts_pvrows[1]
                                         .back.get_param_weighted('qinc')))
    # Check results
    np.testing.assert_allclose(qinc, 123.753462)

    # Check the left pv row back side (no obstruction)
    # This is exactly the same calculated value as in loop-like fast mode
    # because ground is not infinite for back of left pv row
    qinc = eng.run_fast_back_pvrow(
        fn_build_report=lambda pvarray: (pvarray.ts_pvrows[0]
                                         .back.get_param_weighted('qinc')),
        pvrow_index=0)
    # Check results: value taken from loop-like fast mode
    np.testing.assert_allclose(qinc, 138.10421248631152)


def test_run_fast_mode_back_shading(params):
    """Test that PV engine works for timeseries fast mode and float inputs,
    and when there's large direct shading on the back surface.
    Value is very close to loop-style fast mode"""

    params.update({'gcr': 0.6, 'surface_azimuth': 270, 'surface_tilt': 120,
                   'solar_zenith': 70.})
    # Prepare some engine inputs
    irradiance_model = HybridPerezOrdered()
    pvarray = OrderedPVArray.init_from_dict(
        params, param_names=irradiance_model.params)
    fast_mode_pvrow_index = 1
    fast_mode_segment_index = 0

    # Create engine object
    eng = PVEngine(pvarray, irradiance_model=irradiance_model,
                   fast_mode_pvrow_index=fast_mode_pvrow_index,
                   fast_mode_segment_index=fast_mode_segment_index)

    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    # Expected values
    expected_qinc = 683.537153

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'], params['solar_azimuth'],
            params['surface_tilt'], params['surface_azimuth'],
            params['rho_ground'])

    # By providing segment index
    qinc = eng.run_fast_back_pvrow(
        fn_build_report=example_fn_build_report_fast_mode)
    # Check results
    np.testing.assert_allclose(qinc, expected_qinc)

    # Without providing segment index
    eng.fast_mode_segment_index = None
    qinc = eng.run_fast_back_pvrow(
        fn_build_report=example_fn_build_report_fast_mode)
    # Check results
    np.testing.assert_allclose(qinc, expected_qinc)


def test_fast_mode_8760(params, df_inputs_clearsky_8760):
    """Test fast mode with 1 PV row to make sure that is consistent
    after the vectorization update: should get exact same values
    """
    # Get MET data
    df_inputs = df_inputs_clearsky_8760
    timestamps = df_inputs.index
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    surface_azimuth = df_inputs.surface_azimuth.values
    # Run simulation for only 1 PV row
    params.update({'n_pvrows': 1})

    # Run engine
    pvarray = OrderedPVArray.init_from_dict(params)
    eng = PVEngine(pvarray)
    eng.fit(timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth, params['rho_ground'])
    qinc = eng.run_fast_back_pvrow(
        fn_build_report=lambda pvarray: (pvarray.ts_pvrows[0]
                                         .back.get_param_weighted('qinc')),
        pvrow_index=0)

    # Check than annual energy on back is consistent
    np.testing.assert_allclose(np.nansum(qinc) / 1e3, 349.9345317405013)
