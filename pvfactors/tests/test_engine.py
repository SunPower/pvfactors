from pvfactors.engine import PVEngine
from pvfactors.geometry.pvarray import OrderedPVArray
from pvfactors.irradiance import IsotropicOrdered, HybridPerezOrdered
from pvfactors.irradiance.utils import breakup_df_inputs
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
    np.testing.assert_almost_equal(eng.irradiance.direct['front_illum_pvrow'],
                                   DNI)

    # Run timestep
    pvarray = eng.run_full_mode(fn_build_report=lambda pvarray: pvarray)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].front.get_param_weighted('qinc'), 1099.22245374)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.get_param_weighted('qinc'), 1099.6948573)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[2].front.get_param_weighted('qinc'), 1102.76149246)


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
    np.testing.assert_almost_equal(eng.irradiance.direct['front_illum_pvrow'],
                                   DNI)

    # Run timestep
    pvarray = eng.run_full_mode(fn_build_report=lambda pvarray: pvarray)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[0].front.get_param_weighted('qinc'),
        1110.1164773159298)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.get_param_weighted('qinc'), 1110.595903991)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[2].front.get_param_weighted('qinc'), 1112.37717553)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.get_param_weighted('qinc'),
        116.49050349491208)


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
    report = eng.run_full_mode(fn_build_report=fn_report_example)

    # Check values
    np.testing.assert_array_almost_equal(
        report['qinc_front'], [1066.272392, 1065.979824])
    np.testing.assert_array_almost_equal(
        report['qinc_back'], [135.897106, 136.01297])
    np.testing.assert_array_almost_equal(
        report['iso_front'], [42.816637, 42.780206])
    np.testing.assert_array_almost_equal(
        report['iso_back'], [1.727308, 1.726535])


def test_run_fast_mode_isotropic(params):
    """Test that PV engine works for timeseries fast mode and float inputs,
    and using the isotropic irradiance model"""

    # Prepare some engine inputs
    irradiance_model = IsotropicOrdered()
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

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'], params['solar_azimuth'],
            params['surface_tilt'], params['surface_azimuth'],
            params['rho_ground'])
    # Checks
    np.testing.assert_almost_equal(eng.irradiance.direct['front_illum_pvrow'],
                                   DNI)

    # Expected value
    qinc_expected = 122.73453

    # Run fast mode
    qinc = eng.run_fast_mode(
        fn_build_report=lambda pvarray: (pvarray.ts_pvrows[1]
                                         .back.get_param_weighted('qinc')))
    # Check results
    np.testing.assert_allclose(qinc, qinc_expected)

    # Without providing segment index
    eng.fast_mode_segment_index = None
    qinc = eng.run_fast_mode(
        fn_build_report=lambda pvarray: (pvarray.ts_pvrows[1]
                                         .back.get_param_weighted('qinc')))
    # Check results
    np.testing.assert_allclose(qinc, qinc_expected)


def test_run_fast_mode_perez(params):
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
                   fast_mode_pvrow_index=fast_mode_pvrow_index,
                   fast_mode_segment_index=fast_mode_segment_index)

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
    np.testing.assert_almost_equal(eng.irradiance.direct['front_illum_pvrow'],
                                   DNI)

    expected_back_qinc = 119.095285
    # Run fast mode
    qinc = eng.run_fast_mode(
        fn_build_report=lambda pvarray: (pvarray.ts_pvrows[1]
                                         .back.get_param_weighted('qinc')))
    # Check results
    np.testing.assert_allclose(qinc, expected_back_qinc)

    # Without providing segment index
    eng.fast_mode_segment_index = None
    qinc = eng.run_fast_mode(
        fn_build_report=lambda pvarray: (pvarray.ts_pvrows[1]
                                         .back.get_param_weighted('qinc')))
    # Check results
    np.testing.assert_allclose(qinc, expected_back_qinc)


def test_run_fast_mode_segments(params):
    """Test that PV engine works for timeseries fast mode and float inputs.
    Value is very close to loop-like fast mode"""

    # Discretize middle PV row's back side
    params.update({'cut': {1: {'back': 5}}})

    # Prepare some engine inputs
    irradiance_model = HybridPerezOrdered()
    pvarray = OrderedPVArray.init_from_dict(
        params, param_names=irradiance_model.params)
    fast_mode_pvrow_index = 1
    fast_mode_segment_index = 2

    # Create engine object
    eng = PVEngine(pvarray, irradiance_model=irradiance_model,
                   fast_mode_pvrow_index=fast_mode_pvrow_index,
                   fast_mode_segment_index=fast_mode_segment_index)

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
    np.testing.assert_almost_equal(eng.irradiance.direct['front_illum_pvrow'],
                                   DNI)

    # Define report function to grab irradiance from PV row segment
    def fn_report(pvarray): return (pvarray.ts_pvrows[1].back.list_segments[2]
                                    .get_param_weighted('qinc'))

    # Expected value for middle segment
    qinc_expected = 116.572594
    # Run fast mode for specific segment
    qinc_segment = eng.run_fast_mode(fn_build_report=fn_report)
    # Check results
    np.testing.assert_allclose(qinc_segment, qinc_expected)

    # Without providing segment index: the value should be the same as above
    eng.fast_mode_segment_index = None
    qinc_segment = eng.run_fast_mode(fn_build_report=fn_report)
    # Check results
    np.testing.assert_allclose(qinc_segment, qinc_expected)


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

    def fn_report(pvarray): return (pvarray.ts_pvrows[1]
                                    .back.get_param_weighted('qinc'))
    # By providing segment index
    qinc = eng.run_fast_mode(fn_build_report=fn_report)
    # Check results
    np.testing.assert_allclose(qinc, expected_qinc)

    # Without providing segment index
    eng.fast_mode_segment_index = None
    qinc = eng.run_fast_mode(fn_build_report=fn_report)
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
    qinc = eng.run_fast_mode(
        fn_build_report=lambda pvarray: (pvarray.ts_pvrows[0]
                                         .back.get_param_weighted('qinc')),
        pvrow_index=0)

    # Check than annual energy on back is consistent
    np.testing.assert_allclose(np.nansum(qinc) / 1e3, 342.848005)


def test_run_fast_and_full_modes_sequentially(params, fn_report_example):
    """Make sure that can run fast and full modes one after the other
    without making the engine crash"""

    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    # Prepare some engine inputs
    pvarray = OrderedPVArray.init_from_dict(params)
    fast_mode_pvrow_index = 1
    fast_mode_segment_index = 0

    # Create engine object
    eng = PVEngine(pvarray, fast_mode_pvrow_index=fast_mode_pvrow_index,
                   fast_mode_segment_index=fast_mode_segment_index)
    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'], params['solar_azimuth'],
            params['surface_tilt'], params['surface_azimuth'],
            params['rho_ground'])

    # Run fast mode
    def fn_report(pvarray): return (pvarray.ts_pvrows[1]
                                    .back.get_param_weighted('qinc'))
    qinc_fast = eng.run_fast_mode(fn_build_report=fn_report)
    # Run full mode
    report = eng.run_full_mode(fn_build_report=fn_report_example)

    np.testing.assert_allclose(qinc_fast, 119.095285)
    np.testing.assert_allclose(report['qinc_back'], 116.49050349491)


def test_pvengine_float_inputs_perez_transparency_spacing_full(params):
    """Test that module transparency and spacing are having the
    expected effect to calculated PV back side irradiance"""

    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    # --- with 0 transparency and spacing
    # Create models
    irr_params = {'module_transparency': 0.,
                  'module_spacing_ratio': 0.}
    irradiance_model = HybridPerezOrdered(**irr_params)
    pvarray = OrderedPVArray.init_from_dict(params)
    eng = PVEngine(pvarray, irradiance_model=irradiance_model)

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'],
            params['solar_azimuth'],
            params['surface_tilt'],
            params['surface_azimuth'],
            params['rho_ground'])
    # Run timestep
    pvarray = eng.run_full_mode(fn_build_report=lambda pvarray: pvarray)
    no_spacing_transparency_back_qinc = (
        pvarray.ts_pvrows[1].back.get_param_weighted('qinc'))

    # --- with non-0 transparency and spacing
    # Create models
    irr_params = {'module_transparency': 0.1,
                  'module_spacing_ratio': 0.1}
    irradiance_model = HybridPerezOrdered(**irr_params)
    pvarray = OrderedPVArray.init_from_dict(params)
    eng = PVEngine(pvarray, irradiance_model=irradiance_model)

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'],
            params['solar_azimuth'],
            params['surface_tilt'],
            params['surface_azimuth'],
            params['rho_ground'])
    # Run timestep
    pvarray = eng.run_full_mode(fn_build_report=lambda pvarray: pvarray)
    # Checks
    expected_back_qinc = 132.13881181118185  # higher than when params are 0
    w_spacing_transparency_back_qinc = (
        pvarray.ts_pvrows[1].back.get_param_weighted('qinc'))
    np.testing.assert_almost_equal(
        w_spacing_transparency_back_qinc, expected_back_qinc)
    assert no_spacing_transparency_back_qinc < w_spacing_transparency_back_qinc


def test_pvengine_float_inputs_perez_transparency_spacing_fast(params):
    """Test that module transparency and spacing are having the
    expected effect to calculated PV back side irradiance"""

    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    # --- with 0 transparency and spacing
    # Create models
    irr_params = {'module_transparency': 0.,
                  'module_spacing_ratio': 0.}
    irradiance_model = HybridPerezOrdered(**irr_params)
    pvarray = OrderedPVArray.init_from_dict(params)
    eng = PVEngine(pvarray, irradiance_model=irradiance_model)

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'],
            params['solar_azimuth'],
            params['surface_tilt'],
            params['surface_azimuth'],
            params['rho_ground'])
    # Run timestep
    def fn_report(pvarray): return (pvarray.ts_pvrows[1]
                                    .back.get_param_weighted('qinc'))
    no_spacing_transparency_back_qinc = \
        eng.run_fast_mode(fn_build_report=fn_report, pvrow_index=1)

    # --- with non-0 transparency and spacing
    # Create models
    irr_params = {'module_transparency': 0.1,
                  'module_spacing_ratio': 0.1}
    irradiance_model = HybridPerezOrdered(**irr_params)
    pvarray = OrderedPVArray.init_from_dict(params)
    eng = PVEngine(pvarray, irradiance_model=irradiance_model)

    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'],
            params['solar_azimuth'],
            params['surface_tilt'],
            params['surface_azimuth'],
            params['rho_ground'])
    # Run timestep
    w_spacing_transparency_back_qinc = \
        eng.run_fast_mode(fn_build_report=fn_report, pvrow_index=1)
    # Checks
    expected_back_qinc = 134.7143531  # higher than when params are 0
    np.testing.assert_almost_equal(
        w_spacing_transparency_back_qinc, expected_back_qinc)
    assert no_spacing_transparency_back_qinc < w_spacing_transparency_back_qinc


def test_check_direct_shading_continuity():
    """Make sure the calculation is correct when direct shading happens.
    - before v1.3.0, there's a discontinuity (big jump) in prediction
    when direct shading happens. The values are the same as >=v1.3.0 for
    no direct shading, but they are different with direct shading.
    - starting at v1.3.0, the values are more continuous (no big change) when
    going from no direct shading to direct shading, which means it's
    most certainly a better implementation.

    The issue before v1.3.0 could be due to the fact that shadows are merged,
    and there might be a piece of geometry lost there, but not entirely sure.
    Since it's still relatively small, will not dig further.

    Here, we're testing the outputs at 2 timestamps, right before and
    right after direct shading, when varying only solar zenith.
    """

    # Prepare inputs
    n = 2
    inputs = {
        'solar_zenith': [81.275, 81.276],  # right at the limit of direct shadg
        'solar_azimuth': [295.9557133] * n,
        'surface_tilt': [15.18714669] * n,
        'surface_azimuth': [270.] * n,
        'dni': [1000.] * n,
        'dhi': [100.] * n,
        'albedo': [0.2] * n,
        'times': [dt.datetime(2014, 6, 25, 3)] * n}
    # Array parameters
    params = {'n_pvrows': 3,
              'axis_azimuth': 0.0,
              'pvrow_height': 1.5,
              'pvrow_width': 2.5,
              'gcr': 0.4}

    # Create engine and fit to inputs
    pvarray = OrderedPVArray.init_from_dict(params)
    eng = PVEngine(pvarray)
    eng.fit(np.array(inputs['times']),
            np.array(inputs['dni']),
            np.array(inputs['dhi']),
            np.array(inputs['solar_zenith']),
            np.array(inputs['solar_azimuth']),
            np.array(inputs['surface_tilt']),
            np.array(inputs['surface_azimuth']),
            np.array(inputs['albedo']))

    # Check there we are indeed right at the limit of direct shading
    np.testing.assert_array_equal(pvarray.has_direct_shading, [False, True])

    # Run simulation and get output
    pvarray = eng.run_full_mode(fn_build_report=lambda pvarray: pvarray)
    out = pvarray.ts_pvrows[1].back.get_param_weighted('qinc')

    # Check expected outputs: before v1.3.0, expected output is
    # [20.4971271991293, 21.389095477613356], which shows discontinuity
    expected_out = [20.497127, 20.50229]
    np.testing.assert_allclose(out, expected_out)
