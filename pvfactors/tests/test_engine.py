from pvfactors.engine import PVEngine
from pvfactors.geometry.pvarray import OrderedPVArray
from pvfactors.irradiance import IsotropicOrdered, HybridPerezOrdered
from pvfactors.irradiance.utils import breakup_df_inputs
from pvfactors.viewfactors.aoimethods import faoi_fn_from_pvlib_sandia
from pvfactors.viewfactors.calculator import VFCalculator
import numpy as np
import datetime as dt
import pytest


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
    # Check absorbed
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].front.get_param_weighted('qabs'),
        1099.6948573 * 0.99)


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
    # Check absorbed irradiance
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[2].front.get_param_weighted('qabs'),
        1112.37717553 * 0.99)
    np.testing.assert_almost_equal(
        pvarray.ts_pvrows[1].back.get_param_weighted('qabs'),
        116.49050349491208 * 0.97)


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
    np.testing.assert_array_almost_equal(
        report['qabs_back'], report['qinc_back'] * 0.97)


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
    # [20.4971271991293, 21.389095477613356], which shows discontinuity.
    # Update 2021-10-04 for v1.5.2: for pvlib <0.9.0, expected value
    # was [20.497127, 20.50229]. The values changed slightly because
    # of a bugfix to the Perez model in pvlib 0.9.0. See pvfactors PR #121.
    expected_out = [20.936348, 20.942163]
    np.testing.assert_allclose(out, expected_out)


def test_check_tilt_zero_discontinuity():
    """
    Before version 1.5.2, surface_tilt=0 with certain combinations of
    surface_azimuth and axis_azimuth showed anomolous behavior where
    the irradiance at zero tilt was significantly different from the
    irradiance at very small but nonzero tilts.  Additionally, the
    calculated VF matrix could have values outside [0, 1].  See GH #125
    """
    # expected value calculated for surface_tilt=0.001, so should
    # not be significantly different from result for surface_tilt=0
    rear_qinc_expected = 76.10

    timestamps = np.array([dt.datetime(2019, 6, 1, 10)])
    solar_azimuth = np.array([135])
    solar_zenith = np.array([45])
    dni = np.array([200])
    dhi = np.array([400])
    albedo = np.array([0.2])
    surface_tilt = np.array([0.0])

    # the discontinuity did not occur for all combinations of
    # (surface_azimuth, axis_azimuth), so test all four "primary" pairs:
    for surface_azimuth in [90, 270]:
        surface_azimuth = np.array([surface_azimuth])
        for axis_azimuth in [0, 180]:
            params = dict(n_pvrows=3, axis_azimuth=axis_azimuth,
                          pvrow_height=2, pvrow_width=1, gcr=0.4)

            pvarray = OrderedPVArray.init_from_dict(params)
            eng = PVEngine(pvarray)
            eng.fit(timestamps, dni, dhi, solar_zenith, solar_azimuth,
                    surface_tilt, surface_azimuth, albedo)

            # Run simulation and get output
            eng.run_full_mode()
            out = pvarray.ts_pvrows[1].back.get_param_weighted('qinc')
            assert np.all(pvarray.ts_vf_matrix >= 0)
            assert np.all(pvarray.ts_vf_matrix <= 1)
            assert rear_qinc_expected == pytest.approx(out[0], abs=1e-2)


def test_create_engine_with_rho_init(params, pvmodule_canadian):
    """Check that can create PV engine with rho initialization
    from faoi functions"""
    # Create inputs
    pvarray = OrderedPVArray.init_from_dict(params)
    irradiance = HybridPerezOrdered(rho_front=None, rho_back=None)
    faoi_fn = faoi_fn_from_pvlib_sandia(pvmodule_canadian)
    vfcalculator = VFCalculator(faoi_fn_front=faoi_fn, faoi_fn_back=faoi_fn)
    # Create engine
    engine = PVEngine.with_rho_initialization(pvarray, vfcalculator,
                                              irradiance)
    # Check that rho values are the ones calculated
    np.testing.assert_allclose(engine.irradiance.rho_front, 0.02900688)
    np.testing.assert_allclose(engine.irradiance.rho_back, 0.02900688)


def test_engine_w_faoi_fn_in_irradiance_vfcalcs(params, pvmodule_canadian):
    """Run PV engine calcs with faoi functions for AOI losses"""

    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    pvarray = OrderedPVArray.init_from_dict(params)
    # create faoi function
    faoi_fn = faoi_fn_from_pvlib_sandia(pvmodule_canadian)
    # create vf_calculator with faoi function
    vfcalculator = VFCalculator(faoi_fn_front=faoi_fn, faoi_fn_back=faoi_fn)
    # create irradiance model with faoi function
    irradiance_model = HybridPerezOrdered(faoi_fn_front=faoi_fn,
                                          faoi_fn_back=faoi_fn)
    eng = PVEngine(pvarray, irradiance_model=irradiance_model,
                   vf_calculator=vfcalculator)

    # Make sure aoi methods are available
    assert eng.vf_calculator.vf_aoi_methods is not None

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
    np.testing.assert_allclose(
        pvarray.ts_pvrows[0].front.get_param_weighted('qinc'),
        1110.1164773159298)
    np.testing.assert_allclose(
        pvarray.ts_pvrows[1].front.get_param_weighted('qinc'), 1110.595903991)
    np.testing.assert_allclose(
        pvarray.ts_pvrows[2].front.get_param_weighted('qinc'), 1112.37717553)
    np.testing.assert_allclose(
        pvarray.ts_pvrows[1].back.get_param_weighted('qinc'),
        116.49050349491208)
    # Check absorbed irradiance: calculated using faoi functions
    np.testing.assert_allclose(
        pvarray.ts_pvrows[2].front.get_param_weighted('qabs'),
        [1109.1180884], rtol=1e-6)
    np.testing.assert_allclose(
        pvarray.ts_pvrows[1].back.get_param_weighted('qabs'),
        [114.2143503])


def test_engine_variable_albedo(params, df_inputs_clearsky_8760):
    """Run PV engine calcs with variable albedo"""
    n_points = 100  # limiting because circleci is taking very long

    irradiance_model = HybridPerezOrdered()
    pvarray = OrderedPVArray.init_from_dict(params)
    eng = PVEngine(pvarray, irradiance_model=irradiance_model)

    # Manage timeseries inputs
    df_inputs = df_inputs_clearsky_8760[:n_points]

    # Get MET data
    timestamps = df_inputs.index
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    surface_azimuth = df_inputs.surface_azimuth.values
    albedo = np.linspace(0.01, 1, num=n_points)

    # Fit engine
    eng.fit(timestamps, dni, dhi, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, albedo)

    # Run timestep
    pvarray = eng.run_full_mode(fn_build_report=lambda pvarray: pvarray)
    # Check the bifacial gain values
    pvrow = pvarray.ts_pvrows[1]
    bfg = (np.nansum(pvrow.back.get_param_weighted('qinc'))
           / np.nansum(pvrow.front.get_param_weighted('qinc'))) * 100.
    bfg_after_aoi = (np.nansum(pvrow.back.get_param_weighted('qabs'))
                     / np.nansum(pvrow.front.get_param_weighted('qabs'))
                     ) * 100.
    expected_bfg = 14.973198
    expected_bfg_after_aoi = 14.670709
    np.testing.assert_allclose(bfg, expected_bfg)
    np.testing.assert_allclose(bfg_after_aoi, expected_bfg_after_aoi)
