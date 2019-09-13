from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
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
    pvarray = eng.run_full_mode_timestep(0)
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
    np.testing.assert_almost_equal(eng.irradiance.direct['front_illum_pvrow'],
                                   DNI)

    # Run timestep
    pvarray = eng.run_full_mode_timestep(0)
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


def test_fast_mode_loop_like(params):
    """Test value of older and decomissioned loop like fast mode"""

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
    eng.fit(timestamps, DNI, DHI, params['solar_zenith'],
            params['solar_azimuth'], params['surface_tilt'],
            params['surface_azimuth'], params['rho_ground'])
    # Checks
    np.testing.assert_almost_equal(eng.irradiance.direct['front_illum_pvrow'],
                                   DNI)

    # Run timestep
    pvarray = _fast_mode_with_loop(eng.pvarray, eng.irradiance,
                                   eng.vf_calculator, fast_mode_pvrow_index, 0)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    np.testing.assert_almost_equal(
        pvarray.pvrows[1].back.get_param_weighted('qinc'), 119.0505580124769)


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


def test_run_fast_mode_compare_to_loop_like(params):
    """Test that PV engine works for timeseries fast mode and float inputs.

    Check the left pv row back side (no obstruction, since rotation < 0)
    This should be exactly the same calculated value as in loop-like fast mode
    because ground is not infinite for back of left pv row"""

    # Prepare some engine inputs
    irradiance_model = HybridPerezOrdered()
    pvarray = OrderedPVArray.init_from_dict(
        params, param_names=irradiance_model.params)
    fast_mode_pvrow_index = 1
    # Irradiance inputs
    timestamps = dt.datetime(2019, 6, 11, 11)
    DNI = 1000.
    DHI = 100.

    # Create engine object
    eng = PVEngine(pvarray, irradiance_model=irradiance_model,
                   fast_mode_pvrow_index=fast_mode_pvrow_index)
    # Fit engine
    eng.fit(timestamps, DNI, DHI,
            params['solar_zenith'], params['solar_azimuth'],
            params['surface_tilt'], params['surface_azimuth'],
            params['rho_ground'])

    # Run baseline calculation
    pvrow_idx = 0
    time_idx = 0
    pvarray_loop = _fast_mode_with_loop(eng.pvarray, eng.irradiance,
                                        eng.vf_calculator, pvrow_idx,
                                        time_idx)
    # Expected should be: 138.10421248631152
    qinc_expected = pvarray_loop.pvrows[0].back.get_param_weighted('qinc')
    # Run timeseries calculation
    qinc = eng.run_fast_mode(
        fn_build_report=lambda pvarray: (pvarray.ts_pvrows[0]
                                         .back.get_param_weighted('qinc')),
        pvrow_index=0)
    # Check results: value taken from loop-like fast mode
    np.testing.assert_allclose(qinc, qinc_expected)


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


def _fast_mode_with_loop(pvarray, irradiance, vf_calculator, pvrow_idx, idx):
    """Function for running fast mode using subset of view factor matrix
    and with the shapely geometries.

    Saving this for future debugging of new timeseries fast mode.
    """
    # Transform pvarray to time step
    pvarray.transform(idx)

    # Apply irradiance terms to pvarray
    irradiance_vec, rho_vec, invrho_vec, total_perez_vec = \
        irradiance.get_full_modeling_vectors(pvarray, idx)

    # Prepare inputs to view factor calculator
    geom_dict = pvarray.dict_surfaces
    view_matrix, obstr_matrix = pvarray.view_obstr_matrices

    # Indices of the surfaces of the back of the selected pvrows
    list_surface_indices = pvarray.pvrows[pvrow_idx].back.surface_indices

    # Calculate view factors using a subset of view_matrix to
    # gain in calculation speed
    vf_matrix_subset = vf_calculator.get_vf_matrix_subset(
        geom_dict, view_matrix, obstr_matrix, pvarray.pvrows,
        list_surface_indices)
    pvarray.vf_matrix = vf_matrix_subset

    irradiance_vec_subset = irradiance_vec[list_surface_indices]
    # In fast mode, will not care to calculate q0
    qinc = vf_matrix_subset.dot(rho_vec * total_perez_vec) \
        + irradiance_vec_subset

    # Calculate other terms
    isotropic_vec = vf_matrix_subset[:, -1] * total_perez_vec[-1]
    reflection_vec = qinc - irradiance_vec_subset \
        - isotropic_vec

    # Update selected surfaces with values
    for i, surf_idx in enumerate(list_surface_indices):
        surface = geom_dict[surf_idx]
        surface.update_params({'qinc': qinc[i],
                               'isotropic': isotropic_vec[i],
                               'reflection': reflection_vec[i]})

    return pvarray


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
    pvarray = eng.run_full_mode_timestep(0)
    no_spacing_transparency_back_qinc = (
        pvarray.pvrows[1].back.get_param_weighted('qinc'))

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
    pvarray = eng.run_full_mode_timestep(0)
    # Checks
    expected_back_qinc = 132.13881181118185  # higher than when params are 0
    w_spacing_transparency_back_qinc = (
        pvarray.pvrows[1].back.get_param_weighted('qinc'))
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
