from pvfactors.run import run_timeseries_engine, run_parallel_engine
from pvfactors.report import ExampleReportBuilder
from pvfactors.viewfactors.aoimethods import faoi_fn_from_pvlib_sandia
from pvfactors.viewfactors.calculator import VFCalculator
import numpy as np
import mock


def test_run_timeseries_engine(fn_report_example, params_serial,
                               df_inputs_clearsky_8760):
    """Test that running timeseries engine with full mode works consistently"""
    df_inputs = df_inputs_clearsky_8760.iloc[:24, :]
    n = df_inputs.shape[0]

    # Get MET data
    timestamps = df_inputs.index
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    surface_azimuth = df_inputs.surface_azimuth.values

    report = run_timeseries_engine(
        fn_report_example, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'])

    assert len(report['qinc_front']) == n
    # Test value consistency
    np.testing.assert_almost_equal(np.nansum(report['qinc_back']),
                                   541.7115807694377)
    np.testing.assert_almost_equal(np.nansum(report['iso_back']),
                                   18.050083142438311)
    # Check a couple values
    np.testing.assert_almost_equal(report['qinc_back'][7],
                                   11.160301350847325)
    np.testing.assert_almost_equal(report['qinc_back'][-8],
                                   8.642850754173368)


def test_run_timeseries_engine_fast_mode(fn_report_example, params_serial,
                                         df_inputs_clearsky_8760):
    """Test that running timeseries engine with fast mode works consistently.
    Values are supposed to be a little higher than with full mode"""
    df_inputs = df_inputs_clearsky_8760.iloc[:24, :]
    n = df_inputs.shape[0]

    # Get MET data
    timestamps = df_inputs.index
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    surface_azimuth = df_inputs.surface_azimuth.values
    fast_mode_pvrow_index = 1

    def fn_report(pvarray): return {
        'qinc_back': pvarray.ts_pvrows[1].back.get_param_weighted('qinc'),
        'iso_back': pvarray.ts_pvrows[1].back.get_param_weighted('isotropic')}

    report = run_timeseries_engine(
        fn_report, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'],
        fast_mode_pvrow_index=fast_mode_pvrow_index)

    assert len(report['qinc_back']) == n
    # Test value consistency
    np.testing.assert_almost_equal(np.nansum(report['qinc_back']),
                                   548.0011865481954)
    np.testing.assert_almost_equal(np.nansum(report['iso_back']),
                                   18.03732189070727)
    # Check a couple values
    np.testing.assert_almost_equal(report['qinc_back'][7],
                                   11.304105184587364)
    np.testing.assert_almost_equal(report['qinc_back'][-8],
                                   8.743201975668212)


def test_params_irradiance_model():
    """Test that irradiance params are passed correctly in
    run_timeseries_engine"""
    mock_irradiance_model = mock.MagicMock()
    mock_engine = mock.MagicMock()
    mock_pvarray = mock.MagicMock()
    irradiance_params = {'horizon_band_angle': 15.}

    _ = run_timeseries_engine(
        None, None,
        None, None, None, None, None, None,
        None, None, cls_engine=mock_engine, cls_pvarray=mock_pvarray,
        cls_irradiance=mock_irradiance_model,
        irradiance_model_params=irradiance_params)

    mock_irradiance_model.assert_called_once_with(
        horizon_band_angle=irradiance_params['horizon_band_angle'])


def test_run_parallel_engine_with_irradiance_params(params_serial,
                                                    df_inputs_clearsky_8760):
    """Test that irradiance model params are passed correctly in parallel
    simulations"""
    df_inputs = df_inputs_clearsky_8760.iloc[:24, :]

    # Get MET data
    timestamps = df_inputs.index
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    surface_azimuth = df_inputs.surface_azimuth.values
    n_processes = 2

    irradiance_params = {'horizon_band_angle': 6.5}
    report_no_params = run_parallel_engine(
        ExampleReportBuilder, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'], n_processes=n_processes,
        irradiance_model_params=irradiance_params)

    np.testing.assert_almost_equal(np.nansum(report_no_params['qinc_back']),
                                   541.7115807694377)

    # The incident irradiance should be higher with larger horizon band
    irradiance_params = {'horizon_band_angle': 15.}
    report_w_params = run_parallel_engine(
        ExampleReportBuilder, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'], n_processes=n_processes,
        irradiance_model_params=irradiance_params)

    np.testing.assert_almost_equal(np.nansum(report_w_params['qinc_back']),
                                   554.5333279555168)


def test_params_ghi_passed():
    """Test that GHI is passed correctly to run functions"""
    mock_irradiance_model = mock.MagicMock()
    mock_engine = mock.MagicMock()
    mock_pvarray = mock.MagicMock()
    ghi = [1000.]

    _ = run_timeseries_engine(
        None, None,
        None, None, None, None, None, None,
        None, None, cls_engine=mock_engine, cls_pvarray=mock_pvarray,
        cls_irradiance=mock_irradiance_model, ghi=ghi)

    mock_engine.return_value.fit.assert_called_with(
        None, None, None, None, None, None, None, None, ghi=ghi)


def test_run_parallel_engine_with_ghi(params_serial,
                                      df_inputs_clearsky_8760):
    """Test that ghi is correctly passed to models.
    Notes:
    - ghi is not used in full modes, so it will not affect the full mode results
    - so use fast mode instead
    """
    df_inputs = df_inputs_clearsky_8760.iloc[:24, :]

    # Get MET data
    timestamps = df_inputs.index
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values
    ghi = 500. * np.ones_like(dni)
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    surface_azimuth = df_inputs.surface_azimuth.values
    fast_mode_pvrow_index = 1
    n_processes = 2

    # Report with no ghi
    report_no_ghi = run_parallel_engine(
        TestFastReportBuilder, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'], n_processes=n_processes,
        fast_mode_pvrow_index=fast_mode_pvrow_index)

    np.testing.assert_almost_equal(np.nansum(report_no_ghi['qinc_back']),
                                   548.0011865481954)

    # Report with ghi
    report_w_ghi = run_parallel_engine(
        TestFastReportBuilder, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'], n_processes=n_processes,
        fast_mode_pvrow_index=fast_mode_pvrow_index, ghi=ghi)

    np.testing.assert_almost_equal(np.nansum(report_w_ghi['qinc_back']),
                                   771.8440422696128)


class TestFastReportBuilder(object):

    @staticmethod
    def build(pvarray):

        return {'qinc_back': pvarray.ts_pvrows[1].back
                .get_param_weighted('qinc').tolist()}

    @staticmethod
    def merge(reports):

        report = reports[0]
        keys = report.keys()
        for other_report in reports[1:]:
            for key in keys:
                report[key] += other_report[key]

        return report


def test_run_timeseries_faoi_fn(params_serial, pvmodule_canadian,
                                df_inputs_clearsky_8760):
    """Test that in run_timeseries function, faoi functions are used
    correctly"""
    # Prepare timeseries inputs
    df_inputs = df_inputs_clearsky_8760.iloc[:24, :]
    timestamps = df_inputs.index
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    surface_azimuth = df_inputs.surface_azimuth.values

    expected_qinc = 542.018551

    # --- Test without passing vf parameters
    # report function with test in it
    def report_fn_with_tests_no_faoi(pvarray):
        vf_aoi_matrix = pvarray.ts_vf_aoi_matrix
        pvrow = pvarray.ts_pvrows[0]
        list_back_pvrow_idx = [ts_surf.index for ts_surf
                               in pvarray.ts_pvrows[0].all_ts_surfaces]
        # Check that sum of vf_aoi is equal to reflectivity values
        # since no faoi_fn used
        np.testing.assert_allclose(
            vf_aoi_matrix[list_back_pvrow_idx, :, 12].sum(axis=1),
            [0.99, 0., 0.97, 0.])

        return {'qinc_back': pvrow.back.get_param_weighted('qinc'),
                'qabs_back': pvrow.back.get_param_weighted('qabs')}

    # create calculator
    report = run_timeseries_engine(
        report_fn_with_tests_no_faoi, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'],
        vf_calculator_params=None)

    np.testing.assert_allclose(np.nansum(report['qinc_back']), expected_qinc)
    np.testing.assert_allclose(np.nansum(report['qabs_back']), 525.757995)

    # --- Test when passing vf parameters
    # Prepare vf calc params
    faoi_fn = faoi_fn_from_pvlib_sandia(pvmodule_canadian)
    # the following is a very high number to get agreement in
    # integral sums between back and front surfaces
    n_sections = 10000
    vf_calc_params = {'faoi_fn_front': faoi_fn,
                      'faoi_fn_back': faoi_fn,
                      'n_aoi_integral_sections': n_sections}

    def report_fn_with_tests_w_faoi(pvarray):
        vf_aoi_matrix = pvarray.ts_vf_aoi_matrix
        pvrow = pvarray.ts_pvrows[0]

        list_back_pvrow_idx = [ts_surf.index for ts_surf
                               in pvrow.all_ts_surfaces]
        # Check that sum of vf_aoi is consistent
        np.testing.assert_allclose(
            vf_aoi_matrix[list_back_pvrow_idx, :, 12].sum(axis=1),
            [0.97102, 0., 0.971548, 0.], atol=0, rtol=1e-6)

        return {'qinc_back': pvrow.back.get_param_weighted('qinc'),
                'qabs_back': pvrow.back.get_param_weighted('qabs')}
    # create calculator
    report = run_timeseries_engine(
        report_fn_with_tests_w_faoi, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'],
        vf_calculator_params=vf_calc_params)

    np.testing.assert_allclose(np.nansum(report['qinc_back']), expected_qinc)
    np.testing.assert_allclose(np.nansum(report['qabs_back']), 522.299276)


def test_run_parallel_faoi_fn(params_serial, df_inputs_clearsky_8760):
    """Test that in run_parallel function, faoi functions are used
    correctly"""
    # Prepare timeseries inputs
    df_inputs = df_inputs_clearsky_8760.iloc[:24, :]
    timestamps = df_inputs.index
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    surface_azimuth = df_inputs.surface_azimuth.values

    expected_qinc = 542.018551
    # create calculator
    report = run_parallel_engine(
        TestFAOIReportBuilder, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'],
        vf_calculator_params=None)

    np.testing.assert_allclose(np.nansum(report['qinc_back']), expected_qinc)
    np.testing.assert_allclose(np.nansum(report['qabs_back']), 525.757995)

    # --- Test when passing vf parameters
    # the following is a very high number to get agreement in
    # integral sums between back and front surfaces
    n_sections = 10000
    vf_calc_params = {'faoi_fn_front': FaoiClass,
                      'faoi_fn_back': FaoiClass,
                      'n_aoi_integral_sections': n_sections}

    # create calculator
    report = run_parallel_engine(
        TestFAOIReportBuilder, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'],
        vf_calculator_params=vf_calc_params)

    np.testing.assert_allclose(np.nansum(report['qinc_back']), expected_qinc)
    np.testing.assert_allclose(np.nansum(report['qabs_back']), 522.299276)


class TestFAOIReportBuilder(object):

    @staticmethod
    def build(pvarray):
        return {'qinc_back': pvarray.ts_pvrows[0].back
                .get_param_weighted('qinc').tolist(),
                'qabs_back': pvarray.ts_pvrows[0].back
                .get_param_weighted('qabs').tolist()}

    @staticmethod
    def merge(reports):
        report = reports[0]
        keys = report.keys()
        for other_report in reports[1:]:
            for key in keys:
                report[key] += other_report[key]
        return report


class FaoiClass(object):

    @staticmethod
    def faoi(*args, **kwargs):
        fn = faoi_fn_from_pvlib_sandia('Canadian_Solar_CS5P_220M___2009_')
        return fn(*args, **kwargs)
