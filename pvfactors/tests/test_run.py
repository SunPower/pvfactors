from pvfactors.run import run_timeseries_engine, run_parallel_engine
from pvfactors.report import ExampleReportBuilder
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
