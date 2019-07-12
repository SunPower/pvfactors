from pvfactors.run import run_timeseries_engine, run_parallel_engine
from pvfactors.report import ExampleReportBuilder
import numpy as np
import mock


def test_run_timeseries_engine(fn_report_example, params_serial,
                               df_inputs_clearsky_8760):

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


def test_params_irradiance_model():
    """Test that irradiance params are passed correctly in
    run_timeseries_engine"""
    mock_irradiance_model = mock.MagicMock()
    mock_engine = mock.MagicMock()
    irradiance_params = {'horizon_band_angle': 15.}

    _ = run_timeseries_engine(
        None, None,
        None, None, None, None, None, None,
        None, None, cls_engine=mock_engine,
        cls_irradiance=mock_irradiance_model,
        irradiance_model_params=irradiance_params)

    mock_irradiance_model.assert_called_once_with(
        horizon_band_angle=irradiance_params['horizon_band_angle'])


def test_run_parallel_engine_with_irradiance_params(params_serial,
                                                    df_inputs_clearsky_8760):
    """Test that irradiance model params are passed correctly in parallel
    simulations"""
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
