from pvfactors.report import ExampleReportBuilder
import numpy as np
import pytest
import mock


def test_run_timeseries_engine(fn_report_example, params_serial,
                               df_inputs_clearsky_8760):

    from pvfactors.run import run_timeseries_engine
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


def test_run_parallel_engine(params_serial,
                             df_inputs_clearsky_8760):

    from pvfactors.run import run_parallel_engine
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
    report = run_parallel_engine(
        ExampleReportBuilder, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'], n_processes=n_processes)

    assert len(report['qinc_front']) == n
    # Test value consistency: not sure exactly why value is not exactly equal
    # to one in serial test, but difference not significant
    np.testing.assert_almost_equal(np.nansum(report['qinc_back']),
                                   541.7115807694377)
    np.testing.assert_almost_equal(np.nansum(report['iso_back']),
                                   18.050083142438311)
    # Check that the reports were sorted correctly
    np.testing.assert_almost_equal(report['qinc_back'][7],
                                   11.160301350847325)
    np.testing.assert_almost_equal(report['qinc_back'][-8],
                                   8.642850754173368)


@pytest.fixture(scope='function')
def mock_pool(mocker):
    yield mocker.patch('multiprocessing.Pool')


def test_params_irradiance_model(mock_pool):
    """Test that irradiance params are passed correctly in
    run_timeseries_engine and run_parallel_calculations"""
    mock_irradiance_model = mock.MagicMock()
    mock_engine = mock.MagicMock()
    irradiance_params = {'horizon_band_angle': 15.}

    from pvfactors.run import run_timeseries_engine, run_parallel_engine
    _ = run_timeseries_engine(
        None, None,
        None, None, None, None, None, None,
        None, None, cls_engine=mock_engine,
        cls_irradiance=mock_irradiance_model,
        irradiance_model_params=irradiance_params)

    mock_irradiance_model.assert_called_once_with(
        horizon_band_angle=irradiance_params['horizon_band_angle'])

    # Test parallel mode
    mock_irradiance_model = mock.MagicMock()
    mock_pool_obj = mock.MagicMock()
    mock_pool.return_value = mock_pool_obj
    report_builder = mock.MagicMock()
    _ = run_parallel_engine(
        report_builder, [],
        [], [], [], [], [], [],
        [], [], cls_engine=mock_engine,
        cls_irradiance=mock_irradiance_model,
        irradiance_model_params=irradiance_params,
        n_processes=1
    )

    mock_irradiance_model.assert_not_called()
    mock_pool.assert_called_once_with(1)
    # Check that irradiance params passed correctly
    zipped_elements = [item
                       for item in mock_pool_obj.map.call_args_list[0][0][1]]
    assert zipped_elements[0][-2] == irradiance_params
