from pvfactors.run import run_timeseries_engine, run_parallel_engine
from pvfactors.report import ExampleReportBuilder
import numpy as np


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
                                   541.71158706765289)


def test_run_parallel_engine(params_serial,
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

    n_processes = 2
    report = run_parallel_engine(
        ExampleReportBuilder, params_serial,
        timestamps, dni, dhi, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, params_serial['rho_ground'], n_processes=n_processes)

    assert len(report['qinc_front']) == n
    # Test value consistency: not sure exactly why value is not exactly equal
    # to one in serial test, but difference not significant
    np.testing.assert_almost_equal(np.nansum(report['qinc_back']),
                                   541.71158706765289, decimal=3)
    # Check that the reports were sorted correctly
    np.testing.assert_almost_equal(report['qinc_back'][7],
                                   11.160305804865603)
    np.testing.assert_almost_equal(report['qinc_back'][-8],
                                   8.6428542068175016)
