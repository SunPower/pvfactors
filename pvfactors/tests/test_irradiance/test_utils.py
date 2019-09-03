import numpy as np
from pvfactors.irradiance.utils import \
    perez_diffuse_luminance, breakup_df_inputs, \
    calculate_circumsolar_shading, calculate_horizon_band_shading


def test_perez_diffuse_luminance(df_perez_luminance):
    """
    Test that the calculation of luminance -- first step in using the vf model
    with Perez -- is functional
    """
    df_inputs = df_perez_luminance[['surface_tilt', 'surface_azimuth',
                                    'solar_zenith', 'solar_azimuth', 'dni',
                                    'dhi']]
    (timestamps, surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
     dni, dhi) = breakup_df_inputs(df_inputs)
    df_outputs = perez_diffuse_luminance(timestamps, surface_tilt,
                                         surface_azimuth, solar_zenith,
                                         solar_azimuth, dni, dhi)

    col_order = df_outputs.columns
    tol = 1e-8
    np.testing.assert_allclose(df_outputs.values,
                               df_perez_luminance[col_order].values,
                               atol=0, rtol=tol)


def test_calculate_circumsolar_shading():
    """
    Test that the disk shading function stays consistent
    """
    # Test for one value of 20% of the diameter being covered
    percentage_distance_covered = 20.
    percent_shading = calculate_circumsolar_shading(
        percentage_distance_covered, model='uniform_disk')

    # Compare to expected
    expected_disk_shading_perc = 14.2378489933
    atol = 0
    rtol = 1e-8
    np.testing.assert_allclose(expected_disk_shading_perc, percent_shading,
                               atol=atol, rtol=rtol)


def test_calculate_horizon_band_shading():
    """Test that calculation of horizon band shading percentage is correct """

    shading_angle = np.array([-10., 0., 3., 9., 20.])
    horizon_band_angle = 15.
    percent_shading = calculate_horizon_band_shading(shading_angle,
                                                     horizon_band_angle)
    expected_percent_shading = [0., 0., 20., 60., 100.]
    np.testing.assert_allclose(expected_percent_shading, percent_shading)
