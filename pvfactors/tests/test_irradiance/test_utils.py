import numpy as np
from pvfactors.irradiance.utils import \
    perez_diffuse_luminance, breakup_df_inputs


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
