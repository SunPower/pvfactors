"""Module containing specific irradiance modeling functions and tools.
More work probably needs to be done to improve the computation speed here, and
in theory a big part of ``perez_diffuse_luminance`` should be handled by pvlib
"""
import numpy as np
import pandas as pd
from pvlib import atmosphere, irradiance
from pvlib.tools import cosd, sind


def perez_diffuse_luminance(timestamps, surface_tilt, surface_azimuth,
                            solar_zenith, solar_azimuth, dni, dhi):
    """Function used to calculate the luminance and the view factor terms from the
    Perez diffuse light transposition model, as implemented in the
    ``pvlib-python`` library.
    This function was custom made to allow the calculation of the circumsolar
    component on the back surface as well. Otherwise, the ``pvlib``
    implementation would ignore it.

    Parameters
    ----------
    timestamps : array-like
        simulation timestamps
    surface_tilt : array-like
        Surface tilt angles in decimal degrees.
        surface_tilt must be >=0 and <=180.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)
    surface_azimuth : array-like
        The azimuth of the rotated panel,
        determined by projecting the vector normal to the panel's surface to
        the earth's surface [degrees].
    solar_zenith : array-like
        solar zenith angles
    solar_azimuth : array-like
        solar azimuth angles
    dni : array-like
        values for direct normal irradiance
    dhi : array-like
        values for diffuse horizontal irradiance

    Returns
    -------
    df_inputs : `pandas.DataFrame`
        Dataframe with the following columns:
        ['solar_zenith', 'solar_azimuth', 'surface_tilt', 'surface_azimuth',
        'dhi', 'dni', 'vf_horizon', 'vf_circumsolar', 'vf_isotropic',
        'luminance_horizon', 'luminance_circuqmsolar', 'luminance_isotropic',
        'poa_isotropic', 'poa_circumsolar', 'poa_horizon', 'poa_total_diffuse']

    """
    # Create a dataframe to help filtering on all arrays
    df_inputs = pd.DataFrame(
        {'surface_tilt': surface_tilt, 'surface_azimuth': surface_azimuth,
         'solar_zenith': solar_zenith, 'solar_azimuth': solar_azimuth,
         'dni': dni, 'dhi': dhi},
        index=pd.DatetimeIndex(timestamps))

    dni_et = irradiance.get_extra_radiation(df_inputs.index.dayofyear)
    am = atmosphere.get_relative_airmass(df_inputs.solar_zenith)

    # Need to treat the case when the sun is hitting the back surface of pvrow
    aoi_proj = irradiance.aoi_projection(
        df_inputs.surface_tilt, df_inputs.surface_azimuth,
        df_inputs.solar_zenith, df_inputs.solar_azimuth)
    sun_hitting_back_surface = ((aoi_proj < 0) &
                                (df_inputs.solar_zenith <= 90))
    df_inputs_back_surface = df_inputs.loc[sun_hitting_back_surface].copy()
    # Reverse the surface normal to switch to back-surface circumsolar calc
    df_inputs_back_surface.loc[:, 'surface_azimuth'] = (
        df_inputs_back_surface.loc[:, 'surface_azimuth'] - 180.)
    df_inputs_back_surface.loc[:, 'surface_azimuth'] = np.mod(
        df_inputs_back_surface.loc[:, 'surface_azimuth'], 360.
    )
    df_inputs_back_surface.loc[:, 'surface_tilt'] = (
        180. - df_inputs_back_surface.surface_tilt)

    if df_inputs_back_surface.shape[0] > 0:
        # Use recursion to calculate circumsolar luminance for back surface
        df_inputs_back_surface = perez_diffuse_luminance(
            *breakup_df_inputs(df_inputs_back_surface))

    # Calculate Perez diffuse components
    components = irradiance.perez(df_inputs.surface_tilt,
                                  df_inputs.surface_azimuth,
                                  df_inputs.dhi, df_inputs.dni,
                                  dni_et,
                                  df_inputs.solar_zenith,
                                  df_inputs.solar_azimuth,
                                  am,
                                  return_components=True)

    # Calculate Perez view factors:
    a = irradiance.aoi_projection(
        df_inputs.surface_tilt,
        df_inputs.surface_azimuth, df_inputs.solar_zenith,
        df_inputs.solar_azimuth)
    a = np.maximum(a, 0)
    b = cosd(df_inputs.solar_zenith)
    b = np.maximum(b, cosd(85))

    vf_perez = pd.DataFrame({
        'vf_horizon': sind(df_inputs.surface_tilt),
        'vf_circumsolar': a / b,
        'vf_isotropic': (1. + cosd(df_inputs.surface_tilt)) / 2.
    },
        index=df_inputs.index
    )

    # Calculate diffuse luminance
    luminance = pd.DataFrame(
        np.array([
            components['horizon'] / vf_perez['vf_horizon'],
            components['circumsolar'] / vf_perez['vf_circumsolar'],
            components['isotropic'] / vf_perez['vf_isotropic']
        ]).T,
        index=df_inputs.index,
        columns=['luminance_horizon', 'luminance_circumsolar',
                 'luminance_isotropic']
    )
    luminance.loc[components['sky_diffuse'] == 0, :] = 0.

    # Format components column names
    components = components.rename(columns={'isotropic': 'poa_isotropic',
                                            'circumsolar': 'poa_circumsolar',
                                            'horizon': 'poa_horizon'})

    df_inputs = pd.concat([df_inputs, components, vf_perez, luminance],
                          axis=1, join='outer')
    df_inputs = df_inputs.rename(columns={'sky_diffuse': 'poa_total_diffuse'})

    # Adjust the circumsolar luminance when it hits the back surface
    if df_inputs_back_surface.shape[0] > 0:
        df_inputs.loc[sun_hitting_back_surface, 'luminance_circumsolar'] = (
            df_inputs_back_surface.loc[:, 'luminance_circumsolar'])

    return df_inputs


def breakup_df_inputs(df_inputs):
    """Helper function: It is sometimes easier to provide a dataframe than a list
    of arrays: this function does the job of breaking up the dataframe into a
    list of expected 1-dim arrays

    Parameters
    ----------
    df_inputs : ``pandas.DataFrame``
        timestamp-indexed dataframe with following columns:
        'surface_azimuth', 'surface_tilt', 'solar_zenith', 'solar_azimuth',
        'dni', 'dhi'

    Returns
    -------
    all 1-dim ``numpy.ndarray``
        ``timestamps``, ``tilt_angles``, ``surface_azimuth``,
        ``solar_zenith``, ``solar_azimuth``, ``dni``, ``dhi``

    """
    timestamps = pd.to_datetime(df_inputs.index)
    surface_azimuth = df_inputs.surface_azimuth.values
    surface_tilt = df_inputs.surface_tilt.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values

    return (timestamps, surface_tilt, surface_azimuth,
            solar_zenith, solar_azimuth, dni, dhi)
