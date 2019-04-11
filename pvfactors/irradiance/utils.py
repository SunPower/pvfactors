"""Module containing specific irradiance modeling functions and tools.
More work probably needs to be done to improve the computation speed here, and
in theory a big part of ``perez_diffuse_luminance`` should be handled by pvlib
"""
import numpy as np
import pandas as pd
from pvlib import atmosphere, irradiance
from pvlib.tools import cosd, sind
import math
from pvfactors import PVFactorsError
from pvfactors.config import \
    SIGMA, N_SIGMA, GAUSSIAN_DIAMETER_CIRCUMSOLAR, RADIUS_CIRCUMSOLAR


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


def calculate_circumsolar_shading(percentage_distance_covered,
                                  model='uniform_disk'):
    """Select the model to calculate circumsolar shading based on the current PV
    array condition.

    Parameters
    ----------
    percentage_distance_covered : float
        this represents how much of the
        circumsolar diameter is covered by the neighboring row [in %]
    model : str, optional
        name of the circumsolar shading model to use:
        'uniform_disk' and 'gaussian' are the two models currently available
        (Default value = 'uniform_disk')

    Returns
    -------
    float
        shading percentage of circumsolar area

    """
    if model == 'uniform_disk':
        perc_shading = uniform_circumsolar_disk_shading(
            percentage_distance_covered)

    elif model == 'gaussian':
        perc_shading = gaussian_shading(percentage_distance_covered)

    else:
        raise PVFactorsError(
            'calculate_circumsolar_shading: model does not exist: ' +
            '%s' % model)

    return perc_shading


def integral_default_gaussian(y, x):
    """Calculate the value of the integral from x to y of the erf function

    Parameters
    ----------
    y : float
        upper limit
    x : float
        lower limit

    Returns
    -------
    float
        Calculated value of integral

    """
    return 0.5 * (math.erf(x) - math.erf(y))


def gaussian_shading(percentage_distance_covered):
    """Calculate circumsolar shading by assuming that the irradiance profile on
    a 2D section of the circumsolar disk is Gaussian

    Parameters
    ----------
    percentage_distance_covered : float
        [in %], proportion of the
        circumsolar disk area covered by the neighboring pvrow

    Returns
    -------
    float
        percentage shading in terms of irradiance (using gaussian profile)

    """
    if percentage_distance_covered < 0.:
        perc_shading = 0.
    elif percentage_distance_covered > 100.:
        perc_shading = 100.
    else:
        y = - N_SIGMA * SIGMA
        x = (y +
             percentage_distance_covered / 100. *
             GAUSSIAN_DIAMETER_CIRCUMSOLAR)
        area = integral_default_gaussian(y, x)
        total_gaussian_area = integral_default_gaussian(- N_SIGMA * SIGMA,
                                                        N_SIGMA * SIGMA)
        perc_shading = area / total_gaussian_area * 100.

    return perc_shading


def gaussian(x, mu=0., sigma=1.):
    """Gaussian density function

    Parameters
    ----------
    x : float
        argument of function
    mu : float, optional
        mean of the gaussian (Default value = 0.)
    sigma : float, optional
        standard deviation of the gaussian (Default value = 1.)

    Returns
    -------
    float
        value of guassian function at point x

    """
    return (1. / (sigma * np.sqrt(2. * np.pi)) *
            np.exp(- 0.5 * np.power((x - mu) / sigma, 2)))


def uniform_circumsolar_disk_shading(percentage_distance_covered):
    """Calculate the percentage shading of circumsolar irradiance. This
    model considers circumsolar to be a disk, and calculates the
    percentage shaded based on the percentage "distance covered",
    which is how much of the disk's diameter is covered by the
    neighboring object.

    Parameters
    ----------
    percentage_distance_covered : float
        distance covered of circumsolar disk diameter
        [% - values from 0 to 100]

    Returns
    -------
    float
        value of guassian function at point x

    """

    # Define a circumsolar disk
    r_circumsolar = RADIUS_CIRCUMSOLAR
    d_circumsolar = 2 * r_circumsolar
    area_circumsolar = np.pi * r_circumsolar**2.

    # Calculate circumsolar on case by case
    distance_covered = percentage_distance_covered / 100. * d_circumsolar

    if distance_covered <= 0:
        # No shading of circumsolar
        percent_shading = 0.
    elif (distance_covered > 0.) & (distance_covered <= r_circumsolar):
        # Theta is the full circle sector angle (not half) used to
        # calculate circumsolar shading
        theta = 2 * np.arccos((r_circumsolar - distance_covered) /
                              r_circumsolar)  # rad
        area_circle_sector = 0.5 * r_circumsolar ** 2 * theta
        area_triangle_sector = 0.5 * r_circumsolar ** 2 * np.sin(theta)
        area_shaded = area_circle_sector - area_triangle_sector
        percent_shading = area_shaded / area_circumsolar * 100.

    elif (distance_covered > r_circumsolar) & (distance_covered <
                                               d_circumsolar):
        distance_uncovered = d_circumsolar - distance_covered

        # Theta is the full circle sector angle (not half) used to
        # calculate circumsolar shading
        theta = 2 * np.arccos((r_circumsolar - distance_uncovered) /
                              r_circumsolar)  # rad
        area_circle_sector = 0.5 * r_circumsolar ** 2 * theta
        area_triangle_sector = 0.5 * r_circumsolar ** 2 * np.sin(theta)
        area_not_shaded = area_circle_sector - area_triangle_sector
        percent_shading = (1. - area_not_shaded / area_circumsolar) * 100.

    else:
        # Total shading of circumsolar: distance_covered >= d_circumsolar
        percent_shading = 100.

    return percent_shading


def calculate_horizon_band_shading(shading_angle, horizon_band_angle):
    """Calculate the percentage shading of the horizon band; which is just the
    proportion of what's masked by the neighboring row since we assume
    uniform luminance values for the band

    Parameters
    ----------
    shading_angle : float
        the elevation angle to use for shading
    horizon_band_angle : float
        the total elevation angle of the horizon band

    Returns
    -------
    float
        shading percentage of horizon band

    """
    percent_shading = 0.
    if shading_angle >= horizon_band_angle:
        percent_shading = 100.
    elif (shading_angle >= 0) | (shading_angle < horizon_band_angle):
        percent_shading = 100. * shading_angle / horizon_band_angle
    else:
        # No shading
        pass
    return percent_shading
