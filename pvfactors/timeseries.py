# -*- coding: utf-8 -*-

from pvlib import atmosphere, irradiance
from pvlib.tools import cosd, sind
from pvlib.irradiance import aoi_projection
import numpy as np
import pandas as pd
from pvfactors import (logging, print_progress)
from pvfactors.pvarray import Array
from multiprocessing import Pool, cpu_count
import time
from copy import deepcopy


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


COLS_TO_SAVE = ['q0', 'qinc', 'circumsolar_term', 'horizon_term',
                'direct_term', 'irradiance_term', 'isotropic_term',
                'reflection_term', 'horizon_band_shading_pct']
DISTANCE_TOLERANCE = 1e-8
idx_slice = pd.IndexSlice


def array_timeseries_calculate(
    pvarray_parameters, timestamps, solar_zenith, solar_azimuth, surface_tilt,
        surface_azimuth, dni, luminance_isotropic, luminance_circumsolar,
        poa_horizon, poa_circumsolar):
    """
    Calculate the view factor radiosity and irradiance terms for multiple
    times.
    The function inputs assume a diffuse sky dome as represented in the Perez
    diffuse sky transposition model (from ``pvlib-python`` implementation).

    :param dict pvarray_parameters: parameters used to instantiate
        ``pvarray.Array`` class
    :param array-like timestamps: simulation timestamps
    :param array-like solar_zenith: solar zenith angles
    :param array-like solar_azimuth: solar azimuth angles
    :param array-like surface_tilt: Surface tilt angles in decimal degrees.
        surface_tilt must be >=0 and <=180.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)
    :param array-like surface_azimuth: The azimuth of the rotated panel,
        determined by projecting the vector normal to the panel's surface to
        the earth's surface [degrees].
    :param array-like dni: values for direct normal irradiance
    :param array-like luminance_isotropic: luminance of the isotropic sky dome
    :param array-like luminance_circumsolar: luminance of circumsolar area
    :param array-like poa_horizon: POA irradiance horizon component
    :param array-like poa_circumsolar: POA irradiance circumsolar component

    :return: ``df_registries``.
        Concatenated form of the ``pvarray.Array`` surface registries.
    :rtype: :class:`pandas.DataFrame`
    """
    # Instantiate array
    array = Array(**pvarray_parameters)
    # We want to save the whole registry for each timestamp
    list_registries = []
    # We want to record the skipped_ts
    skipped_ts = []
    # Use for printing progress
    n = len(timestamps)
    i = 1
    for idx, ts in enumerate(timestamps):
        try:
            if ((isinstance(solar_zenith[idx], float)) &
                    (solar_zenith[idx] <= 90.)):
                # Run calculation only if daytime
                array.calculate_radiosities_perez(
                    solar_zenith[idx], solar_azimuth[idx], surface_tilt[idx],
                    surface_azimuth[idx], dni[idx], luminance_isotropic[idx],
                    luminance_circumsolar[idx], poa_horizon[idx],
                    poa_circumsolar[idx])

                # Save the whole registry
                registry = deepcopy(array.surface_registry)
                registry['timestamps'] = ts
                list_registries.append(registry)
            else:
                skipped_ts.append(ts)

        except Exception as err:
            LOGGER.debug("Unexpected error: {0}".format(err))
            skipped_ts.append(ts)

        # Printing progress
        print_progress(i, n, prefix='Progress:', suffix='Complete',
                       bar_length=50)
        i += 1

    # Concatenate all surface registries into one dataframe
    if list_registries:
        if skipped_ts:
            df_skipped = pd.DataFrame(
                np.nan, columns=Array.registry_cols,
                index=range(len(skipped_ts))
            ).assign(timestamps=skipped_ts)
            list_registries.append(df_skipped)
        df_registries = (pd.concat(list_registries, axis=0, join='outer',
                                   sort=False)
                         .sort_values(by=['timestamps'])
                         .reset_index(drop=True))
    else:
        df_registries = pd.DataFrame(
            np.nan,
            columns=Array.registry_cols,
            index=range(len(timestamps))
        ).assign(timestamps=timestamps)

    return df_registries


def perez_diffuse_luminance(timestamps, surface_tilt, surface_azimuth,
                            solar_zenith, solar_azimuth, dni, dhi):
    """
    Function used to calculate the luminance and the view factor terms from the
    Perez diffuse light transposition model, as implemented in the
    ``pvlib-python`` library.
    This function was custom made to allow the calculation of the circumsolar
    component on the back surface as well. Otherwise, the ``pvlib``
    implementation would ignore it.

    :param array-like timestamps: simulation timestamps
    :param array-like surface_tilt: Surface tilt angles in decimal degrees.
        surface_tilt must be >=0 and <=180.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)
    :param array-like surface_azimuth: The azimuth of the rotated panel,
        determined by projecting the vector normal to the panel's surface to
        the earth's surface [degrees].
    :param array-like solar_zenith: solar zenith angles
    :param array-like solar_azimuth: solar azimuth angles
    :param array-like dni: values for direct normal irradiance
    :param array-like dhi: values for diffuse horizontal irradiance
    :return: ``df_inputs``, dataframe with the following columns:
        ['solar_zenith', 'solar_azimuth', 'surface_tilt', 'surface_azimuth',
        'dhi', 'dni', 'vf_horizon', 'vf_circumsolar', 'vf_isotropic',
        'luminance_horizon', 'luminance_circumsolar', 'luminance_isotropic',
        'poa_isotropic', 'poa_circumsolar', 'poa_horizon', 'poa_total_diffuse']
    :rtype: class:`pandas.DataFrame`
    """
    # Create a dataframe to help filtering on all arrays
    df_inputs = pd.DataFrame(
        {'surface_tilt': surface_tilt, 'surface_azimuth': surface_azimuth,
         'solar_zenith': solar_zenith, 'solar_azimuth': solar_azimuth,
         'dni': dni, 'dhi': dhi},
        index=pd.DatetimeIndex(timestamps))

    dni_et = irradiance.extraradiation(df_inputs.index.dayofyear)
    am = atmosphere.relativeairmass(df_inputs.solar_zenith)

    # Need to treat the case when the sun is hitting the back surface of pvrow
    aoi_proj = aoi_projection(
        df_inputs.surface_tilt, df_inputs.surface_azimuth,
        df_inputs.solar_zenith, df_inputs.solar_azimuth)
    sun_hitting_back_surface = ((aoi_proj < 0)
                                & (df_inputs.solar_zenith <= 90))
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
    a = aoi_projection(df_inputs.surface_tilt,
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


def calculate_custom_perez_transposition(timestamps, surface_tilt,
                                         surface_azimuth, solar_zenith,
                                         solar_azimuth, dni, dhi):
    """
    Calculate custom perez transposition: some customization is necessary in
    order to get the circumsolar component when the sun is hitting the back
    surface as well.

    :param array-like timestamps: simulation timestamps
    :param array-like surface_tilt: Surface tilt angles in decimal degrees.
        surface_tilt must be >=0 and <=180.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)
    :param array-like surface_azimuth: The azimuth of the rotated panel,
        determined by projecting the vector normal to the panel's surface to
        the earth's surface [degrees].
    :param array-like solar_zenith: solar zenith angles
    :param array-like solar_azimuth: solar azimuth angles
    :param array-like dni: values for direct normal irradiance
    :param array-like dhi: values for diffuse horizontal irradiance
    :return: ``df_custom_perez``, dataframe with the following columns:
        ['solar_zenith', 'solar_azimuth', 'surface_tilt', 'surface_azimuth',
        'dhi', 'dni', 'vf_horizon', 'vf_circumsolar', 'vf_isotropic',
        'luminance_horizon', 'luminance_circumsolar', 'luminance_isotropic',
        'poa_isotropic', 'poa_circumsolar', 'poa_horizon', 'poa_total_diffuse']
    :rtype: class:`pandas.DataFrame`
    """
    # Calculate the perez inputs
    df_custom_perez = perez_diffuse_luminance(
        timestamps, surface_tilt, surface_azimuth,
        solar_zenith, solar_azimuth, dni, dhi)

    return df_custom_perez


def calculate_radiosities_serially_perez(args):
    """ Calculate timeseries results of simulation: run both custom Perez
    diffuse light transposition calculations and ``pvarray.Array`` timeseries
    calculation.

    :param args: tuple of arguments used to run the timeseries calculation.
        List in order: ``pvarray_parameters``, ``timestamps``,
        ``solar_zenith``, ``solar_azimuth``, ``surface_tilt``,
        ``surface_azimuth``, ``dni``, ``dhi``.
        All 1-dimensional arrays.
    :return: ``df_registries``, ``df_custom_perez``; dataframes containing
        the concatenated and timestamped ``pvarray.Array.surface_registry``
        values, and the custom Perez inputs used for it.
    :rtype: both class:`pandas.DataFrame`

    """
    # Get arguments
    (pvarray_parameters, timestamps, solar_zenith, solar_azimuth,
     surface_tilt, surface_azimuth, dni, dhi) = args

    # Run custom perez transposition: in order to get circumsolar on back
    # surface too
    df_custom_perez = calculate_custom_perez_transposition(
        timestamps, surface_tilt, surface_azimuth, solar_zenith,
        solar_azimuth, dni, dhi)

    # Get the necessary inputs
    luminance_isotropic = df_custom_perez.luminance_isotropic.values
    luminance_circumsolar = df_custom_perez.luminance_circumsolar.values
    poa_horizon = df_custom_perez.poa_horizon.values
    poa_circumsolar = df_custom_perez.poa_circumsolar.values

    # Run timeseries calculation
    df_registries = array_timeseries_calculate(
        pvarray_parameters, timestamps, solar_zenith, solar_azimuth,
        surface_tilt, surface_azimuth, dni, luminance_isotropic,
        luminance_circumsolar, poa_horizon, poa_circumsolar)

    return df_registries, df_custom_perez


def calculate_radiosities_parallel_perez(
        pvarray_parameters, timestamps, solar_zenith, solar_azimuth,
        surface_tilt, surface_azimuth, dni, dhi, n_processes=None):
    """ Calculate timeseries results of simulation in parallel:
    run both custom Perez diffuse light transposition calculations and
    ``pvarray.Array`` timeseries calculation.

    :param dict pvarray_parameters: parameters used to instantiate
        ``pvarray.Array`` class
    :param array-like timestamps: simulation timestamps
    :param array-like solar_zenith: solar zenith angles
    :param array-like solar_azimuth: solar azimuth angles
    :param array-like surface_tilt: Surface tilt angles in decimal degrees.
        surface_tilt must be >=0 and <=180.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)
    :param array-like surface_azimuth: The azimuth of the rotated panel,
        determined by projecting the vector normal to the panel's surface to
        the earth's surface [degrees].
    :param array-like dni: values for direct normal irradiance
    :param array-like dhi: values for diffuse horizontal irradiance
    :param int n_processes: (optional, default ``None`` = use all) number of
        processes to use. Default value will be the total number of processors
        on the machine.
    :return: ``df_registries``, ``df_custom_perez``; dataframes containing
        the concatenated and timestamped ``pvarray.Array.surface_registry``
        values, and the custom Perez inputs used for it.
    :rtype: both class:`pandas.DataFrame`
    """

    # Choose number of workers
    if n_processes is None:
        n_processes = cpu_count()

    # Split all arguments according to number of processes
    (list_timestamps, list_surface_azimuth, list_surface_tilt,
     list_solar_zenith, list_solar_azimuth, list_dni, list_dhi) = map(
         np.array_split,
         [timestamps, surface_azimuth, surface_tilt,
          solar_zenith, solar_azimuth, dni, dhi],
        [n_processes] * 7)
    list_parameters = [pvarray_parameters] * n_processes
    # Zip all the arguments together
    list_args = zip(*(list_parameters, list_timestamps, list_solar_zenith,
                      list_solar_azimuth, list_surface_tilt,
                      list_surface_azimuth, list_dni, list_dhi))

    # Start multiprocessing
    pool = Pool(n_processes)
    start = time.time()
    results = pool.map(calculate_radiosities_serially_perez, list_args)
    end = time.time()
    pool.close()
    pool.join()

    LOGGER.info("Parallel calculation elapsed time: %s sec" % str(end - start))

    results_grouped = zip(*results)
    results_concat = map(lambda x: pd.concat(x, axis=0, join='outer',
                                             sort=True)
                         if x[0] is not None else None,
                         results_grouped)

    return results_concat


def get_average_pvrow_outputs(df_registries, values=COLS_TO_SAVE,
                              include_shading=True):
    """ For each pvrow surface (front and back), calculate the weighted
    average irradiance and shading values (weighted by length of surface).

    :param df_registries: timestamped and concatenated :class:`pvarray.Array`
        surface registries
    :type df_registries: ``pandas.DataFrame``
    :param list values: list of column names from the surface registries to
        keep (optional, default=``COLS_TO_SAVE``)
    :param bool include_shading: flag to decide whether to keep column
        indicating if surface has direct shading or not
    :return: ``df_outputs``, dataframe with averaged values and flags
    :rtype: ``pandas.DataFrame``
    """

    weight_col = ['area']
    shade_col = ['shaded']
    indexes = ['timestamps', 'pvrow_index', 'surface_side']
    if include_shading:
        final_cols = values + shade_col
    else:
        final_cols = values

    # Format registry to get averaged outputs for each surface type
    df_outputs = (
        deepcopy(df_registries)
        .query('line_type == "pvrow"')
        .assign(pvrow_index=lambda x: x['pvrow_index'].astype(int))
        .loc[:, values + indexes + weight_col + shade_col]
        # Weighted averages: make sure to close the col value in lambdas
        .assign(**{col: lambda x, y=col: x[y] * x[weight_col[0]]
                   for col in values})
        .assign(shaded=lambda x: pd.to_numeric(x.shaded))
        .groupby(indexes)
        .sum()
        .assign(**{col: lambda x, y=col: x[y] / x[weight_col[0]]
                   for col in values})
        # summed up bool values, so there is shading if the shaded col > 0
        .assign(**{shade_col[0]: lambda x: x[shade_col[0]] > 0})
        # Now pivot data to the right format
        .reset_index()
        .melt(id_vars=indexes, value_vars=final_cols, var_name='term')
        .pivot_table(index=['timestamps'],
                     columns=['pvrow_index', 'surface_side', 'term'],
                     values='value',
                     # The following works because there is no actual
                     # aggregation happening
                     aggfunc='first'  # necessary to keep 'shaded' bool values
                     )
        .pipe(add_df_registries_nans, df_registries)
    )

    # Make sure numerical types are as expected
    df_outputs.loc[:, idx_slice[:, :, values]] = df_outputs.loc[
        :, idx_slice[:, :, values]].astype(float)

    return df_outputs.sort_index()


def get_pvrow_segment_outputs(df_registries, values=COLS_TO_SAVE,
                              include_shading=True):
    """For each discretized segment of a pvrow surface (front and back),
    calculate the weighted average irradiance and shading values
    (weighted by length of surface).

    :param df_registries: timestamped and concatenated :class:`pvarray.Array`
        surface registries
    :type df_registries: ``pandas.DataFrame``
    :param list values: list of column names from the surface registries to
        keep (optional, default=``COLS_TO_SAVE``)
    :param bool include_shading: flag to decide whether to keep column
        indicating if surface has direct shading or not
    :return: ``df_segments``, dataframe with averaged values and flags
    :rtype: ``pandas.DataFrame``
    """

    weight_col = ['area']
    shade_col = ['shaded']
    indexes = ['timestamps', 'pvrow_index', 'surface_side',
               'pvrow_segment_index']
    if include_shading:
        final_cols = values + shade_col
    else:
        final_cols = values

    # Format registry to get averaged outputs for each surface type
    df_segments = (
        deepcopy(df_registries)
        .loc[~ np.isnan(df_registries.pvrow_segment_index).values,
             values + indexes + weight_col + shade_col]
        .assign(
            pvrow_segment_index=lambda x: x['pvrow_segment_index'].astype(int),
            pvrow_index=lambda x: x['pvrow_index'].astype(int),
            shaded=lambda x: pd.to_numeric(x.shaded))
        # Weighted averages: make sure to close the col value in lambdas
        # Include shaded and non shaded segments
        .assign(**{col: lambda x, y=col: x[y] * x[weight_col[0]]
                   for col in values})
        .groupby(indexes)
        .sum()
        .assign(**{col: lambda x, y=col: x[y] / x[weight_col[0]]
                   for col in values})
        # summed up bool values, so there is shading if the shaded col > 0
        .assign(**{shade_col[0]: lambda x: x[shade_col[0]] > 0})
        # Now pivot data to the right format
        .reset_index()
        .melt(id_vars=indexes, value_vars=final_cols, var_name='term')
        .pivot_table(index=['timestamps'],
                     columns=['pvrow_index', 'surface_side',
                              'pvrow_segment_index', 'term'],
                     values='value',
                     # The following works because there is no actual
                     # aggregation happening
                     aggfunc='first'  # necessary to keep 'shaded' bool values
                     )
        .pipe(add_df_registries_nans, df_registries)
    )
    # Make sure numerical types are as expected
    df_segments.loc[:, idx_slice[:, :, :, values]] = df_segments.loc[
        :, idx_slice[:, :, :, values]].astype(float)

    return df_segments.sort_index()


def breakup_df_inputs(df_inputs):
    """
    Helper function: It is sometimes easier to provide a dataframe than a list
    of arrays: this function does the job of breaking up the dataframe into a
    list of expected 1-dim arrays

    :param df_inputs: timestamp-indexed dataframe with following columns:
        'surface_azimuth', 'surface_tilt', 'solar_zenith', 'solar_azimuth',
        'dni', 'dhi'
    :type df_inputs: ``pandas.DataFrame``
    :return: ``timestamps``, ``tilt_angles``, ``surface_azimuth``,
        ``solar_zenith``, ``solar_azimuth``, ``dni``, ``dhi``
    :rtype: all 1-dim arrays
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


def add_df_registries_nans(df, df_registries):
    """ df_registries as nan entries for timestamps that were skipped,
    make sure to add them back in

    :param pd.DataFrame df: dataframe to which the nan values will be added
    :param pd.DataFrame df_registries: dataframe of concatenated registries
    :return: ``pandas.DataFrame`` with nan values added
    """
    df_registries = df_registries.copy().set_index('timestamps')
    list_idx_nans = df_registries.index[
        df_registries.count(axis=1) == 0].tolist()
    if list_idx_nans:
        df_nan = pd.DataFrame(np.nan, columns=df.columns,
                              index=list_idx_nans)
        df = pd.concat([df, df_nan], axis=0, sort=True)

    return df
