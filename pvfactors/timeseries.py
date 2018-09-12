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
    arguments, timestamps, solar_zenith, solar_azimuth, array_tilt,
        array_azimuth, dni, luminance_isotropic, luminance_circumsolar,
        poa_horizon, poa_circumsolar):
    """
    Calculate the view factor radiosity and irradiance terms for multiple times.
    The calculations will be sequential, and they will assume a diffuse sky
    dome as calculated in the Perez diffuse sky transposition model (from
    ``pvlib-python`` implementation).

    :param args: tuple of at least two arguments: ``(arguments, df_inputs)``,
        where ``arguments`` is a ``dict`` that contains the array parameters
        used to instantiate a :class:`pvarray.Array` object, and ``df_inputs``
        is a :class:`pandas.DataFrame` with the following columns:
        ['solar_zenith', 'solar_azimuth', 'array_tilt', 'array_azimuth', 'dhi',
        'dni'], and with the following units: ['deg', 'deg', 'deg', 'deg',
        'W/m2', 'W/m2']. A possible 3rd argument for the tuple is
        ``save_segments``, which is a ``tuple`` of two elements used to save
        all the irradiance terms calculated for one side of a PV row; e.g.
        ``(1, 'front')`` the first element is an ``int`` for the PV row index,
        and the second element a ``str`` to specify the side of the PV row,
        'front' or 'back'
    :return: ``df_outputs, df_bifacial_gain, df_inputs_perez, ipoa_dict``;
        :class:`pandas.DataFrame` objects and ``dict`` where ``df_outputs``
        contains *averaged* irradiance terms for all PV row sides and at each
        time stamp; ``df_bifacial_gain`` contains the calculation of
        back-surface over front-surface irradiance for all PV rows and at each
        time stamp; ``df_inputs_perez`` contains the intermediate input and
        output values from the Perez model calculation in ``pvlib-python``;
        ``ipoa_dict`` is not ``None`` only when the ``save_segments`` input is
        specified, and it is otherwise a ``dict`` where the keys are all the
        calculated irradiance terms' names, and the values are
        :class:`pandas.DataFrame` objects containing the calculated values for
        all the segments of the PV string side (it is a way of getting detailed
        values instead of averages)
    """
    # Instantiate array
    array = Array(**arguments)
    # We want to save the whole registry for each timestamp
    list_registries = []
    # Use for printing progress
    n = len(timestamps)
    i = 1
    for idx, time in enumerate(timestamps):
        try:
            if ((isinstance(solar_zenith[idx], float))
                    & (solar_zenith[idx] <= 90.)):
                # Run calculation only if daytime
                array.calculate_radiosities_perez(
                    solar_zenith[idx], solar_azimuth[idx], array_tilt[idx],
                    array_azimuth[idx], dni[idx], luminance_isotropic[idx],
                    luminance_circumsolar[idx], poa_horizon[idx], poa_circumsolar[idx])

                # Save the whole registry
                registry = deepcopy(array.surface_registry)
                registry['timestamps'] = time
                list_registries.append(registry)

        except Exception as err:
            LOGGER.debug("Unexpected error: {0}".format(err))

        # Printing progress
        print_progress(i, n, prefix='Progress:', suffix='Complete',
                       bar_length=50)
        i += 1

    # Concatenate all surface registries into one dataframe
    if list_registries:
        df_registries = pd.concat(list_registries, axis=0, join='outer')
    else:
        df_registries = None

    return df_registries


def perez_diffuse_luminance(timestamps, array_tilt, array_azimuth,
                            solar_zenith, solar_azimuth, dni, dhi):
    """
    Function used to calculate the luminance and the view factor terms from the
    Perez diffuse light transposition model, as implemented in the
    ``pvlib-python`` library.

    :param df_inputs: class:`pandas.DataFrame` with following columns:
        ['solar_zenith', 'solar_azimuth', 'array_tilt', 'array_azimuth', 'dhi',
        'dni']. Units are: ['deg', 'deg', 'deg', 'deg', 'W/m2', 'W/m2']
    :return: class:`pandas.DataFrame` with the following columns:
        ['solar_zenith', 'solar_azimuth', 'array_tilt', 'array_azimuth', 'dhi',
        'dni', 'vf_horizon', 'vf_circumsolar', 'vf_isotropic',
        'luminance_horizon', 'luminance_circumsolar', 'luminance_isotropic',
        'poa_isotropic', 'poa_circumsolar', 'poa_horizon', 'poa_total_diffuse']
    """

    # Create a dataframe to help filtering on all arrays
    df_inputs = pd.DataFrame(
        {'array_tilt': array_tilt, 'array_azimuth': array_azimuth,
         'solar_zenith': solar_zenith, 'solar_azimuth': solar_azimuth,
         'dni': dni, 'dhi': dhi},
        index=pd.DatetimeIndex(timestamps))

    dni_et = irradiance.extraradiation(df_inputs.index.dayofyear)
    am = atmosphere.relativeairmass(df_inputs.solar_zenith)

    # Need to treat the case when the sun is hitting the back surface of pvrow
    aoi_proj = aoi_projection(df_inputs.array_tilt, df_inputs.array_azimuth,
                              df_inputs.solar_zenith, df_inputs.solar_azimuth)
    sun_hitting_back_surface = ((aoi_proj < 0) &
                                (df_inputs.solar_zenith <= 90))
    df_inputs_back_surface = df_inputs.loc[sun_hitting_back_surface]
    # Reverse the surface normal to switch to back-surface circumsolar calc
    df_inputs_back_surface.loc[:, 'array_azimuth'] -= 180.
    df_inputs_back_surface.loc[:, 'array_azimuth'] = np.mod(
        df_inputs_back_surface.loc[:, 'array_azimuth'], 360.
    )
    df_inputs_back_surface.loc[:, 'array_tilt'] = (
        180. - df_inputs_back_surface.array_tilt)

    if df_inputs_back_surface.shape[0] > 0:
        # Use recursion to calculate circumsolar luminance for back surface
        df_inputs_back_surface = perez_diffuse_luminance(
            *breakup_df_inputs(df_inputs_back_surface))

    # Calculate Perez diffuse components
    diffuse_poa, components = irradiance.perez(df_inputs.array_tilt,
                                               df_inputs.array_azimuth,
                                               df_inputs.dhi, df_inputs.dni,
                                               dni_et,
                                               df_inputs.solar_zenith,
                                               df_inputs.solar_azimuth,
                                               am,
                                               return_components=True)

    # Calculate Perez view factors:
    a = aoi_projection(df_inputs.array_tilt, df_inputs.array_azimuth,
                       df_inputs.solar_zenith, df_inputs.solar_azimuth)
    a = np.maximum(a, 0)
    b = cosd(df_inputs.solar_zenith)
    b = np.maximum(b, cosd(85))

    vf_perez = pd.DataFrame(
        np.array([
            sind(df_inputs.array_tilt),
            a / b,
            (1. + cosd(df_inputs.array_tilt)) / 2.
        ]).T,
        index=df_inputs.index,
        columns=['vf_horizon', 'vf_circumsolar', 'vf_isotropic']
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
    luminance.loc[diffuse_poa == 0, :] = 0.

    # Format components column names
    components = components.rename(columns={'isotropic': 'poa_isotropic',
                                            'circumsolar': 'poa_circumsolar',
                                            'horizon': 'poa_horizon'})

    df_inputs = pd.concat([df_inputs, components, vf_perez, luminance,
                           diffuse_poa],
                          axis=1, join='outer')
    df_inputs = df_inputs.rename(columns={0: 'poa_total_diffuse'})

    # Adjust the circumsolar luminance when it hits the back surface
    if df_inputs_back_surface.shape[0] > 0:
        df_inputs.loc[sun_hitting_back_surface, 'luminance_circumsolar'] = (
            df_inputs_back_surface.loc[:, 'luminance_circumsolar']
        )
    return df_inputs


def calculate_custom_perez_transposition(timestamps, array_tilt, array_azimuth,
                                         solar_zenith, solar_azimuth, dni, dhi):
    """
    Calculate custom perez transposition: some pre-processing is necessary in order
    to get all the components even when the sun is hitting the pv row back surface

    :param pd.DatetimeIndex timestamps:
    :param np.array array_tilt:
    :param np.array array_azimuth:
    :param np.array solar_zenith:
    :param np.array solar_azimuth:
    :param np.array dni:
    :param np.array dhi:
    :return: ``df_custom_perez``, pandas Dataframe
    """
    # Pre-process df_inputs to use the expected format of pvlib's perez model:
    # only positive tilt angles, and switching azimuth angles
    array_azimuth_processed = deepcopy(array_azimuth)
    array_azimuth_processed[array_tilt < 0] = np.remainder(
        array_azimuth[array_tilt < 0] + 180.,
        360.)
    array_tilt_processed = np.abs(array_tilt)

    # Calculate the perez inputs
    df_custom_perez = perez_diffuse_luminance(timestamps, array_tilt_processed,
                                              array_azimuth_processed,
                                              solar_zenith, solar_azimuth, dni, dhi)

    return df_custom_perez


def calculate_radiosities_serially_perez(args):
    """ Timeseries simulation with no parallellization, using Perez model """

    # Get arguments
    (arguments, timestamps, solar_zenith, solar_azimuth, array_tilt, array_azimuth,
     dni, dhi) = args

    # Run custom perez transposition: in order to get circumsolar on back surface too
    df_custom_perez = calculate_custom_perez_transposition(
        timestamps, array_tilt, array_azimuth, solar_zenith, solar_azimuth,
        dni, dhi)

    # Get the necessary inputs
    luminance_isotropic = df_custom_perez.luminance_isotropic.values
    luminance_circumsolar = df_custom_perez.luminance_circumsolar.values
    poa_horizon = df_custom_perez.poa_horizon.values
    poa_circumsolar = df_custom_perez.poa_circumsolar.values

    # Run timeseries calculation
    df_registries = array_timeseries_calculate(
        arguments, timestamps, solar_zenith, solar_azimuth, array_tilt, array_azimuth,
        dni, luminance_isotropic, luminance_circumsolar, poa_horizon, poa_circumsolar)

    return df_registries, df_custom_perez


def calculate_radiosities_parallel_perez(
        array_parameters, timestamps, solar_zenith, solar_azimuth,
        array_tilt, array_azimuth, dni, dhi, n_processes=None):
    """
    Calculate the view factor radiosity and irradiance terms for multiple times.
    The calculations will be run in parallel on the different processors, and
    they will assume a diffuse sky dome as calculated in the Perez diffuse
    sky transposition model (from ``pvlib-python`` implementation).
    This function uses :func:`tools.calculate_radiosities_serially_perez`

    :param dict array_parameters: contains the array parameters used to
        instantiate a :class:`pvarray.Array` object
    :param df_inputs: :class:`pandas.DataFrame` with the following columns:
        ['solar_zenith', 'solar_azimuth', 'array_tilt', 'array_azimuth', 'dhi',
        'dni'], and with the following units: ['deg', 'deg', 'deg', 'deg',
        'W/m2', 'W/m2']
    :param int n_processes: number of processes to use. Default value will be
        the total number of processors on the machine
    :return: concatenated outputs of
        :func:`tools.calculate_radiosities_serially_perez` run and outputted in
        the parallel processes
    """

    # Choose number of workers
    if n_processes is None:
        n_processes = cpu_count()

    # Split all arguments according to number of processes
    (list_timestamps, list_array_azimuth, list_array_tilt,
     list_solar_zenith, list_solar_azimuth, list_dni, list_dhi) = map(
         np.array_split,
         [timestamps, array_azimuth, array_tilt,
          solar_zenith, solar_azimuth, dni, dhi],
        [n_processes] * 7)
    list_parameters = [array_parameters] * n_processes
    # Zip all the arguments together
    list_args = zip(*(list_parameters, list_timestamps, list_solar_zenith,
                      list_solar_azimuth, list_array_tilt, list_array_azimuth,
                      list_dni, list_dhi))

    # import pdb
    # pdb.set_trace()
    # Start multiprocessing
    pool = Pool(n_processes)
    start = time.time()
    results = pool.map(calculate_radiosities_serially_perez, list_args)
    end = time.time()
    pool.close()
    pool.join()

    LOGGER.info("Parallel calculation elapsed time: %s sec" % str(end - start))

    results_grouped = zip(*results)
    results_concat = map(lambda x: pd.concat(x, axis=0, join='outer')
                         if x[0] is not None else None,
                         results_grouped)

    return results_concat


def get_average_pvrow_outputs(df_registries, values=COLS_TO_SAVE,
                              include_shading=True):
    """ Calculate surface side irradiance averages for the pvrows """

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
        # Calculate weighted averages: make sure to close the col value in lambdas
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
                     # The following works because there is no actual aggregation happening
                     aggfunc='first'  # necessary to keep 'shaded' bool values
                     )
    )
    # Make sure numerical types are as expected
    df_outputs.loc[:, idx_slice[:, :, values]] = df_outputs.loc[:, idx_slice[:, :, values]
                                                                ].astype(float)

    return df_outputs


def get_bifacial_gain_outputs(df_outputs):
    """ Calculate irradiance bifacial gain for all pvrows """
    pass


def get_pvrow_segment_outputs(df_registries, values=COLS_TO_SAVE,
                              include_shading=True):
    """ Get only pvrow segment outputs """

    weight_col = ['area']
    shade_col = ['shaded']
    indexes = ['timestamps', 'pvrow_index', 'surface_side', 'pvrow_segment_index']
    if include_shading:
        final_cols = values + shade_col
    else:
        final_cols = values

    # Format registry to get averaged outputs for each surface type
    df_segments = (
        deepcopy(df_registries)
        .loc[~ np.isnan(df_registries.pvrow_segment_index).values,
             values + indexes + weight_col + shade_col]
        .assign(pvrow_segment_index=lambda x: x['pvrow_segment_index'].astype(int),
                pvrow_index=lambda x: x['pvrow_index'].astype(int),
                shaded=lambda x: pd.to_numeric(x.shaded))
        # Calculate weighted averages: make sure to close the col value in lambdas
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
                     columns=['pvrow_index', 'surface_side', 'pvrow_segment_index',
                              'term'],
                     values='value',
                     # The following works because there is no actual aggregation happening
                     aggfunc='first'  # necessary to keep 'shaded' bool values
                     )
    )
    # Make sure numerical types are as expected
    df_segments.loc[:, idx_slice[:, :, :, values]] = df_segments.loc[:, idx_slice[:, :, :, values]
                                                                     ].astype(float)

    return df_segments


def breakup_df_inputs(df_inputs):
    """
    It is sometimes easier to provide a dataframe than a list of arrays:
    this function does the job of breaking up the dataframe into a list of
    arrays
    """
    timestamps = pd.to_datetime(df_inputs.index)
    array_azimuth = df_inputs.array_azimuth.values
    array_tilt = df_inputs.array_tilt.values
    solar_zenith = df_inputs.solar_zenith.values
    solar_azimuth = df_inputs.solar_azimuth.values
    dni = df_inputs.dni.values
    dhi = df_inputs.dhi.values

    return (timestamps, array_tilt, array_azimuth,
            solar_zenith, solar_azimuth, dni, dhi)
