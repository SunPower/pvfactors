# -*- coding: utf-8 -*-

import sys
from pvlib import atmosphere, irradiance
from pvlib.tools import cosd, sind
from pvlib.irradiance import aoi_projection
import numpy as np
import pandas as pd
from pvfactors import logging
from pvfactors.pvarray import Array
from multiprocessing import Pool, cpu_count
import time

DISTANCE_TOLERANCE = 1e-8

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# Define colors used for plotting the 2D arrays
COLOR_dic = {
    'i': '#FFBB33',
    's': '#A7A49D',
    't': '#6699cc',
    'pvrow_illum': '#6699cc',
    'pvrow_shaded': '#ff0000',
    'ground_shaded': '#A7A49D',
    'ground_illum': '#FFBB33'
}


# Plot the array geometries
def plot_line_registry(ax, array, line_types_selected=None):
    """
    Plot a :class:`pvarray.Array` object's shapely geometries based on its
    :attr:`pvarray.Array.line_registry`.

    :param ax: :class:`matplotlib.axes.Axes` object to use for the plot
    :param array: :class:`pvarray.Array` object to plot
    :param list line_types_selected: parameter used to select a subset of
        'line_type' to plot; e.g. 'pvrow' or 'ground'
    :return: None; ``ax`` is updated
    """
    line_registry = array.line_registry.copy()
    line_registry['color'] = (line_registry.line_type.values + '_'
                              + np.where(line_registry.shaded.values,
                                         'shaded', 'illum'))
    # TODO: distance may not exist
    distance = array.pvrow_distance
    height = array.pvrow_height
    n_pvrows = array.n_pvrows
    if line_types_selected:
        for line_type in line_types_selected:
            line_reg_selected = line_registry.loc[line_registry.line_type
                                                  == line_type, :]
            for index, row in line_reg_selected.iterrows():
                LOGGER.debug("Plotting %s", row['line_type'])
                plot_coords(ax, row['geometry'])
                plot_bounds(ax, row['geometry'])
                plot_line(ax, row['geometry'], row['style'],
                          row['shading_type'])
    else:
        for index, row in line_registry.iterrows():
            LOGGER.debug("Plotting %s", row['line_type'])
            plot_coords(ax, row['geometry'])
            plot_bounds(ax, row['geometry'])
            plot_line(ax, row['geometry'], row['style'], row['color'])

    ax.axis('equal')
    ax.set_xlim(- 0.5 * distance, (n_pvrows - 0.5) * distance)
    ax.set_ylim(-height, 2 * height)
    ax.set_title("PV Array", fontsize=20)

    ax.set_xlabel("x [m]", fontsize=20)
    ax.set_ylabel("y [m]", fontsize=20)


def plot_surface_registry(ax, array, line_types_selected=None):
    """
    Plot a :class:`pvarray.Array` object's shapely geometries based on its
    :attr:`pvarray.Array.surface_registry`. The difference with
    :func:`tools.plot_line_registry` is that here the user will see the
    differences between PV row sides, the discretized elements, etc.

    :param ax: :class:`matplotlib.axes.Axes` object to use for the plot
    :param array: :class:`pvarray.Array` object to plot
    :param list line_types_selected: parameter used to select a subset of
        'line_type' to plot; e.g. 'pvrow' or 'ground'
    :return: None (``ax`` is updated)
    """
    # FIXME: repeating code from plot_line_registry
    surface_registry = array.surface_registry.copy()
    surface_registry['color'] = (surface_registry.line_type.values + '_'
                                 + np.where(surface_registry.shaded.values,
                                            'shaded', 'illum'))
    # TODO: distance may not exist
    distance = array.pvrow_distance
    height = array.pvrow_height
    n_pvrows = array.n_pvrows
    if line_types_selected:
        for line_type in line_types_selected:
            surface_reg_selected = surface_registry.loc[
                surface_registry.line_type == line_type, :]
            for index, row in surface_reg_selected.iterrows():
                LOGGER.debug("Plotting %s", row['line_type'])
                plot_coords(ax, row['geometry'])
                plot_bounds(ax, row['geometry'])
                plot_line(ax, row['geometry'], row['style'],
                          row['shading_type'])
    else:
        for index, row in surface_registry.iterrows():
            LOGGER.debug("Plotting %s", row['line_type'])
            plot_coords(ax, row['geometry'])
            plot_bounds(ax, row['geometry'])
            plot_line(ax, row['geometry'], row['style'], row['color'])

    ax.axis('equal')
    ax.set_xlim(- 0.5 * distance, (n_pvrows - 0.5) * distance)
    ax.set_ylim(-height, 2 * height)
    ax.set_title("PV Array", fontsize=20)

    ax.set_xlabel("x [m]", fontsize=20)
    ax.set_ylabel("y [m]", fontsize=20)


# Calculate luminance using pvlib functions and classes
def perez_diffuse_luminance(df_inputs):
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

    dni_et = irradiance.extraradiation(df_inputs.index.dayofyear)
    am = atmosphere.relativeairmass(df_inputs.solar_zenith)

    # Need to treat the case when the sun is hitting the back surface of pvrow
    aoi_proj = aoi_projection(df_inputs.array_tilt, df_inputs.array_azimuth,
                              df_inputs.solar_zenith, df_inputs.solar_azimuth)
    sun_hitting_back_surface = ((aoi_proj < 0) &
                                (df_inputs.solar_zenith <= 90))
    df_inputs_back_surface = df_inputs.loc[sun_hitting_back_surface]
    # Reverse the surface normal to switch to back-surface circumsolar calc
    df_inputs_back_surface['array_azimuth'] -= 180.
    df_inputs_back_surface['array_azimuth'] = np.mod(
        df_inputs_back_surface['array_azimuth'], 360.
    )
    df_inputs_back_surface['array_tilt'] = (
        180. - df_inputs_back_surface.array_tilt)

    if df_inputs_back_surface.shape[0] > 0:
        # Use recursion to calculate circumsolar luminance for back surface
        df_inputs_back_surface = perez_diffuse_luminance(df_inputs_back_surface)

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


def calculate_radiosities_serially_simple(array, df_inputs):
    """
    Calculate the view factor radiosity and irradiance terms for multiple times.
    The calculations will be sequential, and they will assume a completely
    isotropic sky dome.

    :param array: :class:`pvarray.Array` object already configured and
        instantiated
    :param df_inputs: :class:`pandas.DataFrame` with following columns:
        ['solar_zenith', 'solar_azimuth', 'array_tilt', 'array_azimuth', 'dhi',
        'dni']. Units are: ['deg', 'deg', 'deg', 'deg', 'W/m2', 'W/m2']
    :return: ``df_outputs, df_bifacial_gain``; :class:`pandas.DataFrame` objects
        where ``df_outputs`` contains *averaged* irradiance terms for all PV row
        sides and at each time stamp; ``df_bifacial_gain`` contains the
        calculation of back-surface over front-surface irradiance for all PV
        rows and at each time stamp.
    """
    # Create index df_outputs
    iterables = [
        range(array.n_pvrows),
        ['front', 'back'],
        ['q0', 'qinc']
    ]
    multiindex = pd.MultiIndex.from_product(iterables,
                                            names=['pvrow', 'side', 'term'])

    df_outputs = pd.DataFrame(np.nan, columns=df_inputs.index,
                              index=multiindex)
    df_outputs.sort_index(inplace=True)
    df_outputs.loc['array_is_shaded', :] = np.nan
    idx_slice = pd.IndexSlice

    n = df_inputs.shape[0]
    i = 1

    for idx, row in df_inputs.iterrows():
        try:

            if ((isinstance(row['solar_zenith'], float))
                    & (row['solar_zenith'] <= 90.)):
                array.calculate_radiosities_simple(row['solar_zenith'],
                                                   row['solar_azimuth'],
                                                   row['array_tilt'],
                                                   row['array_azimuth'],
                                                   row['dni'], row['dhi'])

                array.surface_registry['q0'] = (
                    array.surface_registry['area'] * array.surface_registry.q0)
                array.surface_registry['qinc'] = (
                    array.surface_registry['area']
                    * array.surface_registry.qinc
                )
                df_summed = array.surface_registry.groupby(
                    ['pvrow_index', 'surface_side']).sum()
                df_avg_irradiance = df_summed.div(df_summed['area'],
                                                  axis=0).loc[
                    idx_slice[:, :], ['q0', 'qinc']].sort_index()
                df_outputs.loc[idx_slice[:, :, 'q0'], idx] = (
                    df_avg_irradiance.loc[
                        idx_slice[:, :], 'q0'].values
                )
                df_outputs.loc[idx_slice[:, :, 'qinc'],
                               idx] = df_avg_irradiance.loc[
                    idx_slice[:, :], 'qinc'].values
                df_outputs.loc['array_is_shaded', idx] = (
                    array.has_direct_shading)
        except Exception as err:
            LOGGER.debug("Unexpected error: {0}".format(err))

        print_progress(i, n, prefix='Progress:', suffix='Complete',
                       bar_length=50)
        i += 1

    try:
        bifacial_gains = (df_outputs.loc[
                          idx_slice[:, 'back', 'qinc'], :].values
                          / df_outputs.loc[
            idx_slice[:, 'front', 'qinc'], :].values)
        df_bifacial_gain = pd.DataFrame(bifacial_gains.T,
                                        index=df_outputs.index,
                                        columns=range(array.n_pvrows))
    except Exception as err:
        LOGGER.warning("Error in calculation of bifacial gain %s" % err)
        df_bifacial_gain = pd.DataFrame(
            np.nan * np.ones((len(df_inputs.index), array.n_pvrows)),
            index=df_inputs.index,
            columns=range(array.n_pvrows))

    return df_outputs, df_bifacial_gain


def calculate_radiosities_serially_perez(args):
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

    if len(args) == 3:
        arguments, df_inputs, save_segments = args
    else:
        arguments, df_inputs = args
        save_segments = None

    array = Array(**arguments)
    # Pre-process df_inputs to use the expected format of pvlib's perez model:
    # only positive tilt angles, and switching azimuth angles
    df_inputs_before_perez = df_inputs.copy()
    df_inputs_before_perez.loc[df_inputs.array_tilt < 0, 'array_azimuth'] = (
        np.remainder(
            df_inputs_before_perez.loc[df_inputs.array_tilt < 0,
                                       'array_azimuth'] + 180.,
            360.)
    )
    df_inputs_before_perez.array_tilt = np.abs(df_inputs_before_perez
                                               .array_tilt)

    # Calculate the perez inputs
    df_inputs_perez = perez_diffuse_luminance(df_inputs_before_perez)

    # Post process: in vf model tilt angles can be negative and azimuth is
    # generally fixed
    df_inputs_perez[['array_azimuth', 'array_tilt']] = (
        df_inputs[['array_azimuth', 'array_tilt']]
    )

    # Create index df_outputs
    cols = ['q0', 'qinc', 'circumsolar_term', 'horizon_term',
            'direct_term', 'irradiance_term', 'isotropic_term',
            'reflection_term']
    iterables = [
        range(array.n_pvrows),
        ['front', 'back'],
        cols
    ]
    multiindex = pd.MultiIndex.from_product(iterables,
                                            names=['pvrow', 'side', 'term'])

    # Initialize df_outputs
    df_outputs = pd.DataFrame(np.nan, columns=df_inputs_perez.index,
                              index=multiindex)
    df_outputs.sort_index(inplace=True)
    df_outputs.loc['array_is_shaded', :] = np.nan

    # Initialize df_outputs_segments
    if save_segments is not None:
        n_cols = len(array.pvrows[save_segments[0]].cut_points)
        cols_segments = range(n_cols + 1)
        irradiance_terms_segments = [
            'qinc', 'direct_term', 'circumsolar_term', 'horizon_term',
            'isotropic_term', 'reflection_term', 'circumsolar_shading_pct',
            'horizon_band_shading_pct'
        ]
        iterables_segments = [
            irradiance_terms_segments,
            cols_segments
        ]
        multiindex_segments = pd.MultiIndex.from_product(
            iterables_segments, names=['irradiance_term', 'segment_index'])
        df_outputs_segments = pd.DataFrame(np.nan,
                                           columns=df_inputs_perez.index,
                                           index=multiindex_segments)
        df_outputs_segments.sort_index(inplace=True)
        df_outputs_segments = df_outputs_segments.transpose()
    else:
        df_outputs_segments = None

    idx_slice = pd.IndexSlice

    n = df_inputs_perez.shape[0]
    i = 1

    for idx, row in df_inputs_perez.iterrows():
        try:
            if ((isinstance(row['solar_zenith'], float))
                    & (row['solar_zenith'] <= 90.)):
                array.calculate_radiosities_perez(row['solar_zenith'],
                                                  row['solar_azimuth'],
                                                  row['array_tilt'],
                                                  row['array_azimuth'],
                                                  row['dni'],
                                                  row['luminance_isotropic'],
                                                  row['luminance_circumsolar'],
                                                  row['poa_horizon'],
                                                  row['poa_circumsolar'])

                # TODO: this will only work if there is no shading on the
                # surfaces
                # Format data to save all the surfaces for a pvrow
                if save_segments is not None:
                    # Select the surface of the pv row with the segments and the
                    # right columns
                    df_pvrow = array.surface_registry.loc[
                        (array.surface_registry.pvrow_index == save_segments[0])
                        & (array.surface_registry.surface_side
                           == save_segments[1]),
                        irradiance_terms_segments
                        + ['shaded']
                    ]
                    # Check that no segment has direct shading before saving
                    # results
                    if df_pvrow.shaded.sum() == 0:
                        # Add the data to the output variable by looping over
                        # irradiance terms
                        for irradiance_term in irradiance_terms_segments:
                            df_outputs_segments.loc[
                                idx, idx_slice[irradiance_term, :]] = (
                                df_pvrow[irradiance_term].values
                            )

                # Format data to save averages for all pvrows and sides
                array.surface_registry[cols] = (
                    array.surface_registry[cols].apply(
                        lambda x: x * array.surface_registry['area'], axis=0)
                )
                df_summed = array.surface_registry.groupby(
                    ['pvrow_index', 'surface_side']).sum()
                df_avg_irradiance = (
                    df_summed.div(df_summed['area'], axis=0).loc[
                        idx_slice[:, :], cols].sort_index().stack())

                # # Assign values to df_outputs
                df_outputs.loc[idx_slice[:, :, cols], idx] = (
                    df_avg_irradiance.loc[idx_slice[:, :, cols]]
                )

                df_outputs.loc['array_is_shaded', idx] = (
                    array.has_direct_shading
                )

        except Exception as err:
            LOGGER.debug("Unexpected error: {0}".format(err))

        print_progress(i, n, prefix='Progress:', suffix='Complete',
                       bar_length=50)
        i += 1

    try:
        bifacial_gains = (df_outputs.loc[
                          idx_slice[:, 'back', 'qinc'], :].values
                          / df_outputs.loc[
            idx_slice[:, 'front', 'qinc'], :].values)
        df_bifacial_gain = pd.DataFrame(bifacial_gains.T,
                                        index=df_inputs_perez.index,
                                        columns=range(array.n_pvrows))
    except Exception as err:
        LOGGER.warning("Error in calculation of bifacial gain %s" % err)
        df_bifacial_gain = pd.DataFrame(
            np.nan * np.ones((len(df_inputs.index), array.n_pvrows)),
            index=df_inputs.index,
            columns=range(array.n_pvrows))

    return (df_outputs.transpose(), df_bifacial_gain, df_inputs_perez,
            df_outputs_segments)


def calculate_radiosities_parallel_perez(array_parameters, df_inputs,
                                         n_processes=None, save_segments=None):
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
    :param tuple save_segments: ``tuple`` of two elements used to save all the
        irradiance terms calculated for one side of a PV row; e.g.
        ``(1, 'front')`` the first element is an ``int`` for the PV row index,
        and the second element a ``str`` to specify the side of the PV row,
        'front' or 'back'
    :return: concatenated outputs of
        :func:`tools.calculate_radiosities_serially_perez` run and outputted in
        the parallel processes
    """

    # Choose number of workers
    if n_processes is None:
        n_processes = cpu_count()

    # Define list of arguments for target function
    list_df_inputs = np.array_split(df_inputs, n_processes, axis=0)
    list_parameters = [array_parameters] * n_processes
    list_save_segments = [save_segments] * n_processes
    list_args = zip(*(list_parameters, list_df_inputs, list_save_segments))

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


# Base functions used to plot the 2D array
def plot_coords(ax, ob):
    try:
        x, y = ob.xy
        ax.plot(x, y, 'o', color='#999999', zorder=1)
    except NotImplementedError:
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, 'o', color='#999999', zorder=1)


def plot_bounds(ax, ob):
    # Check if shadow reduces to one point (for very specific sun alignment)
    if len(ob.boundary) == 0:
        x, y = ob.coords[0]
    else:
        x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, 'o', color='#000000', zorder=1)


def plot_line(ax, ob, line_style, line_color):
    try:
        x, y = ob.xy
        ax.plot(x, y, color=COLOR_dic[line_color], ls=line_style, alpha=0.7,
                linewidth=3, solid_capstyle='round', zorder=2)
    except NotImplementedError:
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, color=COLOR_dic[line_color], ls=line_style, alpha=0.7,
                    linewidth=3, solid_capstyle='round', zorder=2)


# Define function used for progress bar when running long simulations
# Borrowed from: https://gist.github.com/aubricus
def print_progress(iteration, total, prefix='', suffix='', decimals=1,
                   bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
        complete (Int)
        bar_length   - Optional  : character length of bar (Int)
    """
    format_str = "{0:." + str(decimals) + "f}"
    percents = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%',
                                            suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
