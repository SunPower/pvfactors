# -*- coding: utf-8 -*-

import numpy as np
from pvfactors import logging

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


def plot_array_from_registry(ax, registry, line_types_selected=None,
                             fontsize=20):
    """
    Plot a 2D PV array using the ``shapely`` geometry objects located in
    a :class:`pvarray.Array` surface registry.

    :param matplotlib.axes.Axes ax: axes to use for the plot
    :param pd.DataFrame registry: registry containing geometries  to plot
    :param list line_types_selected: parameter used to select a subset of
        'line_type' to plot; e.g. 'pvrow' or 'ground'
    :return: None (``ax`` is updated)
    """

    registry = registry.copy()
    registry.loc[:, 'color'] = (
        registry.line_type.values + '_'
        + np.where(registry.shaded.values, 'shaded', 'illum'))
    # TODO: distance may not exist
    if line_types_selected:
        for line_type in line_types_selected:
            surface_reg_selected = registry.loc[
                registry.line_type == line_type, :]
            for index, row in surface_reg_selected.iterrows():
                LOGGER.debug("Plotting %s", row['line_type'])
                plot_coords(ax, row['geometry'])
                plot_bounds(ax, row['geometry'])
                plot_line(ax, row['geometry'], row['style'],
                          row['shading_type'])
    else:
        for index, row in registry.iterrows():
            LOGGER.debug("Plotting %s", row['line_type'])
            plot_coords(ax, row['geometry'])
            plot_bounds(ax, row['geometry'])
            plot_line(ax, row['geometry'], row['style'], row['color'])

    ax.axis('equal')
    ax.set_xlabel("x [m]", fontsize=fontsize)
    ax.set_ylabel("y [m]", fontsize=fontsize)


def plot_pvarray(ax, pvarray, line_types_selected=None, fontsize=20):
    """
    Plot a 2D PV array from a :class:`pvarray.Array` using its
    :attr:`pvarray.Array.surface_registry`.

    :param ax: :class:`matplotlib.axes.Axes` object to use for the plot
    :param pvarray: object containing the surface registry as attribute
    :type pvarray: :class:`pvarray.Array`
    :param list line_types_selected: parameter used to select a subset of
        'line_type' to plot; e.g. 'pvrow' or 'ground'
    :return: None (``ax`` is updated)
    """

    # FIXME: repeating code from plot_line_registry
    surface_registry = pvarray.surface_registry.copy()
    plot_array_from_registry(ax, surface_registry,
                             line_types_selected=line_types_selected)

    # Plot details
    distance = pvarray.pvrow_distance
    height = pvarray.pvrow_height
    n_pvrows = pvarray.n_pvrows
    ax.set_xlim(- 0.5 * distance, (n_pvrows - 0.5) * distance)
    ax.set_ylim(-height, 2 * height)
    ax.set_title("PV Array", fontsize=fontsize)


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
            ax.plot(x, y, color=COLOR_dic[line_color], ls=line_style,
                    alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
