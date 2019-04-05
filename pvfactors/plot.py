# -*- coding: utf-8 -*-

import numpy as np
from pvfactors import logging
from pvfactors.config import COLOR_DIC

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def plot_array_from_registry(ax, registry, line_types_selected=None,
                             fontsize=20):
    """Plot a 2D PV array using the ``shapely`` geometry objects located in
    an :py:class:`~pvfactors.pvarray.Array` surface registry.

    Parameters
    ----------
    ax : ``matplotlib.axes.Axes``
        axes to use for the plot
    registry : ``pandas.DataFrame``
        registry containing geometries  to plot
    line_types_selected : list
        parameter used to select a subset of
        'line_type' to plot; e.g. 'pvrow' or 'ground' (Default value = None)
    fontsize : int, optional
         Font size to use in plot (Default value = 20)

    Returns
    -------
    None
       Plot is shown in ``ax`` 's figure

    """

    registry = registry.copy()
    registry.loc[:, 'color'] = (
        registry.line_type.values + '_' +
        np.where(registry.shaded.values, 'shaded', 'illum'))
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
    """Plot a 2D PV array from a :py:class:`~pvfactors.pvarray.Array` using its
    ``surface_registry``

    Parameters
    ----------
    ax : ``matplotlib.axes.Axes``
        axes to use for the plot
    pvarray : :py:class:`~pvfactors.pvarray.Array`
        Array object with surface registry attribute
    line_types_selected : list
        parameter used to select a subset of
        'line_type' to plot; e.g. 'pvrow' or 'ground' (Default value = None)
    fontsize : int, optional
         Font size to use in plot (Default value = 20)

    Returns
    -------
    None
       Plot is shown in ``ax`` 's figure

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
    """Plot coordinates of shapely objects

    Parameters
    ----------
    ax : ``matplotlib.pyplot.Axes`` object
        Axes for plotting
    ob : ``Shapely`` object
        Geometry object whose x,y coordinates should be plotted

    """
    try:
        x, y = ob.xy
        ax.plot(x, y, 'o', color='#999999', zorder=1)
    except NotImplementedError:
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, 'o', color='#999999', zorder=1)


def plot_bounds(ax, ob):
    """Plot boundaries of shapely object

    Parameters
    ----------
    ax : ``matplotlib.pyplot.Axes`` object
        Axes for plotting
    ob : ``Shapely`` object
        Geometry object whose boundaries should be plotted

    """
    # Check if shadow reduces to one point (for very specific sun alignment)
    if len(ob.boundary) == 0:
        x, y = ob.coords[0]
    else:
        x, y = zip(*list((p.x, p.y) for p in ob.boundary))
    ax.plot(x, y, 'o', color='#000000', zorder=1)


def plot_line(ax, ob, line_style, line_color):
    """Plot boundaries of shapely line

    Parameters
    ----------
    ax : ``matplotlib.pyplot.Axes`` object
        Axes for plotting
    ob : ``Shapely`` object
        Geometry object whose boundaries should be plotted
    line_style : str
        matplotlib style to use for plotting the line
    line_color :
        matplotlib color to use for plotting the line

    """
    try:
        x, y = ob.xy
        ax.plot(x, y, color=COLOR_DIC[line_color], ls=line_style, alpha=0.7,
                linewidth=3, solid_capstyle='round', zorder=2)
    except NotImplementedError:
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, color=COLOR_DIC[line_color], ls=line_style,
                    alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
