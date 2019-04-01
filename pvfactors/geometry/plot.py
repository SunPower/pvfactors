"""Base functions used to plot 2D PV geometries"""


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


def plot_line(ax, ob, line_color):
    """Plot boundaries of shapely line

    Parameters
    ----------
    ax : ``matplotlib.pyplot.Axes`` object
        Axes for plotting
    ob : ``Shapely`` object
        Geometry object whose boundaries should be plotted
    line_color : str
        matplotlib color to use for plotting the line

    """
    try:
        x, y = ob.xy
        ax.plot(x, y, color=line_color, alpha=0.7,
                linewidth=3, solid_capstyle='round', zorder=2)
    except NotImplementedError:
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, color=line_color,
                    alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
