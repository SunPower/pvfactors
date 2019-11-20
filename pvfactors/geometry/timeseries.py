"""Timeseries geometry tools. They allow the vectorization of geometry
calculations."""

import numpy as np
from pvfactors.config import DISTANCE_TOLERANCE
from pvfactors.geometry.base import PVSurface, ShadeCollection
from shapely.geometry import GeometryCollection


class TsShadeCollection(object):
    """Collection of timeseries surfaces that are all either shaded or
    illuminated. This will be used by both ground and PV row
    geometries."""

    def __init__(self, list_ts_surfaces, shaded):
        """Initialize using list of surfaces and shading status

        Parameters
        ----------
        list_ts_surfaces : \
        list of :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            List of timeseries surfaces in collection
        shaded : bool
            Shading status of the collection
        """
        self._list_ts_surfaces = list_ts_surfaces
        self.shaded = shaded
        # TODO: maybe the ts surfaces should have a "shaded" attribute

    @property
    def list_ts_surfaces(self):
        """List of timeseries surfaces in collection"""
        return self._list_ts_surfaces

    @property
    def length(self):
        """Total length of the collection"""
        length = 0.
        for ts_surf in self._list_ts_surfaces:
            length += ts_surf.length
        return length

    @property
    def n_ts_surfaces(self):
        """Number of timeseries surfaces in the collection"""
        return len(self._list_ts_surfaces)

    def get_param_weighted(self, param):
        """Get timeseries parameter for the collection, after weighting by
        surface length.

        Parameters
        ----------
        param : str
            Name of parameter

        Returns
        -------
        np.ndarray
            Weighted parameter values
        """
        return self.get_param_ww(param) / self.length

    def get_param_ww(self, param):
        """Get timeseries parameter from the collection with weight,
        i.e. after multiplying by the surface lengths.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        np.ndarray
            Timeseries parameter values multiplied by weights
        """

        value = 0
        for ts_surf in self._list_ts_surfaces:
            value += ts_surf.length * ts_surf.get_param(param)
        return value

    def update_params(self, new_dict):
        """Update timeseries surface parameters of the segment.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surfaces
        """
        for ts_surf in self._list_ts_surfaces:
            ts_surf.params.update(new_dict)

    def at(self, idx):
        """Generate a ponctual shade collection for the desired index.

        Parameters
        ----------
        idx : int
            Index to use to generate shade collection

        Returns
        -------
        collection : :py:class:`~pvfactors.geometry.base.ShadeCollection`
        """
        list_surfaces = [ts_surf.at(idx) for ts_surf in self._list_ts_surfaces
                         if not ts_surf.at(idx).is_empty]
        return ShadeCollection(list_surfaces, shaded=self.shaded)


class TsSurface(object):
    """Timeseries surface class: vectorized representation of PV surface
    geometries."""

    def __init__(self, coords, n_vector=None, param_names=None, index=None,
                 shaded=False):
        """Initialize timeseries surface using timeseries coordinates.

        Parameters
        ----------
        coords : :py:class:`~pvfactors.geometry.timeseries.TsLineCoords`
            Timeseries coordinates of full segment
        index : int, optional
            Index of segment (Default = None)
        n_vector : np.ndarray, optional
            Timeseries normal vectors of the side (Default = None)
        index : int, optional
            Index of the timeseries surfaces (Default = None)
        shaded : bool, optional
            Is the surface shaded or not (Default = False)
        """
        # TODO: ts surfaces should have a shaded attribute
        self.coords = coords
        self.param_names = [] if param_names is None else param_names
        # TODO: the following should probably be turned into properties,
        # because if the coords change, they won't be altered. But speed...
        self.n_vector = n_vector
        self.params = dict.fromkeys(self.param_names)
        self.index = index
        self.shaded = shaded

    def at(self, idx):
        """Generate a PV segment geometry for the desired index.

        Parameters
        ----------
        idx : int
            Index to use to generate PV segment geometry

        Returns
        -------
        segment : :py:class:`~pvfactors.geometry.base.PVSurface` \
        or :py:class:`~shapely.geometry.GeometryCollection`
            The returned object will be an empty geometry if its length is
            really small, otherwise it will be a PV surface geometry
        """
        if self.length[idx] < DISTANCE_TOLERANCE:
            # return an empty geometry
            return GeometryCollection()
        else:
            # Get normal vector at idx
            n_vector = (self.n_vector[:, idx] if self.n_vector is not None
                        else None)
            # Get params at idx
            # TODO: should find faster solution
            params = _get_params_at_idx(idx, self.params)
            # Return a pv surface geometry with given params
            return PVSurface(self.coords.at(idx), shaded=self.shaded,
                             index=self.index, normal_vector=n_vector,
                             param_names=self.param_names,
                             params=params)

    def plot_at_idx(self, idx, ax, color):
        """Plot timeseries PV row at a certain index, only if it's not
        too small.

        Parameters
        ----------
        idx : int
            Index to use to plot timeseries PV surface
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color_shaded : str, optional
            Color to use for plotting the PV surface
        """
        if self.length[idx] > DISTANCE_TOLERANCE:
            self.at(idx).plot(ax, color=color)

    @property
    def b1(self):
        """Timeseries coordinates of first boundary point"""
        return self.coords.b1

    @property
    def b2(self):
        """Timeseries coordinates of second boundary point"""
        return self.coords.b2

    @property
    def centroid(self):
        """Timeseries point coordinates of the surface's centroid"""
        return self.coords.centroid

    def get_param(self, param):
        """Get timeseries parameter values of surface

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        np.ndarray
            Timeseries parameter values
        """
        return self.params[param]

    def update_params(self, new_dict):
        """Update timeseries surface parameters.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surface
        """
        self.params.update(new_dict)

    @property
    def length(self):
        """Timeseries length of the surface"""
        return self.coords.length

    @property
    def highest_point(self):
        """Timeseries point coordinates of highest point of surface"""
        return self.coords.highest_point

    @property
    def lowest_point(self):
        """Timeseries point coordinates of lowest point of surface"""
        return self.coords.lowest_point

    @property
    def u_vector(self):
        """Vector orthogonal to the surface's normal vector"""
        u_vector = (None if self.n_vector is None else
                    np.array([-self.n_vector[1, :], self.n_vector[0, :]]))
        return u_vector

    @property
    def is_empty(self):
        """Check if surface is "empty" by checking if its length is always
        zero"""
        return np.nansum(self.length) < DISTANCE_TOLERANCE


class TsLineCoords(object):
    """Timeseries line coordinates class: will provide a helpful shapely-like
    API to invoke timeseries coordinates."""

    def __init__(self, b1_ts_coords, b2_ts_coords, coords=None):
        """Initialize timeseries line coordinates using the timeseries
        coordinates of its boundaries.

        Parameters
        ----------
        b1_ts_coords : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries coordinates of first boundary point
        b2_ts_coords : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries coordinates of second boundary point
        coords : np.ndarray, optional
            Timeseries coordinates as numpy array
        """
        self.b1 = b1_ts_coords
        self.b2 = b2_ts_coords

    def at(self, idx):
        """Get coordinates at a given index

        Parameters
        ----------
        idx : int
            Index to use to get coordinates
        """
        return self.as_array[:, :, idx]

    @classmethod
    def from_array(cls, coords_array):
        """Create timeseries line coordinates from numpy array of coordinates.

        Parameters
        ----------
        coords_array : np.ndarray
            Numpy array of coordinates.
        """
        b1 = TsPointCoords.from_array(coords_array[0, :, :])
        b2 = TsPointCoords.from_array(coords_array[1, :, :])
        return cls(b1, b2)

    @property
    def length(self):
        """Timeseries length of the line."""
        return np.sqrt((self.b2.y - self.b1.y)**2
                       + (self.b2.x - self.b1.x)**2)

    @property
    def as_array(self):
        """Timeseries line coordinates as numpy array"""
        return np.array([[self.b1.x, self.b1.y], [self.b2.x, self.b2.y]])

    @property
    def centroid(self):
        """Timeseries point coordinates of the line coordinates"""
        dy = self.b2.y - self.b1.y
        dx = self.b2.x - self.b1.x
        return TsPointCoords(self.b1.x + 0.5 * dx, self.b1.y + 0.5 * dy)

    @property
    def highest_point(self):
        """Timeseries point coordinates of highest point of timeseries
        line coords"""
        is_b1_highest = self.b1.y >= self.b2.y
        x = np.where(is_b1_highest, self.b1.x, self.b2.x)
        y = np.where(is_b1_highest, self.b1.y, self.b2.y)
        return TsPointCoords(x, y)

    @property
    def lowest_point(self):
        """Timeseries point coordinates of lowest point of timeseries
        line coords"""
        is_b1_highest = self.b1.y >= self.b2.y
        x = np.where(is_b1_highest, self.b2.x, self.b1.x)
        y = np.where(is_b1_highest, self.b2.y, self.b1.y)
        return TsPointCoords(x, y)

    def __repr__(self):
        """Use the numpy array representation of the coords"""
        return str(self.as_array)


class TsPointCoords(object):
    """Timeseries point coordinates: provides a shapely-like API for timeseries
    point coordinates."""

    def __init__(self, x, y):
        """Initialize timeseries point coordinates using numpy array of coords.

        Parameters
        ----------
        x : np.ndarray
            Timeseries x coordinates
        y : np.ndarray
            Timeseries y coordinates
        """
        self.x = x
        self.y = y

    def at(self, idx):
        """Get coordinates at a given index

        Parameters
        ----------
        idx : int
            Index to use to get coordinates
        """
        return self.as_array[:, idx]

    @property
    def as_array(self):
        """Timeseries point coordinates as numpy array"""
        return np.array([self.x, self.y])

    @classmethod
    def from_array(cls, coords_array):
        """Create timeseries point coords from numpy array of coordinates.

        Parameters
        ----------
        coords_array : np.ndarray
            Numpy array of coordinates.
        """
        return cls(coords_array[0, :], coords_array[1, :])

    def __repr__(self):
        """Use the numpy array representation of the point"""
        return str(self.as_array)


def _get_params_at_idx(idx, params_dict):
    """Get the parameter values at given index. Return the whole parameter
    when it's None, a scalar, or a dictionary

    Parameters
    ----------
    idx : int
        Index at which we want the parameter values
    params_dict : dict
        Dictionary of parameters

    Returns
    -------
    Parameter value at index
    """
    if params_dict is None:
        return None
    else:
        return {k: (val if (val is None) or np.isscalar(val)
                    or isinstance(val, dict) else val[idx])
                for k, val in params_dict.items()}
