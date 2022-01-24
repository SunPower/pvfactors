"""Base classes for pvfactors geometry subpackage."""

import numpy as np
from pvfactors import PVFactorsError
from pvfactors.config import (
    DEFAULT_NORMAL_VEC, COLOR_DIC, DISTANCE_TOLERANCE, PLOT_FONTSIZE,
    ALPHA_TEXT, MAX_X_GROUND)
from pvfactors.geometry.plot import plot_coords, plot_bounds, plot_line
from pvfactors.geometry.utils import \
    is_collinear, check_collinear, are_2d_vecs_collinear, difference, contains
from shapely.geometry import GeometryCollection, LineString
from shapely.geometry.collection import geos_geometrycollection_from_py
from shapely.ops import linemerge
from pvlib.tools import cosd, sind


def _check_uniform_shading(list_elements):
    """Check that all :py:class:`~pvfactors.geometry.base.PVSurface` objects in
    list have uniform shading

    Parameters
    ----------
    list_elements : list of :py:class:`~pvfactors.geometry.base.PVSurface`

    Raises
    ------
    PVFactorsError
        if all elements don't have the same shading flag
    """
    shaded = None
    for element in list_elements:
        if shaded is None:
            shaded = element.shaded
        else:
            is_uniform = shaded == element.shaded
            if not is_uniform:
                msg = "All elements should have same shading"
                raise PVFactorsError(msg)


def _coords_from_center_tilt_length(xy_center, tilt, length,
                                    surface_azimuth, axis_azimuth):
    """Calculate ``shapely`` :py:class:`LineString` coordinates from
    center coords, surface angles and length of line.
    The axis azimuth indicates the axis of rotation of the pvrows (if single-
    axis trackers). In the 2D plane, the axis of rotation will be the vector
    normal to that 2D plane and going into the 2D plane (when plotting it).
    The surface azimuth should always be 90 degrees away from the axis azimuth,
    either in the positive or negative direction.
    For instance, a single axis trk with axis azimuth = 0 deg (North), will
    have surface azimuth values equal to 90 deg (East) or 270 deg (West).
    Tilt angles need to always be positive. Given the axis azimuth and surface
    azimuth, a rotation angle will be derived. Positive rotation angles will
    indicate pvrows pointing to the left, and negative rotation angles will
    indicate pvrows pointing to the right (no matter what the the axis azimuth
    is).
    All of these conventions are necessary to make sure that no matter what
    the tilt and surface angles are, we can still identify correctly
    the same pv rows: the leftmost PV row will have index 0, and the rightmost
    will have index -1.

    Parameters
    ----------
    xy_center : tuple
        x, y coordinates of center point of desired linestring
    tilt : float or np.ndarray
        Surface tilt angles desired [deg]. Values should all be positive.
    length : float
        desired length of linestring [m]
    surface_azimuth : float or np.ndarray
        Surface azimuth angles of PV surface [deg]
    axis_azimuth : float
        Axis azimuth of the PV surface, i.e. direction of axis of rotation
        [deg]

    Returns
    -------
    list
        List of linestring coordinates obtained from inputs (could be vectors)
        in the form of [[x1, y1], [x2, y2]], where xi and yi could be arrays
        or scalar values.
    """
    # PV row params
    x_center, y_center = xy_center
    radius = length / 2.
    # Get rotation
    rotation = _get_rotation_from_tilt_azimuth(surface_azimuth, axis_azimuth,
                                               tilt)
    # Calculate coords
    x1 = radius * cosd(rotation + 180.) + x_center
    y1 = radius * sind(rotation + 180.) + y_center
    x2 = radius * cosd(rotation) + x_center
    y2 = radius * sind(rotation) + y_center

    return [[x1, y1], [x2, y2]]


def _get_rotation_from_tilt_azimuth(surface_azimuth, axis_azimuth, tilt):
    """Calculate the rotation angle using surface azimuth, axis azimuth,
    and surface tilt angles. While surface tilt angles need to always be
    positive, rotation angles can be negative.
    In pvfactors, positive rotation angles will indicate pvrows pointing to the
    left, and negative rotation angles will indicate pvrows pointing to the
    right (no matter what the the axis azimuth is).

    Parameters
    ----------
    tilt : float or np.ndarray
        Surface tilt angles desired [deg]. Values should all be positive.
    surface_azimuth : float or np.ndarray
        Surface azimuth angles of PV surface [deg]
    axis_azimuth : float
        Axis azimuth of the PV surface, i.e. direction of axis of rotation
        [deg]

    Returns
    -------
    float or np.ndarray
        Calculated rotation angle(s) in [deg]
    """

    # Calculate rotation of PV row (signed tilt angle)
    is_pointing_right = ((surface_azimuth - axis_azimuth) % 360.) > 180.
    rotation = np.where(is_pointing_right, tilt, -tilt)
    rotation[tilt == 0] = -0.0  # GH 125
    return rotation


def _get_solar_2d_vectors(solar_zenith, solar_azimuth, axis_azimuth):
    """Projection of 3d solar vector onto the cross section of the systems:
    which is the 2D plane we are considering.
    This is needed to calculate shadows.
    Remember that the 2D plane is such that the direction of the torque
    tube vector (or rotation axis) goes into (and normal to) the 2D plane,
    such that positive rotation angles will have the PV surfaces tilted to the
    LEFT and vice versa.

    Parameters
    ----------
    solar_zenith : float or numpy array
        Solar zenith angle [deg]
    solar_azimuth : float or numpy array
        Solar azimuth angle [deg]
    axis_azimuth : float
        Axis azimuth of the PV surface, i.e. direction of axis of rotation
        [deg]

    Returns
    -------
    solar_2d_vector : numpy array
        Two vector components of the solar vector in the 2D plane, with the
        form [x, y], where x and y can be arrays
    """
    solar_2d_vector = np.array([
        # a drawing really helps understand the following
        sind(solar_zenith) * cosd(solar_azimuth - axis_azimuth - 90.),
        cosd(solar_zenith)])

    return solar_2d_vector


class BaseSurface(LineString):
    """Base surfaces will be extensions of :py:class:`LineString` classes,
    but adding an orientation to it (normal vector).
    So two surfaces could use the same linestring, but have opposite
    orientations."""

    def __init__(self, coords, normal_vector=None, index=None,
                 param_names=None, params=None):
        """Create a surface using linestring coordinates.
        Normal vector can have two directions for a given LineString,
        so the user can provide it in order to be specific,
        otherwise it will be automatically
        calculated, but then the surface won't know if it was supposed to be
        pointing "up" or "down". If the surface is empty, the normal vector
        will take the default value.

        Parameters
        ----------
        coords : list
            List of linestring coordinates for the surface
        normal_vector : list, optional
            Normal vector for the surface (Default = None, so will be
            calculated)
        index : int, optional
            Surface index (Default = None)
        param_names : list of str, optional
            Names of the surface parameters, eg reflectivity, total incident
            irradiance, temperature, etc. (Default = None)
        params : dict, optional
            Surface float parameters (Default = None)
        """

        param_names = [] if param_names is None else param_names
        super(BaseSurface, self).__init__(coords)
        if normal_vector is None:
            self.n_vector = self._calculate_n_vector()
        else:
            self.n_vector = np.array(normal_vector)
        self.index = index
        self.param_names = param_names
        self.params = params if params is not None \
            else dict.fromkeys(self.param_names)

    def _calculate_n_vector(self):
        """Calculate normal vector of the surface, if surface is not empty"""
        if not self.is_empty:
            b1, b2 = self.boundary
            dx = b2.x - b1.x
            dy = b2.y - b1.y
            return np.array([-dy, dx])
        else:
            return DEFAULT_NORMAL_VEC

    def plot(self, ax, color=None, with_index=False):
        """Plot the surface on the given axes.

        Parameters
        ----------
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color : str, optional
            Color to use for plotting the surface (Default = None)
        with_index : bool
            Flag to annotate surfaces with their indices (Default = False)
        """
        plot_coords(ax, self)
        plot_bounds(ax, self)
        plot_line(ax, self, color)
        if with_index:
            # Prepare text location
            v = self.n_vector
            v_norm = v / np.linalg.norm(v)
            centroid = self.centroid
            alpha = ALPHA_TEXT
            x = centroid.x + alpha * v_norm[0]
            y = centroid.y + alpha * v_norm[1]
            # Add text
            # FIXME: hack to get a nice plot in jupyter notebook
            if np.abs(x) < MAX_X_GROUND / 2.:
                ax.text(x, y, '{}'.format(self.index),
                        verticalalignment='center',
                        horizontalalignment='center')

    def difference(self, linestring):
        """Calculate remaining surface after removing part belonging from
        provided linestring,

        Parameters
        ----------
        linestring : :py:class:`shapely.geometry.LineString`
            Line string to remove from surface

        Returns
        -------
        :py:class:`shapely.geometry.LineString`
           Resulting difference of current surface minus given linestring
        """
        return difference(self, linestring)

    def get_param(self, param):
        """Get parameter value from surface.

        Parameters
        ----------
        param : str
            Surface parameter to return

        Returns
        -------
        Parameter value to return

        Raises
        ------
        KeyError
            if parameter name not in the surface parameters
        """
        return self.params[param]

    def update_params(self, new_dict):
        """Update surface parameters.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surface
        """
        self.params.update(new_dict)


class PVSurface(BaseSurface):
    """PV surfaces inherit from
    :py:class:`~pvfactors.geometry.base.BaseSurface`. The only difference is
    that PV surfaces have a ``shaded`` attribute.
    """

    def __init__(self, coords=None, normal_vector=None, shaded=False,
                 index=None, param_names=None, params=None):
        """Initialize PV surface.

        Parameters
        ----------
        coords : list, optional
            List of linestring coordinates for the surface
        normal_vector : list, optional
            Normal vector for the surface (Default = None, so will be
            calculated)
        shaded : bool, optional
            Flag telling if surface is shaded or not (Default = False)
        index : int, optional
            Surface index (Default = None)
        param_names : list of str, optional
            Names of the surface parameters, eg reflectivity, total incident
            irradiance, temperature, etc. (Default = None)
        params : dict, optional
            Surface float parameters (Default = None)
        """

        param_names = [] if param_names is None else param_names
        super(PVSurface, self).__init__(coords, normal_vector, index=index,
                                        param_names=param_names, params=params)
        self.shaded = shaded


class ShadeCollection(GeometryCollection):
    """A group of :py:class:`~pvfactors.geometry.base.PVSurface`
    objects that all have the same shading status. The PV surfaces are not
    necessarily contiguous or collinear."""

    def __init__(self, list_surfaces=None, shaded=None, param_names=None):
        """Initialize shade collection.

        Parameters
        ----------
        list_surfaces : list, optional
            List of :py:class:`~pvfactors.geometry.base.PVSurface` object
            (Default = None)
        shaded : bool, optional
            Shading status of the collection. If not specified, will be derived
            from list of surfaces (Default = None)
        param_names : list of str, optional
            Names of the surface parameters, eg reflectivity, total incident
            irradiance, temperature, etc. (Default = None)

        """
        list_surfaces = [] if list_surfaces is None else list_surfaces
        param_names = [] if param_names is None else param_names
        _check_uniform_shading(list_surfaces)
        self.list_surfaces = list_surfaces
        self.shaded = self._get_shading(shaded)
        self.is_collinear = is_collinear(list_surfaces)
        self.param_names = param_names
        super(ShadeCollection, self).__init__(list_surfaces)

    def _get_shading(self, shaded):
        """Get the surface shading from the provided list of pv surfaces.

        Parameters
        ----------
        shaded : bool
            Shading flag passed during initialization

        Returns
        -------
        bool
            Shading status of the collection
        """
        if len(self.list_surfaces):
            return self.list_surfaces[0].shaded
        else:
            return shaded

    def plot(self, ax, color=None, with_index=False):
        """Plot the surfaces in the shade collection.

        Parameters
        ----------
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color : str, optional
            Color to use for plotting the surface (Default = None)
        with_index : bool
            Flag to annotate surfaces with their indices (Default = False)
        """
        for surface in self.list_surfaces:
            surface.plot(ax, color=color, with_index=with_index)

    def add_linestring(self, linestring, normal_vector=None):
        """Add PV surface to the collection using a linestring

        Parameters
        ----------
        linestring : :py:class:`shapely.geometry.LineString`
            Linestring to use to add a PV surface to the collection
        normal_vector : list, optional
            Normal vector to use for the PV surface to create (Default = None,
            will try to get it from collection)
        """
        if normal_vector is None:
            normal_vector = self.n_vector
        surf = PVSurface(coords=linestring.coords,
                         normal_vector=normal_vector, shaded=self.shaded,
                         param_names=self.param_names)
        self.add_pvsurface(surf)

    def add_pvsurface(self, pvsurface):
        """Add PV surface to the collection.

        Parameters
        ----------
        pvsurface : :py:class:`~pvfactors.geometry.base.PVSurface`
            PV Surface to add to collection
        """
        self.list_surfaces.append(pvsurface)
        self.is_collinear = is_collinear(self.list_surfaces)
        super(ShadeCollection, self).__init__(self.list_surfaces)

    def remove_linestring(self, linestring):
        """Remove linestring from shade collection.
        The method will rearrange the PV surfaces to make it work.

        Parameters
        ----------
        linestring : :py:class:`shapely.geometry.LineString`
            Line string to remove from the collection (by differencing)
        """
        new_list_surfaces = []
        for surface in self.list_surfaces:
            # Need to use buffer for intersects bc of floating point precision
            # errors in shapely
            if surface.buffer(DISTANCE_TOLERANCE).intersects(linestring):
                difference = surface.difference(linestring)
                # We want to make sure we can iterate on it, as
                # ``difference`` can be a multi-part geometry or not
                if not hasattr(difference, '__iter__'):
                    difference = [difference]
                for new_geom in difference:
                    if not new_geom.is_empty:
                        new_surface = PVSurface(
                            new_geom.coords, normal_vector=surface.n_vector,
                            shaded=surface.shaded,
                            param_names=surface.param_names)
                        new_list_surfaces.append(new_surface)
            else:
                new_list_surfaces.append(surface)

        self.list_surfaces = new_list_surfaces
        # Force update, even if list is empty
        self.update_geom_collection(self.list_surfaces)

    def update_geom_collection(self, list_surfaces):
        """Force update of geometry collection, even if list is empty
        https://github.com/Toblerity/Shapely/blob/master/shapely/geometry/collection.py#L42

        Parameters
        ----------
        list_surfaces : list of :py:class:`~pvfactors.geometry.base.PVSurface`
            New list of PV surfaces to update the shade collection in place
        """
        self._geom, self._ndim = geos_geometrycollection_from_py(list_surfaces)

    def merge_surfaces(self):
        """Merge all surfaces in the shade collection into one contiguous
        surface, even if they're not contiguous, by using bounds."""
        if len(self.list_surfaces) > 1:
            merged_lines = linemerge(self.list_surfaces)
            minx, miny, maxx, maxy = merged_lines.bounds
            surf_1 = self.list_surfaces[0]
            new_pvsurf = PVSurface(
                coords=[(minx, miny), (maxx, maxy)],
                shaded=self.shaded, normal_vector=surf_1.n_vector,
                param_names=surf_1.param_names)
            self.list_surfaces = [new_pvsurf]
            self.update_geom_collection(self.list_surfaces)

    def cut_at_point(self, point):
        """Cut collection at point if the collection contains it.

        Parameters
        ----------
        point : :py:class:`shapely.geometry.Point`
            Point where to cut collection geometry, if the latter contains the
            former
        """
        for idx, surface in enumerate(self.list_surfaces):
            if contains(surface, point):
                # Make sure that not hitting a boundary
                b1, b2 = surface.boundary
                not_hitting_b1 = b1.distance(point) > DISTANCE_TOLERANCE
                not_hitting_b2 = b2.distance(point) > DISTANCE_TOLERANCE
                if not_hitting_b1 and not_hitting_b2:
                    coords_1 = [b1, point]
                    coords_2 = [point, b2]
                    # TODO: not sure what to do about index yet
                    new_surf_1 = PVSurface(
                        coords_1, normal_vector=surface.n_vector,
                        shaded=surface.shaded,
                        param_names=surface.param_names)
                    new_surf_2 = PVSurface(
                        coords_2, normal_vector=surface.n_vector,
                        shaded=surface.shaded,
                        param_names=surface.param_names)
                    # Now update collection
                    self.list_surfaces[idx] = new_surf_1
                    self.list_surfaces.append(new_surf_2)
                    self.update_geom_collection(self.list_surfaces)
                    # No need to continue the loop
                    break

    def get_param_weighted(self, param):
        """Get the parameter from the collection's surfaces, after weighting
        by surface length.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        float
            Weighted parameter value
        """
        value = self.get_param_ww(param) / self.length
        return value

    def get_param_ww(self, param):
        """Get the parameter from the collection's surfaces with weight, i.e.
        after multiplying by the surface lengths.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        float
            Parameter value multiplied by weights

        Raises
        ------
        KeyError
            if parameter name not in a surface parameters
        """
        value = 0
        for surf in self.list_surfaces:
            value += surf.get_param(param) * surf.length
        return value

    def update_params(self, new_dict):
        """Update surface parameters in the collection.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surface
        """
        for surf in self.list_surfaces:
            surf.update_params(new_dict)

    @property
    def n_vector(self):
        """Unique normal vector of the shade collection, if it exists."""
        if not self.is_collinear:
            msg = "Cannot request n_vector if all elements not collinear"
            raise PVFactorsError(msg)
        if len(self.list_surfaces):
            return self.list_surfaces[0].n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def n_surfaces(self):
        """Number of surfaces in collection."""
        return len(self.list_surfaces)

    @property
    def surface_indices(self):
        """Indices of the surfaces in the collection."""
        return [surf.index for surf in self.list_surfaces]

    @classmethod
    def from_linestring_coords(cls, coords, shaded, normal_vector=None,
                               param_names=None):
        """Create a shade collection with a single PV surface.

        Parameters
        ----------
        coords : list
            List of linestring coordinates for the surface
        shaded : bool
            Shading status desired for the collection
        normal_vector : list, optional
            Normal vector for the surface (Default = None)
        param_names : list of str, optional
            Names of the surface parameters, eg reflectivity, total incident
            irradiance, temperature, etc. (Default = None)
        """
        surf = PVSurface(coords=coords, normal_vector=normal_vector,
                         shaded=shaded, param_names=param_names)
        return cls([surf], shaded=shaded, param_names=param_names)


class PVSegment(GeometryCollection):
    """A PV segment will be a collection of 2 collinear and contiguous
    shade collections, a shaded one and an illuminated one. It inherits from
    :py:class:`shapely.geometry.GeometryCollection`  so that users can still
    call basic geometrical methods and properties on it, eg call length, etc.
    """

    def __init__(self, illum_collection=ShadeCollection(shaded=False),
                 shaded_collection=ShadeCollection(shaded=True), index=None):
        """Initialize PV segment.

        Parameters
        ----------
        illum_collection : \
        :py:class:`~pvfactors.geometry.base.ShadeCollection`, optional
            Illuminated collection of the PV segment (Default = empty shade
            collection with no shading)
        shaded_collection : \
        :py:class:`~pvfactors.geometry.base.ShadeCollection`, optional
            Shaded collection of the PV segment (Default = empty shade
            collection with shading)
        index : int, optional
            Index of the PV segment (Default = None)
        """
        assert shaded_collection.shaded, "surface should be shaded"
        assert not illum_collection.shaded, "surface should not be shaded"
        self._check_collinear(illum_collection, shaded_collection)
        self._shaded_collection = shaded_collection
        self._illum_collection = illum_collection
        self.index = index
        self._all_surfaces = None
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    def _check_collinear(self, illum_collection, shaded_collection):
        """Check that all the surfaces in the PV segment are collinear.

        Parameters
        ----------
        illum_collection :
        :py:class:`~pvfactors.geometry.base.ShadeCollection`, optional
            Illuminated collection
        shaded_collection :
        :py:class:`~pvfactors.geometry.base.ShadeCollection`, optional
            Shaded collection

        Raises
        ------
        PVFactorsError
            If all the surfaces are not collinear
        """
        assert illum_collection.is_collinear
        assert shaded_collection.is_collinear
        # Check that if none or all of the collection is empty, n_vectors are
        # equal
        if (not illum_collection.is_empty) \
           and (not shaded_collection.is_empty):
            n_vec_ill = illum_collection.n_vector
            n_vec_shaded = shaded_collection.n_vector
            assert are_2d_vecs_collinear(n_vec_ill, n_vec_shaded)

    def plot(self, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
             color_illum=COLOR_DIC['pvrow_illum'], with_index=False):
        """Plot the surfaces in the PV Segment.

        Parameters
        ----------
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color_shaded : str, optional
            Color to use for plotting the shaded surfaces (Default =
            COLOR_DIC['pvrow_shaded'])
        color_shaded : str, optional
            Color to use for plotting the illuminated surfaces (Default =
            COLOR_DIC['pvrow_illum'])
        with_index : bool
            Flag to annotate surfaces with their indices (Default = False)
        """
        self._shaded_collection.plot(ax, color=color_shaded,
                                     with_index=with_index)
        self._illum_collection.plot(ax, color=color_illum,
                                    with_index=with_index)

    def cast_shadow(self, linestring):
        """Cast shadow on PV segment using linestring: will rearrange the
        PV surfaces between the shaded and illuminated collections of the
        segment

        Parameters
        ----------
        linestring : :py:class:`shapely.geometry.LineString`
            Linestring casting a shadow on the PV segment
        """
        # Using a buffer may slow things down, but it's quite crucial
        # in order for shapely to get the intersection accurately see:
        # https://stackoverflow.com/questions/28028910/how-to-deal-with-rounding-errors-in-shapely
        intersection = (self._illum_collection.buffer(DISTANCE_TOLERANCE)
                        .intersection(linestring))
        if not intersection.is_empty:
            # Split up only if interesects the illuminated collection
            # print(intersection)
            self._shaded_collection.add_linestring(intersection,
                                                   normal_vector=self.n_vector)
            # print(self._shaded_collection.length)
            self._illum_collection.remove_linestring(intersection)
            # print(self._illum_collection.length)
            super(PVSegment, self).__init__([self._shaded_collection,
                                             self._illum_collection])

    def cut_at_point(self, point):
        """Cut PV segment at point if the segment contains it.

        Parameters
        ----------
        point : :py:class:`shapely.geometry.Point`
            Point where to cut collection geometry, if the latter contains the
            former
        """
        if contains(self, point):
            if contains(self._illum_collection, point):
                self._illum_collection.cut_at_point(point)
            else:
                self._shaded_collection.cut_at_point(point)

    def get_param_weighted(self, param):
        """Get the parameter from the segment's surfaces, after weighting
        by surface length.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        float
            Weighted parameter value
        """
        value = self.get_param_ww(param) / self.length
        return value

    def get_param_ww(self, param):
        """Get the parameter from the segment's surfaces with weight, i.e.
        after multiplying by the surface lengths.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        float
            Parameter value multiplied by weights

        Raises
        ------
        KeyError
            if parameter name not in a surface parameters
        """
        value = 0
        value += self._shaded_collection.get_param_ww(param)
        value += self._illum_collection.get_param_ww(param)
        return value

    def update_params(self, new_dict):
        """Update surface parameters in the collection.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surfaces
        """
        self._shaded_collection.update_params(new_dict)
        self._illum_collection.update_params(new_dict)

    @property
    def n_vector(self):
        """Since shaded and illum surfaces are supposed to be collinear,
        this should return either surfaces' normal vector. If both empty,
        return Default value for normal vector."""
        if not self.illum_collection.is_empty:
            return self.illum_collection.n_vector
        elif not self.shaded_collection.is_empty:
            return self.shaded_collection.n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def n_surfaces(self):
        """Number of surfaces in collection."""
        n_surfaces = self._illum_collection.n_surfaces \
            + self._shaded_collection.n_surfaces
        return n_surfaces

    @property
    def surface_indices(self):
        """Indices of the surfaces in the PV segment."""
        list_indices = []
        list_indices += self._illum_collection.surface_indices
        list_indices += self._shaded_collection.surface_indices
        return list_indices

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None, param_names=None):
        """Create a PV segment with a single PV surface.

        Parameters
        ----------
        coords : list
            List of linestring coordinates for the surface
        shaded : bool, optional
            Shading status desired for the resulting PV surface
            (Default = False)
        normal_vector : list, optional
            Normal vector for the surface (Default = None)
        index : int, optional
            Index of the segment (Default = None)
        param_names : list of str, optional
            Names of the surface parameters, eg reflectivity, total incident
            irradiance, temperature, etc. (Default = None)
        """
        col = ShadeCollection.from_linestring_coords(
            coords, shaded=shaded, normal_vector=normal_vector,
            param_names=param_names)
        # Realized that needed to instantiate other_col, otherwise could
        # end up with shared collection among different PV segments
        other_col = ShadeCollection(list_surfaces=[], shaded=not shaded,
                                    param_names=param_names)
        if shaded:
            return cls(illum_collection=other_col,
                       shaded_collection=col, index=index)
        else:
            return cls(illum_collection=col,
                       shaded_collection=other_col, index=index)

    @property
    def shaded_collection(self):
        """Shaded collection of the PV segment"""
        return self._shaded_collection

    @shaded_collection.setter
    def shaded_collection(self, new_collection):
        """Set shaded collection of the PV segment with new one.

        Parameters
        ----------
        new_collection : :py:class:`pvfactors.geometry.base.ShadeCollection`
            New collection to use for update
        """
        assert new_collection.shaded, "surface should be shaded"
        self._shaded_collection = new_collection
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @shaded_collection.deleter
    def shaded_collection(self):
        """Delete shaded collection of PV segment and replace with empty one.
        """
        self._shaded_collection = ShadeCollection(shaded=True)
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @property
    def illum_collection(self):
        """Illuminated collection of the PV segment."""
        return self._illum_collection

    @illum_collection.setter
    def illum_collection(self, new_collection):
        """Set illuminated collection of the PV segment with new one.

        Parameters
        ----------
        new_collection : :py:class:`pvfactors.geometry.base.ShadeCollection`
            New collection to use for update
        """
        assert not new_collection.shaded, "surface should not be shaded"
        self._illum_collection = new_collection
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @illum_collection.deleter
    def illum_collection(self):
        """Delete illuminated collection of PV segment and replace with empty
        one."""
        self._illum_collection = ShadeCollection(shaded=False)
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @property
    def shaded_length(self):
        """Length of the shaded collection of the PV segment.

        Returns
        -------
        float
            Length of the shaded collection
        """
        return self._shaded_collection.length

    @property
    def all_surfaces(self):
        """List of all the :py:class:`pvfactors.geometry.base.PVSurface`

        Returns
        -------
        list of :py:class:`~pvfactors.geometry.base.PVSurface`
            PV surfaces in the PV segment
        """
        if self._all_surfaces is None:
            self._all_surfaces = []
            self._all_surfaces += self._illum_collection.list_surfaces
            self._all_surfaces += self._shaded_collection.list_surfaces
        return self._all_surfaces


class BaseSide(GeometryCollection):
    """A side represents a fixed collection of PV segments objects that should
    all be collinear, with the same normal vector"""

    def __init__(self, list_segments=None):
        """Create a side geometry.

        Parameters
        ----------
        list_segments : list of :py:class:`~pvfactors.geometry.base.PVSegment`, optional
            List of PV segments for side (Default = None)
        """
        list_segments = [] if list_segments is None else list_segments
        check_collinear(list_segments)
        self.list_segments = tuple(list_segments)
        self._all_surfaces = None
        super(BaseSide, self).__init__(list_segments)

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None, n_segments=1, param_names=None):
        """Create a Side with a single PV surface, or multiple discretized
        identical ones.

        Parameters
        ----------
        coords : list
            List of linestring coordinates for the surface
        shaded : bool, optional
            Shading status desired for the resulting PV surface
            (Default = False)
        normal_vector : list, optional
            Normal vector for the surface (Default = None)
        index : int, optional
            Index of the segments (Default = None)
        n_segments : int, optional
            Number of same-length segments to use (Default = 1)
        param_names : list of str, optional
            Names of the surface parameters, eg reflectivity, total incident
            irradiance, temperature, etc. (Default = None)
        """
        if n_segments == 1:
            list_pvsegments = [PVSegment.from_linestring_coords(
                coords, shaded=shaded, normal_vector=normal_vector,
                index=index, param_names=param_names)]
        else:
            # Discretize coords and create segments accordingly
            linestring = LineString(coords)
            fractions = np.linspace(0., 1., num=n_segments + 1)
            list_points = [linestring.interpolate(fraction, normalized=True)
                           for fraction in fractions]
            list_pvsegments = []
            for idx in range(n_segments):
                new_coords = list_points[idx:idx + 2]
                # TODO: not clear what to do with the index here
                pvsegment = PVSegment.from_linestring_coords(
                    new_coords, shaded=shaded, normal_vector=normal_vector,
                    index=index, param_names=param_names)
                list_pvsegments.append(pvsegment)
        return cls(list_segments=list_pvsegments)

    @property
    def n_vector(self):
        """Normal vector of the Side."""
        if len(self.list_segments):
            return self.list_segments[0].n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def shaded_length(self):
        """Shaded length of the Side."""
        shaded_length = 0.
        for segment in self.list_segments:
            shaded_length += segment.shaded_length
        return shaded_length

    @property
    def n_surfaces(self):
        """Number of surfaces in the Side object."""
        n_surfaces = 0
        for segment in self.list_segments:
            n_surfaces += segment.n_surfaces
        return n_surfaces

    @property
    def all_surfaces(self):
        """List of all surfaces in the Side object."""
        if self._all_surfaces is None:
            self._all_surfaces = []
            for segment in self.list_segments:
                self._all_surfaces += segment.all_surfaces
        return self._all_surfaces

    @property
    def surface_indices(self):
        """List of all surface indices in the Side object."""
        list_indices = []
        for seg in self.list_segments:
            list_indices += seg.surface_indices
        return list_indices

    def plot(self, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
             color_illum=COLOR_DIC['pvrow_illum'], with_index=False):
        """Plot the surfaces in the Side object.

        Parameters
        ----------
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting
        color_shaded : str, optional
            Color to use for plotting the shaded surfaces (Default =
            COLOR_DIC['pvrow_shaded'])
        color_shaded : str, optional
            Color to use for plotting the illuminated surfaces (Default =
            COLOR_DIC['pvrow_illum'])
        with_index : bool
            Flag to annotate surfaces with their indices (Default = False)
        """
        for segment in self.list_segments:
            segment.plot(ax, color_shaded=color_shaded,
                         color_illum=color_illum, with_index=with_index)

    def cast_shadow(self, linestring):
        """Cast shadow on Side using linestring: will rearrange the
        PV surfaces between the shaded and illuminated collections of the
        segments.

        Parameters
        ----------
        linestring : :py:class:`shapely.geometry.LineString`
            Linestring casting a shadow on the Side object
        """
        for segment in self.list_segments:
            segment.cast_shadow(linestring)

    def merge_shaded_areas(self):
        """Merge shaded areas of all PV segments"""
        for seg in self.list_segments:
            seg._shaded_collection.merge_surfaces()

    def cut_at_point(self, point):
        """Cut Side at point if the side contains it.

        Parameters
        ----------
        point : :py:class:`shapely.geometry.Point`
            Point where to cut side geometry, if the latter contains the
            former
        """
        if contains(self, point):
            for segment in self.list_segments:
                # Nothing will happen to the segments that do not contain
                # the point
                segment.cut_at_point(point)

    def get_param_weighted(self, param):
        """Get the parameter from the side's surfaces, after weighting
        by surface length.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        float
            Weighted parameter value
        """
        value = self.get_param_ww(param) / self.length
        return value

    def get_param_ww(self, param):
        """Get the parameter from the side's surfaces with weight, i.e.
        after multiplying by the surface lengths.

        Parameters
        ----------
        param: str
            Surface parameter to return

        Returns
        -------
        float
            Parameter value multiplied by weights

        Raises
        ------
        KeyError
            if parameter name not in a surface parameters
        """
        value = 0
        for seg in self.list_segments:
            value += seg.get_param_ww(param)
        return value

    def update_params(self, new_dict):
        """Update surface parameters in the Side.

        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surfaces
        """
        for seg in self.list_segments:
            seg.update_params(new_dict)


class BasePVArray(object):
    """Base class for PV arrays in pvfactors. Will provide basic
    capabilities."""

    registry_cols = ['geom', 'line_type', 'pvrow_index', 'side',
                     'pvsegment_index', 'shaded', 'surface_index']

    def __init__(self, axis_azimuth=None):
        """Initialize Base of PV array.

        Parameters
        ----------
        axis_azimuth : float, optional
            Azimuth angle of rotation axis [deg] (Default = None)
        """
        # All PV arrays should have a fixed axis azimuth in pvfactors
        self.axis_azimuth = axis_azimuth

        # The are required attributes of any PV array
        self.ts_pvrows = None
        self.ts_ground = None

    @property
    def n_ts_surfaces(self):
        """Number of timeseries surfaces in the PV array."""
        n_ts_surfaces = 0
        n_ts_surfaces += self.ts_ground.n_ts_surfaces
        for ts_pvrow in self.ts_pvrows:
            n_ts_surfaces += ts_pvrow.n_ts_surfaces
        return n_ts_surfaces

    @property
    def all_ts_surfaces(self):
        """List of all timeseries surfaces in PV array"""
        all_ts_surfaces = []
        all_ts_surfaces += self.ts_ground.all_ts_surfaces
        for ts_pvrow in self.ts_pvrows:
            all_ts_surfaces += ts_pvrow.all_ts_surfaces
        return all_ts_surfaces

    @property
    def ts_surface_indices(self):
        """List of indices of all the timeseries surfaces"""
        return [ts_surf.index for ts_surf in self.all_ts_surfaces]

    def plot_at_idx(self, idx, ax, merge_if_flag_overlap=True,
                    with_cut_points=True, x_min_max=None,
                    with_surface_index=False):
        """Plot all the PV rows and the ground in the PV array at a desired
        step index. This can be called before transforming the array, and
        after fitting it.

        Parameters
        ----------
        idx : int
            Selected timestep index for plotting the PV array
        ax : :py:class:`matplotlib.pyplot.axes` object
            Axes for plotting the PV array geometries
        merge_if_flag_overlap : bool, optional
            Decide whether to merge all shadows if they overlap
            (Default = True)
        with_cut_points :  bool, optional
            Decide whether to include the saved cut points in the created
            PV ground geometry (Default = True)
        x_min_max : tuple, optional
            List of minimum and maximum x coordinates for the flat ground
            surface [m] (Default = None)
        with_surface_index : bool, optional
            Plot the surfaces with their index values (Default = False)
        """
        # Plot pv array structures
        self.ts_ground.plot_at_idx(
            idx, ax, color_shaded=COLOR_DIC['ground_shaded'],
            color_illum=COLOR_DIC['ground_illum'],
            merge_if_flag_overlap=merge_if_flag_overlap,
            with_cut_points=with_cut_points, x_min_max=x_min_max,
            with_surface_index=with_surface_index)

        for ts_pvrow in self.ts_pvrows:
            ts_pvrow.plot_at_idx(
                idx, ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                color_illum=COLOR_DIC['pvrow_illum'],
                with_surface_index=with_surface_index)

        # Plot formatting
        ax.axis('equal')
        if self.distance is not None:
            n_pvrows = self.n_pvrows
            ax.set_xlim(- 0.5 * self.distance,
                        (n_pvrows - 0.5) * self.distance)
        if self.height is not None:
            ax.set_ylim(- self.height, 2 * self.height)
        ax.set_xlabel("x [m]", fontsize=PLOT_FONTSIZE)
        ax.set_ylabel("y [m]", fontsize=PLOT_FONTSIZE)

    def fit(self, *args, **kwargs):
        """Not implemented."""
        raise NotImplementedError

    def update_params(self, new_dict):
        """Update timeseries surface parameters in the collection.
        Parameters
        ----------
        new_dict : dict
            Parameters to add or update for the surfaces
        """
        self.ts_ground.update_params(new_dict)
        for ts_pvrow in self.ts_pvrows:
            ts_pvrow.update_params(new_dict)

    def _index_all_ts_surfaces(self):
        """Add unique indices to all surfaces in the PV array."""
        for idx, ts_surface in enumerate(self.all_ts_surfaces):
            ts_surface.index = idx
