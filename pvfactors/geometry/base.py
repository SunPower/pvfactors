"""Base classes for geometry subpackage."""

import numpy as np
from pvfactors import PVFactorsError
from pvfactors.config import DEFAULT_NORMAL_VEC, TOL_COLLINEAR
from shapely.geometry import GeometryCollection, LineString
from pvlib.tools import cosd, sind


def is_collinear(list_elements):
    """Check that all :py:class:`~pvfactors.pvsurface.PVSegment`
    or :py:class:`~pvfactors.pvsurface.PVSurface` objects in list
    are collinear"""
    is_col = True
    u_direction = None  # will be orthogonal to normal vector
    for element in list_elements:
        if not element.is_empty:
            if u_direction is None:
                # set u_direction if not defined already
                u_direction = np.array([- element.n_vector[1],
                                        element.n_vector[0]])
            else:
                # check that collinear
                dot_prod = u_direction.dot(element.n_vector)
                is_col = np.abs(dot_prod) < TOL_COLLINEAR
                if not is_col:
                    return is_col
    return is_col


def check_collinear(list_elements):
    """Raise error if all :py:class:`~pvfactors.pvsurface.PVSegment`
    or :py:class:`~pvfactors.pvsurface.PVSurface` objects in list
    are not collinear"""
    is_col = is_collinear(list_elements)
    if not is_col:
        msg = "All elements should be collinear"
        raise PVFactorsError(msg)


def check_uniform_shading(list_elements):
    """Check that all :py:class:`~pvfactors.pvsurface.PVSurface` objects in
    list are collinear"""
    shaded = None
    for element in list_elements:
        if shaded is None:
            shaded = element.shaded
        else:
            is_uniform = shaded == element.shaded
            if not is_uniform:
                msg = "All elements should have same shading"
                raise PVFactorsError(msg)


def are_2d_vecs_collinear(u1, u2):
    n1 = np.array([-u1[1], u1[0]])
    dot_prod = n1.dot(u2)
    return np.abs(dot_prod) < TOL_COLLINEAR


def coords_from_center_tilt_length(xy_center, tilt, length):
    """Calculate ``shapely`` :py:class:`LineString` coordinate from
    center coords, surface tilt angle and length of line"""
    # Create the three trackers
    x_center, y_center = xy_center
    radius = length / 2.
    x1 = radius * cosd(tilt + 180.) + x_center
    y1 = radius * sind(tilt + 180.) + y_center
    x2 = radius * cosd(tilt) + x_center
    y2 = radius * sind(tilt) + y_center

    return [(x1, y1), (x2, y2)]


class BaseSurface(LineString):
    """Base surfaces will be extensions of :py:class:`LineString` classes,
    but adding an orientation to it. So two surfaces could use the same
    linestring, but have opposite orientations."""

    def __init__(self, coords, normal_vector=None):
        """Normal vector can have two directions for a given LineString,
        so the user can provide it in order to be specific,
        otherwise it will be automatically
        calculated, but then the surface won't know if it was supposed to be
        pointing "up" or "down". If the surface is empty, the normal vector
        will take the default value."""

        super(BaseSurface, self).__init__(coords)
        if normal_vector is None:
            self.n_vector = self._calculate_n_vector()
        else:
            self.n_vector = np.array(normal_vector)

    def _calculate_n_vector(self):
        """Calculate normal vector of the surface if not empty"""
        if not self.is_empty:
            b1, b2 = self.boundary
            dx = b2.x - b1.x
            dy = b2.y - b1.y
            return np.array([-dy, dx])
        else:
            return DEFAULT_NORMAL_VEC


class ShadeCollection(GeometryCollection):
    """A group :py:class:`~pvfactors.pvsurface.PVSurface`
    objects that have the same shading status. The pv surfaces are not
    necessarily contiguous or collinear."""

    def __init__(self, list_surfaces=[], shaded=None):
        """
        Parameters
        ----------
        shaded : bool, optional
            Will be used only if no surfaces were passed
        """
        check_uniform_shading(list_surfaces)
        self.list_surfaces = list_surfaces
        self.shaded = self._get_shading(shaded)
        self.is_collinear = is_collinear(list_surfaces)
        super(ShadeCollection, self).__init__(list_surfaces)

    def _get_shading(self, shaded):
        """Get the surface shading from the provided list of pv surfaces."""
        if len(self.list_surfaces):
            return self.list_surfaces[0].shaded
        else:
            return shaded

    @property
    def n_vector(self):
        if not self.is_collinear:
            msg = "Cannot request n_vector if all elements not collinear"
            raise PVFactorsError(msg)
        if len(self.list_surfaces):
            return self.list_surfaces[0].n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @classmethod
    def from_linestring_coords(cls, coords, shaded, normal_vector=None):
        surf = PVSurface(coords=coords, normal_vector=normal_vector)
        return cls([surf], shaded=shaded)


class BaseSide(GeometryCollection):
    """A side represents a fixed collection of
    :py:class:`~pvfactors.pvsurfaces.PVSegment` objects that should
    all be collinear, with the same normal vector"""

    def __init__(self, list_segments=[]):
        check_collinear(list_segments)
        self.list_segments = tuple(list_segments)
        super(BaseSide, self).__init__(list_segments)

    @property
    def n_vector(self):
        if len(self.list_segments):
            return self.list_segments[0].n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @property
    def shaded_length(self):
        shaded_length = 0.
        for segment in self.list_segments:
            shaded_length += segment.shaded_length
        return shaded_length

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None):
        list_pvsegments = [PVSegment.from_linestring_coords(
            coords, shaded=shaded, normal_vector=normal_vector, index=index)]
        return cls(list_pvsegments=list_pvsegments)


class PVSurface(BaseSurface):
    """Rigid PV surfaces are extensions of base surfaces, as they add PV
    related properties and methods, like reflectivity, shading, etc. They can
    only represent only 1 ``shapely`` :py:class:`LineString` geometry."""

    def __init__(self, coords=None, normal_vector=None, reflectivity=0.03,
                 shaded=False, index=None):

        super(PVSurface, self).__init__(coords, normal_vector)
        self.reflectivity = reflectivity
        self.shaded = shaded
        self.index = index

    @classmethod
    def from_center_tilt_width(cls, center, tilt, width, reflectivity=0.03,
                               shaded=False):
        # TODO: Calculate necessary info to instantiate
        coords = None
        normal_vector = None

        return cls(coords, normal_vector, reflectivity=reflectivity,
                   shaded=shaded)


class PVSegment(GeometryCollection):
    """A PV segment will be a collection of 2 collinear and contiguous
    PV surfaces, a shaded one and an illuminated one. It inherits from
    ``shapely``
    :py:class:`GeometryCollection` so that we can still call basic geometrical
    methods and properties on it, eg call length and get sum of lengths
    here of the two surfaces"""

    def __init__(self, illum_collection=ShadeCollection(shaded=False),
                 shaded_collection=ShadeCollection(shaded=True), index=None):
        assert shaded_collection.shaded, "surface should be shaded"
        assert not illum_collection.shaded, "surface should not be shaded"
        self._check_collinear(illum_collection, shaded_collection)
        self._shaded_collection = shaded_collection
        self._illum_collection = illum_collection
        self.index = index
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    def _check_collinear(self, illum_collection, shaded_collection):
        assert illum_collection.is_collinear
        assert shaded_collection.is_collinear
        # Check that if none or all of the collection is empty, n_vectors are
        # equal
        if (not illum_collection.is_empty) \
           and (not shaded_collection.is_empty):
            n_vec_ill = illum_collection.n_vector
            n_vec_shaded = shaded_collection.n_vector
            assert are_2d_vecs_collinear(n_vec_ill, n_vec_shaded)

    @property
    def n_vector(self):
        """Since shaded and illum surfaces are supposed to be collinear,
        this should return either surfaces' normal vector. If both empty,
        return None."""
        if not self.illum_collection.is_empty:
            return self.illum_collection.n_vector
        elif not self.shaded_collection.is_empty:
            return self.shaded_collection.n_vector
        else:
            return DEFAULT_NORMAL_VEC

    @classmethod
    def from_linestring_coords(cls, coords, shaded=False, normal_vector=None,
                               index=None):
        """Create a PVSegment from the coordinates of a single ``shapely``
        :py:class:`LineString`"""
        col = ShadeCollection.from_linestring_coords(
            coords, shaded=shaded, normal_vector=normal_vector)
        if shaded:
            return cls(shaded_collection=col, index=index)
        else:
            return cls(illum_collection=col, index=index)

    @property
    def shaded_collection(self):
        return self._shaded_collection

    @shaded_collection.setter
    def shaded_collection(self, new_collection):
        assert new_collection.shaded, "surface should be shaded"
        self._shaded_collection = new_collection
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @shaded_collection.deleter
    def shaded_collection(self):
        self._shaded_collection = ShadeCollection(shaded=True)
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @property
    def illum_collection(self):
        return self._illum_collection

    @illum_collection.setter
    def illum_collection(self, new_collection):
        assert not new_collection.shaded, "surface should not be shaded"
        self._illum_collection = new_collection
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @illum_collection.deleter
    def illum_collection(self):
        self._illum_collection = ShadeCollection(shaded=False)
        super(PVSegment, self).__init__([self._shaded_collection,
                                         self._illum_collection])

    @property
    def shaded_length(self):
        return self._shaded_collection.length
