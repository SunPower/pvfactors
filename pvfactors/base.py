import numpy as np
from pvfactors import PVFactorsError
from pvfactors.config import DEFAULT_NORMAL_VEC, TOL_COLLINEAR
from shapely.geometry import GeometryCollection


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
