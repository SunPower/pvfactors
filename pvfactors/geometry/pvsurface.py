"""Classes related to pv surfaces and segments."""

from pvfactors.config import DEFAULT_NORMAL_VEC
from shapely.geometry import GeometryCollection
from pvfactors.geometry.base import \
    ShadeCollection, are_2d_vecs_collinear, BaseSurface


class PVSurface(BaseSurface):
    """Rigid PV surfaces are extensions of base surfaces, as they add PV
    related properties and methods, like reflectivity, shading, etc. They can
    only represent only 1 ``shapely`` :py:class:`LineString` geometry."""

    def __init__(self, coords=None, normal_vector=None, reflectivity=0.03,
                 shaded=False):

        super(PVSurface, self).__init__(coords, normal_vector)
        self.reflectivity = reflectivity
        self.shaded = shaded

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
                 shaded_collection=ShadeCollection(shaded=True)):
        assert shaded_collection.shaded, "surface should be shaded"
        assert not illum_collection.shaded, "surface should not be shaded"
        self._check_collinear(illum_collection, shaded_collection)
        self._shaded_collection = shaded_collection
        self._illum_collection = illum_collection
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
