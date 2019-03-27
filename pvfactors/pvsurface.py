from shapely.geometry import LineString, GeometryCollection


class BaseSurface(LineString):
    """Base surfaces will be extensions of :py:class:`LineString` classes,
    but adding an orientation to it. So two surfaces could use the same
    linestring, but have opposite orientations."""

    def __init__(self, coords, normal_vector):

        super(BaseSurface, self).__init__(coords)
        self.n_vec = normal_vector


class PVSurfaceRigid(BaseSurface):
    """Rigid PV surfaces are extensions of base surfaces, as they add PV
    related properties and methods, like reflectivity, shading, etc. They can
    only represent only 1 ``shapely`` :py:class:`LineString` geometry."""

    def __init__(self, coords=None, normal_vector=None, reflectivity=0.03,
                 shaded=False):

        super(PVSurfaceRigid, self).__init__(coords, normal_vector)
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


class PVSurfaceFluid(GeometryCollection):
    """A fluid PV surface will be a collection of 2 rigid PV surfaces,
    a shaded one and an unshaded one. It inherits from ``shapely``
    :py:class:`GeometryCollection` so that we can still call basic geometrical
    methods and properties on it, eg like length (and get sum of lengths
    here)"""

    def __init__(self, illum_surface=PVSurfaceRigid(shaded=False),
                 shaded_surface=PVSurfaceRigid(shaded=True)):
        assert shaded_surface.shaded, "surface should be shaded"
        assert not illum_surface.shaded, "surface should not be shaded"
        self._shaded_surface = shaded_surface
        self._illum_surface = illum_surface
        super(PVSurfaceFluid, self).__init__([self._shaded_surface,
                                              self._illum_surface])

    @property
    def shaded_surface(self):
        return self._shaded_surface

    @shaded_surface.setter
    def shaded_surface(self, new_surface):
        assert new_surface.shaded, "surface should be shaded"
        self._shaded_surface = new_surface
        super(PVSurfaceFluid, self).__init__([self._shaded_surface,
                                              self._illum_surface])

    @shaded_surface.deleter
    def shaded_surface(self):
        self._shaded_surface = PVSurfaceRigid()
        super(PVSurfaceFluid, self).__init__([self._shaded_surface,
                                              self._illum_surface])

    @property
    def illum_surface(self):
        return self._illum_surface

    @illum_surface.setter
    def illum_surface(self, new_surface):
        assert not new_surface.shaded, "surface should not be shaded"
        self._illum_surface = new_surface
        super(PVSurfaceFluid, self).__init__([self._shaded_surface,
                                              self._illum_surface])

    @illum_surface.deleter
    def illum_surface(self):
        self._illum_surface = PVSurfaceRigid()
        super(PVSurfaceFluid, self).__init__([self._shaded_surface,
                                              self._illum_surface])
