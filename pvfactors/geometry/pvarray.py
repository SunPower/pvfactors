"""Implement PV array classes, which will use PV rows and ground geometries"""
from pvfactors.geometry.pvground import PVGround
from pvfactors.geometry.pvrow import PVRow


class OrderedPVArray(object):
    """An ordered PV array has a flat horizontal ground, and pv rows which
    are all at the same height, with the same surface tilt and azimuth angles,
    and also all equally spaced. These simplifications allow faster and easier
    calculations."""

    def __init__(self, list_pvrows=[], ground=None):
        self.pvrows = list_pvrows
        self.ground = ground

    @classmethod
    def from_dict(cls, parameters):
        ground = PVGround.as_flat()
        list_pvrows = []
        for idx in range(parameters['n_pvrows']):
            pass
        return cls(list_pvrows=list_pvrows, ground=ground)

    def update_tilt(self, new_tilt):
        pass

    def cast_shadows(self, sun_vector):
        pass

    def plot(self, ax):
        pass
