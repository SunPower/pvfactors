"""Implement PV array classes, which will use PV rows and ground geometries"""
from pvfactors.geometry.pvground import PVGround
from pvfactors.geometry.pvrow import PVRow
from pvfactors.config import X_ORIGIN_PVROWS


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
        # Create ground
        ground = PVGround.as_flat()
        # Create pvrows
        list_pvrows = []
        width = parameters['pvrow_width']
        tilt = parameters['surface_tilt']
        y_center = parameters['pvrow_height']
        distance = width / parameters['gcr']
        for idx in range(parameters['n_pvrows']):
            x_center = X_ORIGIN_PVROWS + idx * distance
            pvrow = PVRow.from_center_tilt_width(
                (x_center, y_center), tilt, width, index=idx)
            list_pvrows.append(pvrow)
        return cls(list_pvrows=list_pvrows, ground=ground)

    def update_tilt(self, new_tilt):
        pass

    def cast_shadows(self, sun_vector):
        pass

    def plot(self, ax):
        pass
