"""Implement PV array classes, which will use PV rows and ground geometries"""
from pvfactors.geometry.pvground import PVGround
from pvfactors.geometry.pvrow import PVRow
from pvfactors.config import X_ORIGIN_PVROWS, COLOR_DIC, PLOT_FONTSIZE
from pvfactors.geometry.base import get_solar_2d_vector


class OrderedPVArray(object):
    """An ordered PV array has a flat horizontal ground, and pv rows which
    are all at the same height, with the same surface tilt and azimuth angles,
    and also all equally spaced. These simplifications allow faster and easier
    calculations."""

    y_ground = 0.  # ground will be at height = 0 by default

    def __init__(self, list_pvrows=[], ground=None, surface_tilt=None,
                 surface_azimuth=None, axis_azimuth=None, solar_zenith=None,
                 solar_azimuth=None, gcr=None, height=None, distance=None):
        self.pvrows = list_pvrows
        self.ground = ground
        self.gcr = gcr
        self.height = height
        self.distance = distance
        self.solar_zenith = solar_zenith
        self.solar_azimuth = solar_azimuth
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        self.axis_azimuth = axis_azimuth
        self.solar_2d_vector = get_solar_2d_vector(solar_zenith, solar_azimuth,
                                                   axis_azimuth)
        self.illum_side = None

    @classmethod
    def from_dict(cls, parameters):
        """Create ordered PV array from dictionary of parameters"""
        # Create ground
        ground = PVGround.as_flat(y_ground=cls.y_ground)
        # Create pvrows
        list_pvrows = []
        width = parameters['pvrow_width']
        tilt = parameters['surface_tilt']
        surface_azimuth = parameters['surface_azimuth']
        axis_azimuth = parameters['axis_azimuth']
        gcr = parameters['gcr']
        y_center = parameters['pvrow_height'] + cls.y_ground
        distance = width / gcr
        # Discretization params
        cut = parameters.get('cut', {})
        # Loop for pvrow creation
        for idx in range(parameters['n_pvrows']):
            # Make equally spaced pv rows
            x_center = X_ORIGIN_PVROWS + idx * distance
            pvrow = PVRow.from_center_tilt_width(
                (x_center, y_center), tilt, width, surface_azimuth,
                axis_azimuth, index=idx, cut=cut.get(idx, {}))
            list_pvrows.append(pvrow)
        return cls(list_pvrows=list_pvrows, ground=ground, surface_tilt=tilt,
                   surface_azimuth=surface_azimuth,
                   axis_azimuth=axis_azimuth,
                   solar_zenith=parameters['solar_zenith'],
                   solar_azimuth=parameters['solar_azimuth'],
                   gcr=gcr, height=y_center, distance=distance)

    def update_tilt(self, new_tilt):
        pass

    def cast_shadows(self):
        """Use calculated solar_2d_vector and array configuration to calculate
        shadows being casted in the array"""
        self.illum_side = (
            'front' if self.pvrows[0].front.n_vector.dot(
                self.solar_2d_vector) >= 0 else 'back')

    def plot(self, ax):
        """Plot PV array"""
        # Plot pv array structures
        self.ground.plot(ax, color_shaded=COLOR_DIC['ground_shaded'],
                         color_illum=COLOR_DIC['ground_illum'])
        for pvrow in self.pvrows:
            pvrow.plot(ax, color_shaded=COLOR_DIC['pvrow_shaded'],
                       color_illum=COLOR_DIC['pvrow_illum'])

        # Plot formatting
        ax.axis('equal')
        n_pvrows = len(self.pvrows)
        ax.set_xlim(- 0.5 * self.distance, (n_pvrows - 0.5) * self.distance)
        ax.set_ylim(- self.height, 2 * self.height)
        ax.set_xlabel("x [m]", fontsize=PLOT_FONTSIZE)
        ax.set_ylabel("y [m]", fontsize=PLOT_FONTSIZE)
