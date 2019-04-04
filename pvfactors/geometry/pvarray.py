"""Implement PV array classes, which will use PV rows and ground geometries"""
from pvfactors.geometry.pvground import PVGround
from pvfactors.geometry.pvrow import PVRow
from pvfactors.config import X_ORIGIN_PVROWS, COLOR_DIC, PLOT_FONTSIZE
from pvfactors.geometry.base import get_solar_2d_vector
from pvfactors.geometry.utils import projection
from shapely.geometry import LineString, Point


class OrderedPVArray(object):
    """An ordered PV array has a flat horizontal ground, and pv rows which
    are all at the same height, with the same surface tilt and azimuth angles,
    and also all equally spaced. These simplifications allow faster and easier
    calculations."""

    y_ground = 0.  # ground will be at height = 0 by default

    def __init__(self, list_pvrows=[], ground=None, surface_tilt=None,
                 surface_azimuth=None, axis_azimuth=None, solar_zenith=None,
                 solar_azimuth=None, gcr=None, height=None, distance=None):
        self.pvrows = list_pvrows  # ordered from left to right
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

        # Initialize shading attributes
        self.illum_side = None
        self.has_direct_shading = False

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
        shadows being casted in the ordered pv array.
        The logic here will be quite specific to ordered pv arrays"""
        self.illum_side = ('front' if self.pvrows[0].front.n_vector.dot(
            self.solar_2d_vector) >= 0 else 'back')
        last_gnd_2 = None
        last_gnd_1 = None
        # Cast pvrow shadows on ground
        for idx, pvrow in enumerate(self.pvrows):
            b1, b2 = pvrow.boundary
            gnd_1 = projection(b1, self.solar_2d_vector,
                               self.ground.original_linestring)
            gnd_2 = projection(b2, self.solar_2d_vector,
                               self.ground.original_linestring)
            self.ground.cast_shadow(LineString([gnd_1, gnd_2]))
            # Check that there's direct shading
            if idx == 1:
                # There's inter-row shading if ground shadows overlap
                if self.illum_side == 'front':
                    self.has_direct_shading = gnd_1.x < last_gnd_2.x
                else:
                    self.has_direct_shading = gnd_2.x < last_gnd_1.x
            last_gnd_2 = gnd_2
            last_gnd_1 = gnd_1

        # Calculate direct shading on pvrows
        sun_on_the_right = self.solar_2d_vector[0] >= 0
        if self.has_direct_shading:
            if sun_on_the_right:
                # right pvrow shades left pvrow
                shaded_pvrow = self.pvrows[0]
                front_pvrow = self.pvrows[1]
                proj_initial = projection(front_pvrow.highest_point,
                                          self.solar_2d_vector,
                                          shaded_pvrow.original_linestring)
                # Use distance translation to cast shadows on pvrows
                for idx, pvrow in enumerate(self.pvrows[:-1]):
                    proj = Point(proj_initial.x + idx * self.distance,
                                 proj_initial.y)
                    shaded_side = getattr(pvrow, self.illum_side)
                    shaded_side.cast_shadow(
                        LineString([pvrow.lowest_point, proj]))
            else:
                # left pvrow shades right pvrow
                shaded_pvrow = self.pvrows[1]
                front_pvrow = self.pvrows[0]
                proj_initial = projection(front_pvrow.highest_point,
                                          self.solar_2d_vector,
                                          shaded_pvrow.original_linestring)
                # Use distance translation to cast shadows on pvrows
                for idx, pvrow in enumerate(self.pvrows[1:]):
                    proj = Point(proj_initial.x + idx * self.distance,
                                 proj_initial.y)
                    shaded_side = getattr(pvrow, self.illum_side)
                    shaded_side.cast_shadow(
                        LineString([pvrow.lowest_point, proj]))
            # -----> merge ground shadows, since continuous

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
