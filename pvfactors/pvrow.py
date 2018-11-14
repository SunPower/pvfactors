# -*- coding: utf-8 -*-

from pvfactors import PVFactorsError
from pvfactors.pvcore import LinePVArray, Y_GROUND
from shapely.geometry import LineString, Point
from shapely.affinity import affine_transform
import numpy as np


class PVRowBase(object):
    """
    ``PVRowBase`` exists for future developments of the model. It is the
     base class for PV Rows that will contain all the boiler plate code
     shared by sub classes like :class:`PVRowLine`, or for instance
     ``PVRowRoof``.
    """

    def __init__(self):

        self.front = None
        self.back = None
        self.shadow = None
        self.shadow_line_index = None
        self.lines = []
        self.highest_point = None  # for shading purposes
        self.lowest_point = None  # for shading purposes
        self.left_point = None  # for shading purposes
        self.right_point = None  # for shading purposes
        self.director_vector = None  # for shading purposes
        self.normal_vector = None  # for shading purposes
        # Complete line will have the full PVRow linestring with possibly
        # multiple points on it, but still one linestring only
        self.complete_linestring = None  # for obstruction purposes

    def create_lines(self, *args, **kwargs):
        raise NotImplementedError

    def get_shadow_bounds(self, *args, **kwargs):
        raise NotImplementedError

    def translate_2d_lines(self, x_off, y_off):
        """
        Translate all the :class:`pvfactors.components.LinePVArray` objects in
        the line attribute in the x and y directions.
        /!\ method is unfinished

        :param float x_off: translation in the x direction
        :param float y_off: translation in the y direction
        :return: None
        """
        matrix_2d_translation = [1, 1, 1, 1, x_off, y_off]
        for line in self.lines:
            line['shapeline'] = affine_transform(line['shapeline'],
                                                 matrix_2d_translation)
        # and update lines in line_registry


class PVRowLine(PVRowBase):
    """
    ``PVRowLine`` is sub-classed from :class:`PVRowBase`, and its core is
    made of :class:`pvfactors.components.LinePVArray` objects. It is a
    container that has methods and attributes relative to PV Rows and their
    shadows.
    ``PVRowLine`` can only create PV rows that have the shape of
    straight lines. So it won't be able to create shapes like dual-tilt
    systems for instance.

    :param line_registry: line registry passed by :class:`pvarray.Array`
        object
    :type line_registry: :class:`pvcore.Registry`
    :param float x_center: x coordinate of center of the line [m]
    :param float y_center: y coordinate of center of the line [m]
    :param int index: PV row index, used to distinguish different PV rows
    :param float surface_tilt: Surface tilt angles in decimal degrees.
        surface_tilt must be >=0 and <=180.
        The tilt angle is defined as degrees from horizontal
    :param float pvrow_width: width of the PV row, which is the length of
    the PV row line [m]
    """

    def __init__(self, line_registry, x_center, y_center, index, surface_tilt,
                 pvrow_width):
        super(PVRowLine, self).__init__()
        assert ((surface_tilt >= 0) and (surface_tilt < 180)), (
            "Range of surface_tilt incorrect: {}".format(surface_tilt))
        self.width = pvrow_width
        self.tilt = surface_tilt
        self.index = index
        self.x_center = x_center
        self.y_center = y_center
        (self.lines, self.highest_point, self.lowest_point, self.right_point,
         self.left_point) = (
            self.create_lines(self.tilt, index))
        self.line_registry_indices = line_registry.pvgeometry.add(self.lines)
        # Complete line will have the full pvrow linestring with possibly
        # multiple points on it, but still one linestring only
        self.complete_linestring = self.lines[0]['geometry']
        self.cut_points = []

    def create_lines(self, tilt, index):
        """
        Create the :class:`pvcore.LinePVArray` objects that the PV row
        is made out of, based on the inputted geometrical parameters.

        :param float tilt: Surface tilt angles in decimal degrees.
            surface_tilt must be >=0 and <=180.
            The tilt angle is defined as degrees from horizontal
        :param int index: PV row index, used to distinguish different PV rows
        :return: [line_pvarray], highest_point, lowest_point,
                right_point, left_point / which
                are: list of :class:`pvcore.LinePVArray`;
                :class:`shapely.Point` of the line with biggest y coordinate;
                :class:`shapely.Point` of the line with smallest y coordinate;
                :class:`shapely.Point` of the line with biggest x coordinate;
                :class:`shapely.Point` of the line with smallest x coordinate
        """
        tilt_rad = np.radians(tilt)

        # Create the three trackers
        radius = self.width / 2.
        x1 = radius * np.cos(tilt_rad + np.pi) + self.x_center
        y1 = radius * np.sin(tilt_rad + np.pi) + self.y_center
        x2 = radius * np.cos(tilt_rad) + self.x_center
        y2 = radius * np.sin(tilt_rad) + self.y_center

        highest_point = Point(x2, y2)
        lowest_point = Point(x1, y1)
        right_point = Point(x2, y2) if x2 >= x1 else Point(x1, y1)
        left_point = Point(x1, y1) if x2 >= x1 else Point(x2, y2)

        geometry = LineString([(x1, y1), (x2, y2)])
        line_pvarray = LinePVArray(geometry=geometry, line_type='pvrow',
                                   shaded=False, pvrow_index=index)
        return ([line_pvarray], highest_point, lowest_point,
                right_point, left_point)

    def get_shadow_bounds(self, solar_2d_vector):
        """
        Calculate the x coordinates of the boundary points of the shadow lines
        on the ground, assuming Y_GROUND is the y coordinate of the ground.
        Note: this shadow construction is more or less ignored when direct
        shading happens between rows, leading to one continous shadows formed
        by all the PV rows in the array.

        :param list solar_2d_vector: projection of solar vector into the 2D
        plane of the array geometry
        :return: x1_shadow, x2_shadow // ``float`` smallest x coordinate of
        shadow, ``float`` largest x coordinate of shadow
        """
        list_x_values = []
        for line in self.lines:
            geometry = line['geometry']
            b1 = geometry.boundary[0]
            b2 = geometry.boundary[1]
            # consider line equation: u*x + v*y + c = 0
            # for lines 1 and 2 (parallel)
            u = - solar_2d_vector[1]
            v = solar_2d_vector[0]
            # Find intercepts for lines 1 and 2
            c_1 = - (u * b1.x + v * b1.y)
            c_2 = - (u * b2.x + v * b2.y)
            # Find intersections of ray lines with horizontal ground line
            x1_shadow = - (c_1 + v * Y_GROUND) / u
            x2_shadow = - (c_2 + v * Y_GROUND) / u
            list_x_values.append(x1_shadow)
            list_x_values.append(x2_shadow)
        x1_shadow = min(list_x_values)
        x2_shadow = max(list_x_values)
        return x1_shadow, x2_shadow

    @property
    def facing(self):
        """
        This property is mainly used to calculate the view_matrix
        :return: direction that the pvrow front surfaces are facing
        """
        if self.tilt == 0.:
            direction = 'up'
        elif self.tilt > 0:
            direction = 'left'
        elif self.tilt < 0:
            direction = 'right'
        else:
            raise PVFactorsError("Unknown facing condition for pvrow")
        return direction

    def calculate_cut_points(self, n_segments):
        """
        Calculate the points of the PV row geometry on which the PV line will
        be cut and discretized. The list of cut points is saved into the object

        :param int n_segments: number of segments wanted for the discretization
        :return: None
        """
        fractions = np.linspace(0., 1., num=n_segments + 1)[1:-1]
        list_points = [self.complete_linestring.interpolate(fraction,
                                                            normalized=True)
                       for fraction in fractions]
        self.cut_points = list_points
