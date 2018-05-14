# -*- coding: utf-8 -*-

import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
import math
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# TODO: hard coding these values is not ideal
MAX_X_GROUND = 1e2
MIN_X_GROUND = - MAX_X_GROUND
Y_GROUND = 0.
THRESHOLD_EDGE_POINT = 1e3
RADIUS_CIRCUMSOLAR = 1.

# TODO: hard-coding for registry
DISTANCE_TOLERANCE = 1e-8
THRESHOLD_DISTANCE_TOO_CLOSE = 1e-10

# Gaussian shading default parameters: TOTAL_GAUSSIAN_AREA dependent on these
SIGMA = 1. / np.sqrt(2.)
MU = 0.
N_SIGMA = 3.
GAUSSIAN_DIAMETER_CIRCUMSOLAR = 2. * N_SIGMA * SIGMA


class Registry(gpd.GeoDataFrame):
    """
    The ``Registry`` class is used to store and index most of the
    information, inputs and outputs of the calculations performed in the
    view factor model, including the ``shapely`` geometries. It is the
    entity passed by the :class:`pvarray.Array` to the
    :class:`view_factor.ViewFactorCalculator`.
    It is subclassed from :class:`geopandas.GeoDataFrame`, and contains
    additional methods to handle the ``shapely`` geometry manipulation.
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def add(self, list_lines_pvarray):
        """
        Add list of objects of class :class:`pvcore.LinePVArray` to the
        registry.

        :param list_lines_pvarray: list of objects of type
            :class:`pvcore.LinePVArray`
        :return: ``idx_list`` -- ``int``, the list of registry indices that were
        added to the registry
        """
        # Find the start index that will be used to add entries to the registry
        if len(self.index) > 0:
            start_index = self.index[-1] + 1
        else:
            start_index = 0
        idx_list = []
        # Loop through list of PV array lines
        for counter, line_pvarray in enumerate(list_lines_pvarray):
            idx = start_index + counter
            for key in line_pvarray.keys():
                self.loc[idx, key] = line_pvarray[key]
            idx_list.append(idx)

        return idx_list

    def split_ground_geometry_from_edge_points(self, edge_points):
        """
        Break up ground lines into multiple ones at the pv row "edge points",
        which are the intersections of pv row lines and ground lines. This is
        important to separate the ground lines that a pv row's front surface
         sees from the ones its back surface does.

        :param edge_points: list of :class:`shapely.Point` objects
        :return: None
        """
        for point in edge_points:
            gdf_ground = self.loc[self.loc[:, 'line_type'] == 'ground', :]
            geoentry_to_break_up = gdf_ground.loc[gdf_ground.contains(point)]
            if geoentry_to_break_up.shape[0] == 1:
                self.break_and_add_entries(geoentry_to_break_up, point)
            elif geoentry_to_break_up.shape[0] > 1:
                raise Exception("geoentry_to_break_up.shape[0] cannot be larger"
                                " than 1")

    def split_pvrow_geometry(self, idx, line_shadow, pvrow_top_point):
        """
        Break up pv row line into two pv row lines, a shaded one and an unshaded
         one. This function requires knowing the pv row line index in the
        registry, the "shadow line" that intersects with the pv row, and the top
         point of the pv row in order to decide which pv row line will be shaded
         or not after break up.

        :param int idx: index of shaded pv row entry
        :param line_shadow: :class:`shapely.LineString` object representing the
            "shadow line" intersecting with the pv row line
        :param pvrow_top_point: the highest point of the pv row line (in the
            elevation direction)
        :return: None
        """
        # Define geometry to work on
        geometry = self.loc[idx, 'geometry']
        # Find intersection point
        point_intersect = geometry.intersection(line_shadow)
        # Check that the intersection is not too close to a boundary: if it
        # is it can create a "memory access error" it seems
        is_too_close = [
            point.distance(point_intersect) < THRESHOLD_DISTANCE_TOO_CLOSE
            for point in geometry.boundary]
        if True in is_too_close:
            # Leave it as it is and do not split geometry: it should not
            # affect the calculations for a good threshold
            pass
        else:
            # Cut geometry in two pieces
            list_new_lines = self.cut_linestring(geometry, point_intersect)
            # Add new geometry to index
            new_registry_entry = self.loc[idx, :].copy()
            new_registry_entry['shaded'] = True
            if pvrow_top_point in list_new_lines[0].boundary:
                geometry_ill = gpd.GeoSeries(list_new_lines[0])
                geometry_shaded = gpd.GeoSeries(list_new_lines[1])
            elif pvrow_top_point in list_new_lines[1].boundary:
                geometry_ill = gpd.GeoSeries(list_new_lines[1])
                geometry_shaded = gpd.GeoSeries(list_new_lines[0])
            else:
                raise Exception("split_pvrow_geometry: unknown error occured")

            # Update registry
            self.at[idx, 'geometry'] = geometry_ill.values[0]
            new_registry_entry['geometry'] = geometry_shaded.values[0]
            self.loc[self.shape[0] + 1, :] = new_registry_entry

    def cut_pvrow_geometry(self, list_points, pvrow_index, side):
        """
        Break up pv row lines into multiple segments based on the list of points
         specified. This is the "discretization" of the pvrow segments. For now,
         it only works for pv rows.

        :param list_points: list of :class:`shapely.Point`, breaking points for
            the pv row lines.
        :param pvrow_index: pv row index to specify the PV row to discretize;
            note that this could return multiple entries from the registry.
        :param side: only do it for one side of the selected PV row.
        :return: None
        """
        # TODO: is currently not able to work for other surfaces than pv rows...
        for point in list_points:
            gdf_selected = self.loc[(self['pvrow_index'] == pvrow_index)
                                    & (self['surface_side'] == side), :]
            geoentry_to_break_up = gdf_selected.loc[
                gdf_selected.distance(point) < DISTANCE_TOLERANCE]
            if geoentry_to_break_up.shape[0] == 1:
                self.break_and_add_entries(geoentry_to_break_up, point)
            elif geoentry_to_break_up.shape[0] > 1:
                raise Exception("geoentry_to_break_up.shape[0] cannot be larger"
                                " than 1")

    def break_and_add_entries(self, geoentry_to_break_up, point):
        """
        Break up a surface geometry into two objects at a point location.

        :param geoentry_to_break_up: registry entry to break up
        :param point: :class:`shapely.Point` object used to decide where to
            break up entry.
        :return: None
        """
        # Get geometry
        idx = geoentry_to_break_up.index
        geometry = geoentry_to_break_up.geometry.values[0]
        line_1, line_2 = self.cut_linestring(geometry, point)
        geometry_1 = gpd.GeoSeries(line_1)
        geometry_2 = gpd.GeoSeries(line_2)
        self.at[idx, 'geometry'] = geometry_1.values
        new_registry_entry = self.loc[idx, :].copy()
        new_registry_entry['geometry'] = geometry_2.values
        self.loc[self.shape[0], :] = new_registry_entry.values[0]

    @staticmethod
    def cut_linestring(line, point):
        """
        Adapted from shapely documentation. Cuts a line in two at a calculated
        distance from its starting point

        :param line: :class:`shapely.LineString` object to cut
        :param point: :class:`shapely.Point` object to use for the cut
        :return: list of two :class:`shapely.LineString` objects
        """

        distance = line.project(point)
        assert ((distance >= 0.0) & (distance <= line.length)), (
            "cut_linestring: the lines didn't intersect")
        # There could be multiple points in a line
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            if pd == distance:
                return [
                    LineString(coords[:i + 1]),
                    LineString(coords[i:])]
            if pd > distance:
                cp = line.interpolate(distance)
                return [
                    LineString(coords[:i] + [(cp.x, cp.y)]),
                    LineString([(cp.x, cp.y)] + coords[i:])]


class LinePVArray(dict):

    _list_line_types = ['pvrow', 'ground', None]

    def __init__(self, geometry=None, style='-', line_type=None,
                 shaded=None, pvrow_index=None):
        """
        ``LinePVArray`` is the general class that is used to instantiate all the
        initial line objects of the pv array before putting them into the
        surface registry.
        It is a sub-class of a dictionary with already defined keys.

        :param geometry: ``shapely`` geometry object
        :param str style: ``matplotlib`` plotting style for the line. E.g. '--'.
        :param str line_type: type of surface in the :class:`pvarray.Array`,
            e.g. 'pvrow' or 'ground'
        :param bool shaded: specifies if surface is shaded (from direct shading)
        :param pvrow_index: if the surface's ``line_type`` is a 'pvrow', this
            will be its pv row index (which is different from its
            :attr:`pvarray.Array.surface_registry` index)
        """
        if line_type in self._list_line_types:
            super(LinePVArray, self).__init__(geometry=geometry, style=style,
                                              line_type=line_type,
                                              shaded=shaded,
                                              pvrow_index=pvrow_index)
        else:
            raise ValueError("'line_type' cannot be: %s, \n possible values "
                             "are: %s" % (str(line_type),
                                          str(self._list_line_types)))


class EdgePointDoesNotExist(Exception):
    pass


class VFArrayUpdateException(Exception):
    pass


def calculate_circumsolar_shading(percentage_distance_covered,
                                  model='uniform_disk'):
    """
    Select the model to calculate circumsolar shading based on the current PV
    array condition.

    :param float percentage_distance_covered: this represents how much of the
        circumsolar diameter is covered by the neighboring row [in %]
    :param str model: name of the circumsolar shading model to use:
        'uniform_disk' and 'gaussian' are the two models currently available
    :return: a ``float`` representing the percentage shading of the
        circumsolar disk [in %]
    """
    if model == 'uniform_disk':
        perc_shading = uniform_circumsolar_disk_shading(
            percentage_distance_covered)

    elif model == 'gaussian':
        perc_shading = gaussian_shading(percentage_distance_covered)

    else:
        raise Exception('calculate_circumsolar_shading: model does not exist: '
                        + '%s' % model)

    return perc_shading


def integral_default_gaussian(y, x):
    """
    Calculate the value of the integral from x to y of the erf function

    :param float y: upper limit
    :param float x: lower limit
    :return: ``float``, value of the integral
    """
    return 0.5 * (math.erf(x) - math.erf(y))


def gaussian_shading(percentage_distance_covered):
    """
    Calculate circumsolar shading by assuming that the irradiance profile on
    a 2D section of the circumsolar disk is Gaussian

    :param float percentage_distance_covered: [in %], proportion of the
        circumsolar disk covered by the neighboring pvrow
    :return: ``float`` representing the total circumsolar shading percentage
        in [%]
    """
    if percentage_distance_covered < 0.:
        perc_shading = 0.
    elif percentage_distance_covered > 100.:
        perc_shading = 100.
    else:
        y = - N_SIGMA * SIGMA
        x = (y
             + percentage_distance_covered / 100. *
             GAUSSIAN_DIAMETER_CIRCUMSOLAR)
        area = integral_default_gaussian(y, x)
        total_gaussian_area = integral_default_gaussian(- N_SIGMA * SIGMA,
                                                        N_SIGMA * SIGMA)
        perc_shading = area / total_gaussian_area * 100.

    return perc_shading


def gaussian(x, mu=0., sigma=1.):
    """
    Gaussian density function

    :param float x: argument of function
    :param float mu: mean of the gaussian
    :param float sigma: standard deviation of the gaussian
    :return: ``float``, value of the density function evaluated at ``x``
    """
    return (1. / (sigma * np.sqrt(2. * np.pi))
            * np.exp(- 0.5 * np.power((x - mu) / sigma, 2)))


def uniform_circumsolar_disk_shading(percentage_distance_covered):
    """
    Calculate the percentage shading of circumsolar irradiance. This
    model considers circumsolar to be a disk, and calculates the
    percentage shaded based on the percentage "distance covered",
    which is how much of the disk's diameter is covered by the
    neighboring object.

    :param float percentage_distance_covered: distance covered of
    circumsolar disk diameter [% - values from 0 to 100]
    :return: ``percent_shading``, circumsolar disk shading percentage [%]
    """

    # Define a circumsolar disk
    r_circumsolar = RADIUS_CIRCUMSOLAR
    d_circumsolar = 2 * r_circumsolar
    area_circumsolar = np.pi * r_circumsolar**2.

    # Calculate circumsolar on case by case
    distance_covered = percentage_distance_covered / 100. * d_circumsolar

    if distance_covered <= 0:
        # No shading of circumsolar
        percent_shading = 0.
    elif (distance_covered > 0.) & (distance_covered <= r_circumsolar):
        # Theta is the full circle sector angle (not half) used to
        # calculate circumsolar shading
        theta = 2 * np.arccos((r_circumsolar - distance_covered) /
                              r_circumsolar)  # rad
        area_circle_sector = 0.5 * r_circumsolar ** 2 * theta
        area_triangle_sector = 0.5 * r_circumsolar ** 2 * np.sin(theta)
        area_shaded = area_circle_sector - area_triangle_sector
        percent_shading = area_shaded / area_circumsolar * 100.

    elif (distance_covered > r_circumsolar) & (distance_covered
                                               < d_circumsolar):
        distance_uncovered = d_circumsolar - distance_covered

        # Theta is the full circle sector angle (not half) used to
        # calculate circumsolar shading
        theta = 2 * np.arccos((r_circumsolar - distance_uncovered) /
                              r_circumsolar)  # rad
        area_circle_sector = 0.5 * r_circumsolar ** 2 * theta
        area_triangle_sector = 0.5 * r_circumsolar ** 2 * np.sin(theta)
        area_not_shaded = area_circle_sector - area_triangle_sector
        percent_shading = (1. - area_not_shaded / area_circumsolar) * 100.

    else:
        # Total shading of circumsolar: distance_covered >= d_circumsolar
        percent_shading = 100.

    return percent_shading


def calculate_horizon_band_shading(shading_angle, horizon_band_angle):
    """
    Calculate the percentage shading of the horizon band; which is just the
    proportion of what's masked by the neighboring row since we assume
    uniform luminance values for the band

    :param float shading_angle: the elevation angle to use for shading
    :param float horizon_band_angle: the total elevation angle of the horizon
        band
    :return: ``float``, percentage shading of the horizon band [in % - from 0
        to 100]
    """
    percent_shading = 0.
    if shading_angle >= horizon_band_angle:
        percent_shading = 100.
    elif (shading_angle >= 0) | (shading_angle < horizon_band_angle):
        percent_shading = 100. * shading_angle / horizon_band_angle
    else:
        # No shading
        pass
    return percent_shading


def find_edge_point(b1_pvrow, b2_pvrow):
    """
    Return edge point formed by pv row line and ground line. This assumes a
    flat ground surface, located at the hard-coded elevation ``Y_GROUND`` value.

    :param b1_pvrow: :class:`shapely.Point` object, first boundary point of the
        pv row line.
    :param b2_pvrow: :class:`shapely.Point` object, second boundary point of the
        pv row line.
    """

    u_vector = [b1_pvrow.x - b2_pvrow.x, b1_pvrow.y - b2_pvrow.y]
    n_vector = [u_vector[1], -u_vector[0]]
    intercept = - (n_vector[0] * b1_pvrow.x + n_vector[1] * b1_pvrow.y)

    x_edge_point = - (intercept
                      + n_vector[1] * Y_GROUND) / np.float64(n_vector[0])

    # TODO: need to find a better way to deal with this case
    if np.abs(x_edge_point) > THRESHOLD_EDGE_POINT:
        LOGGER.warning("find_edge_point: it looks like the tilt should be "
                       "approximated with a flat case instead")

    return Point(x_edge_point, Y_GROUND)
