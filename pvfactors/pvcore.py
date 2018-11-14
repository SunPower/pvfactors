# -*- coding: utf-8 -*-

from shapely.geometry import Point
from pvfactors import (PVFactorsError, PVFactorsEdgePointDoesNotExist,
                       PVFactorsArrayUpdateException)
import numpy as np
import math
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

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
            raise PVFactorsError("'line_type' cannot be: %s, \n possible "
                                 "values are: %s" % (
                                     str(line_type),
                                     str(self._list_line_types)))


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
        raise PVFactorsError(
            'calculate_circumsolar_shading: model does not exist: '
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
    flat ground surface, located at the hard-coded elevation ``Y_GROUND`` value

    :param b1_pvrow: :class:`shapely.Point` object, first boundary point of the
        pv row line.
    :param b2_pvrow: :class:`shapely.Point` object, second boundary point of
        the pv row line.
    """

    u_vector = [b1_pvrow.x - b2_pvrow.x, b1_pvrow.y - b2_pvrow.y]
    n_vector = [u_vector[1], -u_vector[0]]
    intercept = - (n_vector[0] * b1_pvrow.x + n_vector[1] * b1_pvrow.y)

    if n_vector[0]:
        x_edge_point = - (intercept
                          + n_vector[1] * Y_GROUND) / np.float64(n_vector[0])
    else:
        x_edge_point = np.inf

    # TODO: need to find a better way to deal with this case
    if np.abs(x_edge_point) > THRESHOLD_EDGE_POINT:
        LOGGER.debug("find_edge_point: it looks like the tilt should be "
                     "approximated with a flat case")

    return Point(x_edge_point, Y_GROUND)
