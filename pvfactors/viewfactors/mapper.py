"""Classes that map view types to vf calculation methods"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from shapely.geometry import LineString
from pvfactors.config import REVERSE_VIEW_DICT
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

np.set_printoptions(suppress=True)


class VFMapperOrderedPVArray(object):
    """This class maps the "view types" of the
    :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` ``view_matrix``
    property to some view factor calculation functions.
    For instance, the method to calculate the view factor between 2 PV row
    surfaces will be different than between a PV row surface and a ground one.
    """
    reverse_view = REVERSE_VIEW_DICT

    def __init__(self):
        """Initialize the mapping between "view types" and calculation methods.
        """
        self.function_mapping = {
            "back_gnd": self.vf_hottel_pvrow_ground,
            "gnd_back": self.vf_hottel_pvrow_ground,
            "back_gnd_obst": self.vf_hottel_pvrow_ground,
            "gnd_back_obst": self.vf_hottel_pvrow_ground,
            "front_gnd_obst": self.vf_hottel_pvrow_ground,
            "gnd_front_obst": self.vf_hottel_pvrow_ground,
            "pvrows": self.vf_trk_to_trk}

    def vf_hottel_pvrow_ground(self, line_1, line_2, obstructing_line):
        """Use Hottel method to calculate view factor between PV row surface
        and the ground. Can account for obstructing object.

        Parameters
        ----------
        line_1: ``shapely.LineString``
            Line for surface 1: pv row or ground
        line_2: ``shapely.LineString``
            Line for surface 2: pv row or ground
        obstructing_line: :py:class:`~pvfactors.geometry.pvrow.PVRow` object
            Line for obstructing the view between line_1 and line_2

        Returns
        -------
        float
            calculated view factor vf_12

        """
        # FIXME: specific to pvlinestring
        obs_1 = obstructing_line
        # Implement the Hottel method
        b1_line1 = line_1.boundary[0]
        b2_line1 = line_1.boundary[1]
        a_1 = line_1.length

        b1_line2 = line_2.boundary[0]
        b2_line2 = line_2.boundary[1]

        # TODO: Speed up this calculation
        # calculation of intersects takes a lot of time in length_string
        length_x_1, state_1 = self.length_string(b1_line1, b2_line2, obs_1)
        length_x_2, state_2 = self.length_string(b2_line1, b1_line2, obs_1)
        length_1, state_3 = self.length_string(b1_line1, b1_line2, obs_1)
        length_2, state_4 = self.length_string(b2_line1, b2_line2, obs_1)

        logging.debug("... obstruction of uncrossed string 1: %s", state_1)
        logging.debug("... obstruction of uncrossed string 2: %s", state_2)
        logging.debug("... obstruction of crossed string 1: %s", state_3)
        logging.debug("... obstruction of crossed string 2: %s", state_4)

        if state_1 * state_2 * state_3 * state_4:
            # Complete obstruction: can happen when direct shading and when
            # considering top segments of the pv string line
            logging.info("Hottel front: View is completely obstructed")
            vf_12 = 0.
        else:
            list_sums = [length_x_1 + length_x_2, length_1 + length_2]
            sum_crossed = max(list_sums)
            sum_uncrossed = min(list_sums)
            vf_12 = 1. / (2. * a_1) * (sum_crossed - sum_uncrossed)
            # Must be positive

        return vf_12

    def vf_trk_to_trk(self, line_1, line_2, *args):
        """Use parallel plane formula to calculate view factor between PV rows.
        This assumes that pv row  # 2 is just a translation along the x-axis of
        pv row  # 1; meaning that only their x-values are different and they
        have the same elevation and rotation angle.

        Parameters
        ----------
        line_1: ``shapely.LineString``
            Line for surface 1: pv row or ground
        line_2: ``shapely.LineString``
            Line for surface 2: pv row or ground
        *args: tuple
            Will not be used: here for convenience and consistency of arguments
            between all view factor calculation methods

        Returns
        -------
        float
            calculated view factor vf

        """

        b1_line1 = line_1.boundary[0]
        b2_line1 = line_1.boundary[1]
        b1_line2 = line_2.boundary[0]
        b2_line2 = line_2.boundary[1]

        length_3 = b1_line1.distance(b1_line2)
        length_4 = b2_line1.distance(b2_line2)
        length_1 = b2_line1.distance(b1_line2)
        length_2 = b1_line1.distance(b2_line2)
        w1 = b1_line1.distance(b2_line1)

        vf = self.vf_parallel_planes(length_1, length_2, length_3, length_4,
                                     w1)

        return vf

    @staticmethod
    def length_string(pt1, pt2, obstruction):
        """Calculate the length of a string between two boundary points of two
        lines as defined in the Hottel method. This accounts for potentially
        obstructing objects.
        It assumes that the obstructing line is a
        :py:class:`~pvfactors.geometry.pvrow.PVRow` object
        and that it has a ``lowest_point`` attribute in order to calculate the
        Hottel string length when there is obstruction.

        Parameters
        ----------
        pt1: ``shapely.Point``
            a point from line 1
        pt2: ``shapely.Point``
            a point from line 2
        obstruction: :py:class:`~pvfactors.geometry.pvrow.PVRow` object
            PV row obstructing the view between line_1 and line_2

        Returns
        -------
        length: float
            the length of the Hottel string
        is_obstructing: bool
            boolean flag specifying if there's obstruction or not between the
            two points

        """
        # FIXME: specific to pvlinestrings, and assumes b1_obstruction should
        # be used if obstruction
        is_obstructing = False
        if obstruction is not None:
            string = LineString([pt1, pt2])
            is_obstructing = string.intersects(obstruction.original_linestring)
        if is_obstructing:
            b1_obstruction = obstruction.lowest_point
            length = (pt1.distance(b1_obstruction) +
                      b1_obstruction.distance(pt2))
        else:
            length = pt1.distance(pt2)

        return length, is_obstructing

    @staticmethod
    def vf_parallel_planes(length_1, length_2, length_3, length_4, w1):
        """See: http://www.thermalradiation.net/sectionc/C-2a.htm

        Parameters
        ----------
        length_1: float

        length_2: float

        length_3: float

        length_4: float

        w1: float

        Returns
        -------
        float
            view factor between the two parallel lines

        """
        list_sums = [length_1 + length_2, length_3 + length_4]
        sum_crossed = max(list_sums)
        sum_uncrossed = min(list_sums)
        vf_1_to_2 = (sum_crossed - sum_uncrossed) / (2. * w1)

        return vf_1_to_2
