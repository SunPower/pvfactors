"""Classes that map view types to vf calculation methods"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from shapely.geometry import LineString
from pvfactors.config import REVERSE_VIEW_DICT, THRESHOLD_VF_12
import numpy as np
import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


np.set_printoptions(suppress=True)


class VFMapperOrderedPVArray(object):
    """This class calculates the view factors based on the surface registry of
    :py:class:`~pvfactors.pvarray.Array` class. Its only attribute is a mapping
    between "view types" and the "low-level" view factor calculation functions

    """
    reverse_view = REVERSE_VIEW_DICT

    def __init__(self):
        self.function_mapping = {
            "back_gnd": self.vf_hottel_pvrowline_back,
            "gnd_back": self.vf_hottel_pvrowline_back,
            "back_gnd_obst": self.vf_hottel_pvrowline_back,
            "gnd_back_obst": self.vf_hottel_pvrowline_back,
            "front_gnd_obst": self.vf_hottel_pvrowline_front,
            "gnd_front_obst": self.vf_hottel_pvrowline_front,
            "pvrows": self.vf_trk_to_trk}

    def vf_hottel_pvrowline_back(self, line_1, line_2, obstructing_line):
        """Use Hottel method to calculate view factor between PV row back-surface
        and the ground. Can account for obstructing object.

        Parameters
        ----------
        line_1 : ``shapely.LineString``
            Line for surface 1: pv row or ground
        line_2 : ``shapely.LineString``
            Line for surface 2: pv row or ground
        obstructing_line : ``shapely.LineString``
            Line for obstructing the view between line_1 and line_2

        Returns
        -------
        float
            calculated view factor vf_12

        """
        # FIXME: specific to pvlinestring
        obs_1 = obstructing_line
        # Implement the Hottel method
        b1_line1, b2_line1 = line_1.boundary
        a_1 = line_1.length

        b1_line2, b2_line2 = line_2.boundary

        # TODO: Speed up this calculation
        # calculation of intersects takes a lot of time in length_string
        uncrossed_length1, state_1 = self.length_string(b1_line1, b1_line2,
                                                        obs_1)
        uncrossed_length2, state_2 = self.length_string(b2_line1, b2_line2,
                                                        obs_1)
        crossed_length1, state_3 = self.length_string(b1_line1, b2_line2,
                                                      obs_1)
        crossed_length2, state_4 = self.length_string(b2_line1, b1_line2,
                                                      obs_1)
        logging.debug("... obstruction of uncrossed string 1: %s", state_1)
        logging.debug("... obstruction of uncrossed string 2: %s", state_2)
        logging.debug("... obstruction of crossed string 1: %s", state_3)
        logging.debug("... obstruction of crossed string 2: %s", state_4)

        if state_1 * state_2 * state_3 * state_4:
            # Complete obstruction: can happen when direct shading and when
            # considering top segments of the pv string line
            logging.info("Hottel back: View is completely obstructed")
            vf_12 = 0.
        else:
            vf_12 = 1. / (2. * a_1) * (crossed_length1 + crossed_length2 -
                                       uncrossed_length1 - uncrossed_length2)

        if vf_12 < 0:
            LOGGER.debug("Hottel pvrow front: unexpected value for "
                         "vf_12 = %.4f" % vf_12)

        return vf_12

    def vf_hottel_pvrowline_front(self, line_1, line_2, obstructing_line):
        """Use Hottel method to calculate view factor between PV row front-surface
        and the ground. Can account for obstructing object.

        Parameters
        ----------
        line_1 : ``shapely.LineString``
            Line for surface 1: pv row or ground
        line_2 : ``shapely.LineString``
            Line for surface 2: pv row or ground
        obstructing_line : ``shapely.LineString``
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
        uncrossed_length1, state_1 = self.length_string(b1_line1, b2_line2,
                                                        obs_1)
        uncrossed_length2, state_2 = self.length_string(b2_line1, b1_line2,
                                                        obs_1)
        crossed_length1, state_3 = self.length_string(b1_line1, b1_line2,
                                                      obs_1)
        crossed_length2, state_4 = self.length_string(b2_line1, b2_line2,
                                                      obs_1)

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
            vf_12 = 1. / (2. * a_1) * (crossed_length1 + crossed_length2 -
                                       uncrossed_length1 - uncrossed_length2)
            # FIXME: temporary fix, I'm getting negative values smaller that
            # 5e-5 when ran in a notebook
            if (vf_12 < 0) and (vf_12 > - THRESHOLD_VF_12):
                vf_12 = 0.

        if vf_12 < 0:
            LOGGER.debug("Hottel pvrow front: unexpected value for "
                         "vf_12 = %.4f" % vf_12)

        return vf_12

    def vf_trk_to_trk(self, line_1, line_2, *args):
        """Use parallel plane formula to calculate view factor between PV rows.
        This assumes that pv row #2 is just a translation along the x-axis of
        pv row #1; meaning that only their x-values are different and they have
        the same elevation or rotation angle.

        Parameters
        ----------
        line_1 : ``shapely.LineString``
            Line for surface 1: pv row or ground
        line_2 : ``shapely.LineString``
            Line for surface 2: pv row or ground
        *args : tuple

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
        :py:class:`~pvfactors.pvrow.PVRowLine`
        and that it has a ``lowest_point`` attribute in order to calculate the
        Hottel string length when there is obstruction.

        Parameters
        ----------
        pt1 : ``shapely.Point``
            a point from line 1
        pt2 : ``shapely.Point``
            a point from line 2
        obstruction : ``shapely.LineString``
            Line obstructing the view between line_1 and line_2

        Returns
        -------
        length : float
            the length of the Hottel string
        is_obstructing : bool
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
        length_1 : float

        length_2 : float

        length_3 : float

        length_4 : float

        w1 : float


        Returns
        -------
        float
            view factor between the two parallel lines

        """
        vf_1_to_2 = (length_1 + length_2 - length_3 - length_4) / (2. * w1)

        return vf_1_to_2
