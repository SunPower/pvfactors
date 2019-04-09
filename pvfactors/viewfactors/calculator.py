"""Module with classes and functions to calculate views and view factors"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from shapely.geometry import LineString, Polygon
from pvfactors.config import REVERSE_VIEW_DICT, THRESHOLD_VF_12
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

np.set_printoptions(suppress=True)


class VFCalculator(object):

    def __init__(self, mapper=None):
        if mapper is None:
            mapper = VFMapperOrderedPVArray()
        self.mapper = mapper

    def get_vf_matrix(self, geom_dict, view_matrix, obstr_matrix, list_pvrow):
        """Calculate the view factors based on the inputs of the
        :py:class:`~pvfactors.pvarray.Array` class, and using a mapping of
        "view types" with view factor calculation methods.
        The method uses a faster implementation by only calculating half of the
        view factors, and uses the symmetry property of the transformed view
        factor matrix to calculate the other half.
        The symmetry property comes from the reciprocity property of view
        factors: :math:`A_1 * VF_{1-2} = A_2 * VF_{2-1}`,
        where A_i is the area of surface i

        Parameters
        ----------
        geom_dict : ``OrderedDictionary``
            Ordered dictionary of all indexed PV surfaces
        view_matrix : ``numpy.ndarray``
            This matrix specifies what surfaces each surface views, and
            also what type of view it is
        obstr_matrix : ``numpy.ndarray``
            Complementing ``view_matrix`` by providing
            additional arguments for the calculations; i.e. what pvrows are
            obstructing the view between surface i and surface j
        list_pvrows : list
            List of pvrows that can obstruct views

        Returns
        -------
        ``numpy.ndarray``
            matrix of view factors (values from 0 to 1 in theory)

        """
        n_all_surfaces = view_matrix.shape[0]
        view_factors = np.zeros((n_all_surfaces, n_all_surfaces), dtype=float)

        # --- First deal with finite surfaces from the registry, and treat only
        # half of the views because symmetry will be used next
        n_finite_surfaces = n_all_surfaces - 1  # no sky
        view_matrix_upper_finite_surfaces = np.triu(
            view_matrix[:n_finite_surfaces, :n_finite_surfaces])
        indices_views_finite = np.where(view_matrix_upper_finite_surfaces)

        n_views = len(indices_views_finite[0])
        geometries = list(geom_dict.values())
        for i in range(n_views):
            idx = (indices_views_finite[0][i], indices_views_finite[1][i])
            view = REVERSE_VIEW_DICT[view_matrix[idx]]
            line_i = geometries[idx[0]]
            line_j = geometries[idx[1]]
            obstr_index = obstr_matrix[idx]
            if obstr_index is not None:
                obstructing_pvrow = list_pvrow[obstr_matrix[idx]]
            else:
                obstructing_pvrow = None
            # The following line takes the most time to execute (looped)
            view_factors[idx] = self.mapper.function_mapping[view](
                line_i, line_j, obstructing_pvrow)

        # Use the reciprocity property of view factors to speed up the
        # vfactor calculation: A_1 * F_1-2 = A_2 * F_2-1 ==> symmetric matrx
        areas = np.array([surf.length for surf in geometries])
        matrix_areas = np.diag(areas)
        matrix_areas_inv = np.diag(1. / areas)

        upper_matrix_reciprocity = np.dot(matrix_areas,
                                          view_factors[:n_finite_surfaces,
                                                       :n_finite_surfaces])

        total_matrix_reciprocity = (upper_matrix_reciprocity +
                                    upper_matrix_reciprocity.T)
        finite_vf_matrix = np.dot(matrix_areas_inv, total_matrix_reciprocity)
        view_factors[:n_finite_surfaces, :n_finite_surfaces] = finite_vf_matrix

        # --- Then do the calculations for the sky, which is the remaining
        # portion of the hemisphere
        view_factors[:-1, -1] = 1. - np.sum(view_factors[:-1, :-1], axis=1)
        return view_factors

    def get_viewfactors(self, surface_idx):
        raise NotImplementedError

    def get_viewfactor(self, surf_1_idx, surf_2_idx):
        raise NotImplementedError


class VFMapperOrderedPVArray(object):
    """This class calculates the view factors based on the surface registry of
    :py:class:`~pvfactors.pvarray.Array` class. Its only attribute is a mapping
    between "view types" and the "low-level" view factor calculation functions

    """

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
