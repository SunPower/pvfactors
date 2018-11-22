# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pvfactors import PVFactorsArrayUpdateException
from pvfactors.pvcore import (LinePVArray,
                              find_edge_point, Y_GROUND,
                              MAX_X_GROUND, MIN_X_GROUND,
                              calculate_circumsolar_shading,
                              calculate_horizon_band_shading)
from pvfactors.pvrow import PVRowLine
from pvfactors.view_factors import ViewFactorCalculator, VIEW_DICT
from shapely.geometry import LineString, Point
import numpy as np
import pandas as pd
from pvfactors.pvgeometry import PVGeometry
from pandas import DataFrame as Registry
from pandas import notnull, Series
from scipy import linalg
from pvlib.tools import cosd, sind
from pvlib.irradiance import aoi as aoi_function
import logging
import copy

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

X_ORIGIN_PVROWS = 0.
DEFAULT_EDGE_PT_X = X_ORIGIN_PVROWS
DELTA_MAX_MIN_GROUND_WHEN_TOO_SMALL_BIG = 1
DEFAULT_CIRCUMSOLAR_ANGLE = 30.
DEFAULT_HORIZON_BAND_ANGLE = 6.5


class ArrayBase(object):
    """
    ``ArrayBase`` exists for future developments of the model. It is the
     base class for PV arrays that will contain all the boiler plate code
     shared by sub classes like ``Array``, or for instance more complex
     PV array classes with varying GCR values, or non-flat ground, etc.

    :param int n_pvrows: number of PV rows in PV array
    :param float pvrow_height: height of PV rows
    """
    registry_cols = [
        # LinePVArray keys
        'geometry', 'style', 'line_type', 'shaded', 'pvrow_index',
        # Geometry cols
        'pvrow_segment_index', 'index_pvrow_neighbor',
        'edge_point', 'surface_side', 'surface_centroid',
        'area', 'line_registry_index',
        # Irradiance terms
        'reflectivity', 'irradiance_term', 'direct_term',
        'isotropic_term', 'circumsolar_term',
        'horizon_term', 'circumsolar_shading_pct',
        'horizon_band_shading_pct', 'q0', 'qinc']
    registry_numeric_cols = [
        'pvrow_index', 'pvrow_segment_index', 'index_pvrow_neighbor',
        'area', 'line_registry_index', 'reflectivity', 'irradiance_term',
        'direct_term', 'isotropic_term', 'circumsolar_term',
        'horizon_term', 'circumsolar_shading_pct',
        'horizon_band_shading_pct', 'q0', 'qinc']

    def __init__(self, n_pvrows, pvrow_height):
        self.n_pvrows = n_pvrows
        self.pvrow_height = pvrow_height
        self.pvrows = []

        self.line_registry = self.initialize_registry()
        self.surface_registry = None
        self.view_matrix = None
        self.args_matrix = None
        self.view_factor_calculator = None
        self.vf_matrix = None
        self.inv_reflectivity_matrix = None

    @staticmethod
    def initialize_registry():
        """
        Create an empty line registry based on the property keys of PV Array
        lines.

        :return: empty :class:`pvcore.Registry` object
        """
        # Create the line and surface registries
        # TODO: line_pvarray_keys should not be dependent on specific classes
        # like ``LinePVArray()``
        registry = Registry(columns=ArrayBase.registry_cols)
        registry[ArrayBase.registry_numeric_cols] = (
            registry[ArrayBase.registry_numeric_cols].astype(float))
        return registry

    def create_pvrows_array(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def tilt(self):
        pvrow = self.pvrows[0]
        if hasattr(pvrow, 'tilt'):
            return pvrow.tilt
        else:
            raise AttributeError("The property 'tilt' is not defined for Array"
                                 " object")

    @property
    def pvrow_distance(self):
        pvrow = self.pvrows[0]
        if hasattr(pvrow, 'width') & (self.gcr is not None):
            return float(pvrow.width) / self.gcr
        else:
            raise AttributeError("The property 'pvrow_distance' is not "
                                 "defined for Array object")


class Array(ArrayBase):
    """
    | Create the array object. This will call the :meth:`update_view_factors`
    | method which creates the shapely geometry and calculate the view
    | factors based on the inputs.
    | Azimuth angles are counted positive going East from North. E.g. 0 deg is
    | North and 90 degrees is East.
    | #FIXME The array azimuth uses a different convention than pvlib: for the
    | torque-tube axis to be oriented South-North, the array azimuth angle
    | needs to be 90 deg. And 0 deg would be East-West orientation.
    | This assumes an equal spacing (or GCR) between all pv rows.
    | This assumes that all pv rows have identical rotation angles.
    | This assumes that all pv rows are at the same elevation (only x-values
    | change from a pv row to another).

    :param int n_pvrows: number of PV rows in parallel
    :param float pvrow_height: height of PV rows, measured from ground to
        center [meters]
    :param float pvrow_width: width of PV rows, in the considered 2D
        dimension [meters]
    :param float surface_tilt: Surface tilt angles in decimal degrees.
        surface_tilt must be >=0 and <=180.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)
    :param float surface_azimuth: The azimuth of the rotated panel,
        determined by projecting the vector normal to the panel's surface
        to the earth's surface [degrees].
    :param float solar_zenith: zenith angle of the sun [degrees]
    :param float solar_azimuth: azimuth angle of the sun [degrees]
    :param float rho_ground: ground albedo
    :param float rho_back_pvrow: reflectivity of PV row's back surface
    :param float rho_front_pvrow: reflectivity of PV row's front surface
    :param float gcr: ground coverage ratio of the PV array
    :param kwargs: possible options are: ``pvrow_class`` if the user wants
        to specify its own PV Row class; ``cut`` if the user wants to
        discretize some pv rows, e.g. [(0, 5, 'front'), (4, 2, 'back')]
        will discretize the front surface of the first PV row into 5 segments,
        and the back surface of the 5th pv row into 2 segments;
        ``calculate_front_circ_horizon_shading`` is a boolean that indicates
        whether to calculate front circumsolar & horizon band shading or not;
        ``circumsolar_angle`` would be the full (not half) angle of the
        circumsolar disk; ``horizon_band_angle`` would be the horizon band
        elevation angle
    """

    _pvrow_class = PVRowLine
    _view_factor_calculator = ViewFactorCalculator

    def __init__(self, n_pvrows=3, pvrow_height=1.5, pvrow_width=1.,
                 surface_tilt=20., surface_azimuth=180., solar_zenith=0.,
                 solar_azimuth=180., rho_ground=0.2, rho_back_pvrow=0.05,
                 rho_front_pvrow=0.03, gcr=0.3, **kwargs):

        # Set up all the initial class attributes
        super(Array, self).__init__(n_pvrows, pvrow_height)
        self.pvrow_class = kwargs.get('pvrow_class', self._pvrow_class)
        self.view_factor_calculator = self._view_factor_calculator()
        # Fixed array parameters
        self.gcr = gcr
        self.pvrow_width = pvrow_width
        self.rho_ground = rho_ground
        self.rho_back_pvrow = rho_back_pvrow
        self.rho_front_pvrow = rho_front_pvrow
        self.cut = kwargs.get('cut', [])
        self.circumsolar_angle = kwargs.get('circumsolar_angle',
                                            DEFAULT_CIRCUMSOLAR_ANGLE)  # deg
        self.horizon_band_angle = kwargs.get('horizon_band_angle',
                                             DEFAULT_HORIZON_BAND_ANGLE)  # deg
        self.calculate_front_circ_horizon_shading = kwargs.get(
            'calculate_front_circ_horizon_shading', False
        )
        self.circumsolar_model = kwargs.get('circumsolar_model',
                                            'uniform_disk')
        # Variable array parameters
        self.solar_zenith = None
        self.solar_azimuth = None
        self.surface_azimuth = None
        self.surface_tilt = None
        self.illum_ground_indices = None
        self.has_direct_shading = None
        self.solar_2d_vector = None
        self.irradiance_terms = None

        # Update array from initial parameters
        self.update_view_factors(solar_zenith, solar_azimuth,
                                 surface_tilt, surface_azimuth)

    def update_view_factors(self, solar_zenith, solar_azimuth, surface_tilt,
                            surface_azimuth):
        """
        Create new line and surface registries based on new inputs, and re-cal-
        culate the view factor matrix of the updated system.

        :param float solar_zenith: zenith angle of the sun [in deg]
        :param float solar_azimuth: azimuth angle of the sun [in deg]
        :param float surface_tilt: Surface tilt angles in decimal degrees.
            surface_tilt must be >=0 and <=180.
            The tilt angle is defined as degrees from horizontal
            (e.g. surface facing up = 0, surface facing horizon = 90)
        :param float surface_azimuth: The azimuth of the rotated panel,
            determined by projecting the vector normal to the panel's surface
            to the earth's surface [degrees].
        :return: None
        """
        self.line_registry = self.initialize_registry()
        # Check on which side the light is incident
        sun_on_front_surface = aoi_function(surface_tilt, surface_azimuth,
                                            solar_zenith, solar_azimuth) <= 90

        # Update array parameters
        self.surface_azimuth = surface_azimuth
        self.surface_tilt = surface_tilt
        self.solar_zenith = solar_zenith
        self.solar_azimuth = solar_azimuth

        # ------- Line creation: returning the line registry
        LOGGER.debug("...building line registry")
        # Create the PV rows / structures
        self.pvrows = self.create_pvrows_array(self.n_pvrows,
                                               self.pvrow_height)

        # Create the ground and the shadows on it
        self.create_pvrow_shadows(surface_azimuth, solar_zenith, solar_azimuth)
        self.create_ill_ground()
        edge_points = self.find_edge_points()
        self.create_remaining_illum_ground(edge_points)
        # Assuming the edge points are ordered
        # --- Add edge points to geometries
        self.line_registry.pvgeometry.split_ground_geometry_from_edge_points(
            edge_points)

        # ------- Surface creation: returning the surface registry, a line may
        # represent 2 surfaces (front and back)
        LOGGER.debug("...building surface registry")
        self.create_surface_registry()

        # -------- Interrow shading
        if self.has_direct_shading:
            LOGGER.debug("...calculating interrow shading")
            self.calculate_interrow_direct_shading(sun_on_front_surface)

        # -------- Update the surface areas (lengths) after shading calculation
        self.surface_registry.loc[
            :, 'area'] = self.surface_registry.pvgeometry.length

        # ------- View factors: define the surface views and calculate view
        # factors
        LOGGER.debug("...building view matrix and finding obstructions")
        (self.view_matrix,
         self.args_matrix) = self.create_view_matrix()
        LOGGER.debug("...calculating view factors")
        self.vf_matrix = self.view_factor_calculator.calculate_view_factors(
            self.surface_registry, self.view_matrix, self.args_matrix
        )

    def update_irradiance_terms_perez(self, solar_zenith, solar_azimuth,
                                      surface_tilt, surface_azimuth, dni,
                                      luminance_isotropic,
                                      luminance_circumsolar,
                                      poa_horizon, poa_circumsolar):
        """
        Calculate the irradiance source terms of all surfaces using values
        pre-calculated from the Perez transposition model.

        :param float solar_zenith: zenith angle of the sun [degrees]
        :param float solar_azimuth: azimuth angle of the sun [degrees]
        :param float surface_tilt: Surface tilt angles in decimal degrees.
            surface_tilt must be >=0 and <=180.
            The tilt angle is defined as degrees from horizontal
            (e.g. surface facing up = 0, surface facing horizon = 90)
        :param float surface_azimuth: azimuth angle of the PV surfaces. All PV
            surfaces must have the same azimuth angle [degrees]
        :param float dni: direct normal irradiance [W/m2]
        :param float luminance_isotropic: luminance of the isotropic part of
            the sky dome [W/m2/sr]
        :param float luminance_circumsolar: luminance of the circumsolar part
            of the sky dome [W/m2/sr]
        :param float poa_horizon: plane-of-array horizon component of the
            irradiance as calculated by Perez for the front surface of a PV row
            [W/m2]
        :param float poa_circumsolar: plane-of-array circumsolar component of
            the irradiance as calculated by Perez for the front surface of a PV
            row [W/m2]
        :return: None
        """
        self.surface_registry['circumsolar_term'] = 0.
        self.surface_registry['horizon_term'] = 0.
        self.surface_registry['direct_term'] = 0.
        self.surface_registry['surface_centroid'] = (
            self.surface_registry.pvgeometry.centroid
        )
        self.surface_registry['circumsolar_shading_pct'] = 0.
        self.surface_registry['horizon_band_shading_pct'] = 0.
        self.irradiance_terms = np.zeros(self.surface_registry.shape[0] + 1)

        # --- Calculate terms
        dni_ground = dni * cosd(solar_zenith)
        circumsolar_ground = luminance_circumsolar
        # FIXME: only works for pvrows as lines
        aoi_frontsurface = aoi_function(surface_tilt, surface_azimuth,
                                        solar_zenith, solar_azimuth)

        # --- Assign terms to surfaces
        # Illuminated ground
        self.surface_registry.loc[
            ~ self.surface_registry.shaded
            & (self.surface_registry.line_type == 'ground'),
            'direct_term'] = dni_ground
        self.surface_registry.loc[
            ~ self.surface_registry.shaded
            & (self.surface_registry.line_type == 'ground'),
            'circumsolar_term'] = circumsolar_ground

        # -- PVRow surfaces

        # Set the horizon diffuse light
        # TODO: poa_horizon should not be negative... but it can be in the
        # Perez model from pvlib
        poa_horizon = np.abs(poa_horizon)

        # Add poa horizon contribution to all pvrow surfaces
        self.surface_registry.loc[
            (self.surface_registry.line_type == 'pvrow'),
            'horizon_term'] += poa_horizon

        # Apply horizon band shading for the pvrow back surfaces
        self.apply_horizon_band_shading('back')

        if aoi_frontsurface <= 90.:
            # Direct light is incident on front side of pvrows
            dni_pvrow = dni * cosd(aoi_frontsurface)
            circumsolar_pvrow = poa_circumsolar
            self.surface_registry.loc[
                ~ self.surface_registry.shaded
                & (self.surface_registry.line_type == 'pvrow')
                & (self.surface_registry.surface_side == 'front'),
                'direct_term'] = dni_pvrow
            self.surface_registry.loc[
                ~ self.surface_registry.shaded
                & (self.surface_registry.line_type == 'pvrow')
                & (self.surface_registry.surface_side == 'front'),
                'circumsolar_term'] = circumsolar_pvrow
            if self.calculate_front_circ_horizon_shading:
                self.apply_front_circumsolar_horizon_shading()
        else:
            # Direct light is incident on back side of pvrows
            aoi_backsurface = 180. - aoi_frontsurface
            dni_pvrow = dni * cosd(aoi_backsurface)
            vf_circumsolar_backsurface = (cosd(aoi_backsurface)
                                          / cosd(solar_zenith))
            circumsolar_pvrow = (luminance_circumsolar
                                 * vf_circumsolar_backsurface)
            self.surface_registry.loc[
                ~ self.surface_registry.shaded
                & (self.surface_registry.line_type == 'pvrow')
                & (self.surface_registry.surface_side == 'back'),
                'direct_term'] = dni_pvrow
            self.surface_registry.loc[
                ~ self.surface_registry.shaded
                & (self.surface_registry.line_type == 'pvrow')
                & (self.surface_registry.surface_side == 'back'),
                'circumsolar_term'] = circumsolar_pvrow

        # Sum up the terms
        self.surface_registry['irradiance_term'] = (
            self.surface_registry['direct_term']
            + self.surface_registry['circumsolar_term']
            + self.surface_registry['horizon_term']
        )

        # Set up irradiance terms vector
        self.irradiance_terms[:-1] = (self.surface_registry.irradiance_term
                                      .values)
        # Isotropic sky dome luminance value
        self.irradiance_terms[-1] = luminance_isotropic

    def apply_horizon_band_shading(self, pvrow_side):
        """
        Calculate the amount of diffuse shading happening on the horizon
        band components for the 'back' pvrow surfaces

        :param str pvrow_side:
        :return: None
        """

        slice_registry = self.surface_registry.loc[
            (self.surface_registry.line_type == 'pvrow')
            & (self.surface_registry.surface_side == pvrow_side)
            & notnull(self.surface_registry.index_pvrow_neighbor), :]

        # Calculate the shading angle in 2D plane for each back surface
        for index, row in slice_registry.iterrows():
            row_point = row.surface_centroid
            neighbor_point = self.pvrows[
                int(row.index_pvrow_neighbor)].highest_point
            shading_angle = np.abs(np.arctan(
                (neighbor_point.y - row_point.y)
                / (neighbor_point.x - row_point.x))
            ) * 180. / np.pi
            # shading_angle = 0.
            percent_horizon_shading = calculate_horizon_band_shading(
                shading_angle, self.horizon_band_angle)
            self.surface_registry.loc[index, 'horizon_term'] *= (
                1. - percent_horizon_shading / 100.)
            self.surface_registry.loc[index, 'horizon_band_shading_pct'] = (
                percent_horizon_shading)

    def apply_front_circumsolar_horizon_shading(self):
        """
        Calculate what amount of diffuse shading is happening on the
        circumsolar and horizon band components and apply it to all 'front'
        surfaces of pvrows. It just updates the corresponding irradiance
        terms that will be used in the mathematical formulation.

        :return: None
        """

        slice_registry = self.surface_registry.loc[
            (self.surface_registry.surface_side == 'front')
            & notnull(self.surface_registry.index_pvrow_neighbor), :]

        # Calculate the solar and circumsolar elevation angles in 2D plane
        solar_2d_elevation = np.abs(
            np.arctan(self.solar_2d_vector[1] / self.solar_2d_vector[0])
        ) * 180. / np.pi
        lower_angle_circumsolar = (solar_2d_elevation
                                   - self.circumsolar_angle / 2.)

        # Calculate the shading angle in 2D plane as well for each surface
        for index, row in slice_registry.iterrows():
            row_point = row.surface_centroid
            neighbor_point = self.pvrows[
                int(row.index_pvrow_neighbor)].highest_point
            shading_angle = np.abs(np.arctan(
                (neighbor_point.y - row_point.y)
                / (neighbor_point.x - row_point.x))
            ) * 180. / np.pi
            percentage_circ_angle_covered = ((shading_angle -
                                              lower_angle_circumsolar)
                                             / self.circumsolar_angle) * 100.
            percent_circ_shading = calculate_circumsolar_shading(
                percentage_circ_angle_covered, model=self.circumsolar_model)
            percent_horizon_shading = calculate_horizon_band_shading(
                shading_angle, self.horizon_band_angle)
            self.surface_registry.loc[index, 'circumsolar_term'] *= (
                1. - percent_circ_shading / 100.)
            self.surface_registry.loc[index, 'horizon_term'] *= (
                1. - percent_horizon_shading / 100.)
            self.surface_registry.loc[index, 'circumsolar_shading_pct'] = (
                percent_circ_shading)
            self.surface_registry.loc[index, 'horizon_band_shading_pct'] = (
                percent_horizon_shading)

    def update_reflectivity_matrix(self):
        """
        Update new surface registry with reflectivity values for all surfaces,
        and calculate inverse of the reflectivity matrix.

        :return: None
        """
        self.surface_registry['reflectivity'] = np.nan

        # Assign reflectivity to all surfaces
        self.surface_registry.loc[
            self.surface_registry.line_type == 'ground',
            'reflectivity'] = self.rho_ground
        self.surface_registry.loc[
            (self.surface_registry.line_type == 'pvrow')
            & (self.surface_registry.surface_side == 'front'),
            'reflectivity'] = self.rho_front_pvrow
        self.surface_registry.loc[
            (self.surface_registry.line_type == 'pvrow')
            & (self.surface_registry.surface_side == 'back'),
            'reflectivity'] = self.rho_back_pvrow
        # Create inv reflectivity matrix
        self.inv_reflectivity_matrix = np.diag(
            list(1. / self.surface_registry.reflectivity.values) + [1])

    def calculate_radiosities_perez(
            self, solar_zenith, solar_azimuth, surface_tilt, surface_azimuth,
            dni, luminance_isotropic, luminance_circumsolar, poa_horizon,
            poa_circumsolar):
        """
        Solve linear system of equations to calculate radiosity terms based on
        the specified inputs and using Perez diffuse light transposition model
        pre-calculated values

        :param float solar_zenith: zenith angle of the sun [degrees]
        :param float solar_azimuth: azimuth angle of the sun [degrees]
        :param float surface_tilt: Surface tilt angles in decimal degrees.
            surface_tilt must be >=0 and <=180.
            The tilt angle is defined as degrees from horizontal
            (e.g. surface facing up = 0, surface facing horizon = 90)
        :param float surface_azimuth: The azimuth of the rotated panel,
            determined by projecting the vector normal to the panel's surface
            to the earth's surface [degrees].
        :param float dni: direct normal irradiance [W/m2]
        :param float luminance_isotropic: luminance of the isotropic part of
            the sky dome [W/m2/sr]
        :param float luminance_circumsolar: luminance of the circumsolar part
            of the sky dome [W/m2/sr]
        :param float poa_horizon: plane-of-array horizon component of the
            irradiance as calculated by Perez for the front surface of a PV row
            [W/m2]
        :param float poa_circumsolar: plane-of-array circumsolar component of
            the irradiance as calculated by Perez for the front surface of a PV
            row [W/m2]
        :return: None; updating :attr:`surface_registry`
        """
        # Update the array configuration
        try:
            self.update_view_factors(solar_zenith, solar_azimuth,
                                     surface_tilt, surface_azimuth)
        except Exception as err:
            raise PVFactorsArrayUpdateException(
                "Could not calculate shapely array or view factors because of "
                "error: %s" % err)

        self.update_irradiance_terms_perez(solar_zenith, solar_azimuth,
                                           surface_tilt, surface_azimuth, dni,
                                           luminance_isotropic,
                                           luminance_circumsolar,
                                           poa_horizon, poa_circumsolar)
        self.update_reflectivity_matrix()

        # Do calculation
        a_mat = self.inv_reflectivity_matrix - self.vf_matrix
        q0 = linalg.solve(a_mat, self.irradiance_terms)
        qinc = np.dot(self.vf_matrix, q0) + self.irradiance_terms
        # Assign to surfaces
        self.surface_registry.loc[:, 'q0'] = q0[:-1]
        self.surface_registry.loc[:, 'qinc'] = qinc[:-1]
        self.calculate_sky_and_reflection_components()

    def calculate_sky_and_reflection_components(self):
        """
        Assuming that the calculation of view factors and radiosity terms is
        completed, calculate the irradiance components of the isotropic sky
        dome and of the reflections from surrounding surfaces
        (pv rows and ground) for all the surfaces in the PV array.
        Update the surface registry.

        :return: None
        """

        # FIXME: not very robust, make sure to have a test for it
        self.surface_registry['isotropic_term'] = (self.vf_matrix[:-1, -1]
                                                   * self.irradiance_terms[-1])
        self.surface_registry['reflection_term'] = (
            self.surface_registry['qinc']
            - self.surface_registry['irradiance_term']
            - self.surface_registry['isotropic_term']
        )

# ------- Line creation
    def create_pvrows_array(self, n_pvrows, pvrow_height):
        """
        Create list of PV rows in array, counting from left to right.
        In the 2D plane that will be considered, no matter the array azimuth
        angle will be, POSITIVE tilts will lead to pv surfaces tilted to the
        LEFT, and NEGATIVE tilts will lead to PV surfaces tilted to the RIGHT.
        So in the case of a single axis tracker, the direction of the torque
        tube will be the normal vector going out of the 2D plane.

        :param int n_pvrows: number of PV rows in the array
        :param float pvrow_height: height of the PV rows, measured from ground
            to the center of the row
        :return: list of :class:`pvrow.PVRowLine` objects, for now.
        """
        # Assume that all rows are at the same height
        y_center = pvrow_height
        x_center = X_ORIGIN_PVROWS
        index = 0
        pvrow = self.pvrow_class(self.line_registry, x_center, y_center, index,
                                 self.surface_tilt, self.pvrow_width)
        pvrows = [pvrow]
        if n_pvrows > 1:
            distance = pvrow.width / self.gcr
            for i in range(1, n_pvrows):
                x_center = i * distance
                pvrow = self.pvrow_class(
                    self.line_registry, x_center, y_center,
                    i, self.surface_tilt, self.pvrow_width)
                pvrows.append(pvrow)

        return pvrows

    def create_pvrow_shadows(self, surface_azimuth,
                             solar_zenith, solar_azimuth):
        """
        Create the PV row shadows cast on the ground. Since the PV array is in
        2D, the approach here is to project the solar vector into the 2D plane
        considered here. The next step is to calculate the shadow boundaries
        based on the PV row position and the solar angle using some geometry.
        The calculated shadow lines are added to the :attr:`line_registry`.
        Assumption: if there is direct shading between rows, this will mean
        that there is one continuous shadow on the ground formed by all the
        trackers' shadows.


        :param float surface_azimuth: The azimuth of the rotated panel,
            determined by projecting the vector normal to the panel's surface
            to the earth's surface [degrees].
        :param float solar_zenith: sun's zenith angle
        :param float solar_azimuth: sun's azimuth angle
        :return: None
        """
        # Projection of 3d solar vector onto the cross section of the systems:
        # which is the 2d plane we are considering: needed to calculate shadows
        # Remember that the 2D plane is such that the direction of the torque
        # tube vector goes out of (and normal to) the 2D plane, such that
        # positive tilt angles will have the PV surfaces tilted to the LEFT
        # and vice versa
        solar_2d_vector = [
            # a drawing really helps understand the following
            - sind(solar_zenith) * cosd(surface_azimuth - solar_azimuth),
            cosd(solar_zenith)]
        # for a line of equation a*x + b*y + c = 0, we calculate intercept c
        # and can derive x_0 such that crosses with line y = 0: x_0 = - c / a
        list_x_shadows = []
        list_shadow_line_pvarrays = []
        # TODO: speed improvement can be made by translating the shadow
        # boundaries and removing most of the for loop calculation
        pvrow = None
        for idx_pvrow, pvrow in enumerate(self.pvrows):
            self.has_direct_shading = False
            x1_shadow, x2_shadow = pvrow.get_shadow_bounds(solar_2d_vector)
            list_x_shadows.append((x1_shadow, x2_shadow))
            # Check if there is direct shading: if yes, the shadows will
            # be grouped into one continuous shadow
            if idx_pvrow == 1:
                if list_x_shadows[0][1] > list_x_shadows[1][0]:
                    self.has_direct_shading = True
                    # Get the bounds of the big shadow
                    x1_shadow = list_x_shadows[0][0]
                    _, x2_shadow = (self.pvrows[-1]
                                    .get_shadow_bounds(solar_2d_vector))
                    shadow_geometry = LineString([(x1_shadow, 0),
                                                  (x2_shadow, 0)])
                    shadow_line_pvarray = LinePVArray(geometry=shadow_geometry,
                                                      line_type='ground',
                                                      shaded=True)
                    list_shadow_line_pvarrays = [shadow_line_pvarray]
                    for pvrow_inner_loop in self.pvrows:
                        pvrow_inner_loop.shadow = shadow_line_pvarray
                    break
            shadow_geometry = LineString([(x1_shadow, 0), (x2_shadow, 0)])
            shadow_line_pvarray = LinePVArray(geometry=shadow_geometry,
                                              line_type='ground',
                                              shaded=True)
            list_shadow_line_pvarrays.append(shadow_line_pvarray)
            pvrow.shadow = shadow_line_pvarray

        self.solar_2d_vector = solar_2d_vector
        pvrow.shadow_line_index = (
            self.line_registry.pvgeometry.add(list_shadow_line_pvarrays))

    def create_ill_ground(self):
        """
        Create illuminated ground areas between shadows and add them to the
        line registry.
        The function assumes that the shadows are ordered and sorted from left
        to right.

        :return: None; updated :attr:`line_registry`
        """
        df_bounds_shadows = (self.line_registry
                             .loc[(self.line_registry['line_type'] == 'ground')
                                  & self.line_registry.shaded]
                             .pvgeometry.bounds)
        shadow_indices = df_bounds_shadows.index
        self.illum_ground_indices = []
        # Use the boundary pts defined by each shadow object to find the 2
        # points necessary to build the illuminated ground line in-between shad
        if df_bounds_shadows.shape[0] > 1:
            for idx in range(df_bounds_shadows.shape[0] - 1):
                point_1 = Point(
                    df_bounds_shadows.loc[shadow_indices[idx], 'maxx'],
                    df_bounds_shadows.loc[shadow_indices[idx], 'maxy'])
                point_2 = Point(
                    df_bounds_shadows.loc[shadow_indices[idx + 1], 'minx'],
                    df_bounds_shadows.loc[shadow_indices[idx + 1], 'miny']
                )
                if point_1 != point_2:
                    # If the two points are different, it means that there is
                    # some illum ground between the shadows -> create geom
                    geometry = LineString([point_1, point_2])
                    ill_gnd_line_pvarray = LinePVArray(geometry=geometry,
                                                       line_type='ground',
                                                       shaded=False)
                    self.illum_ground_indices.append(
                        self.line_registry.pvgeometry.add(
                            [ill_gnd_line_pvarray]))

    def find_edge_points(self):
        """
        Edge points are defined as the virtual intersection of the pvrow
        lines and the ground. They determine what part of the ground the front
        surface of the pvrow sees, and same for the back surface.

        :return: list of :class:`shapely.Point` objects; the
            :attr:`line_registry` is also updated
        """

        # --- Find edge points
        # Do it for each line in the pvrows
        edge_points = []
        for line in self.pvrows[0].lines:
            b1 = line['geometry'].boundary[0]
            b2 = line['geometry'].boundary[1]
            # Find the edge point in this case: just math at this point
            edge_pt = find_edge_point(b1, b2)
            self.line_registry.loc[
                self.line_registry.pvrow_index == 0,
                'edge_point'] = Series(edge_pt).values
            edge_points.append(edge_pt)
        # Use simple vector translation to add other edge points for other
        # trackers
        if (len(self.pvrows) > 1) & (len(edge_points) > 0):
            new_edge_points = []
            for i in range(len(self.pvrows) - 1):
                for edge_pt in edge_points:
                    new_point = Point(edge_pt.x + (i + 1)
                                      * self.pvrow_distance,
                                      edge_pt.y)
                    # FIXME: this is not going to work if not single line
                    self.line_registry.loc[
                        self.line_registry.pvrow_index == i + 1,
                        'edge_point'] = Series(new_point).values
                    new_edge_points.append(new_point)
            edge_points += new_edge_points

        return edge_points

    def create_remaining_illum_ground(self, edge_points):
        """
        Create the remaining illuminated parts of the ground, at the outer
        edges of the PV array.
        The areas are supposed to be infinite, but for model simplicity they
        are implemented as being very large (fixed values).

        :param list edge_points: **sorted** list of :class:`shapely.Point`
            objects representing the intersection of PV row lines and the
            ground
        :return: None; updating :attr:`line_registry`
        """
        if edge_points:
            x_min_edge_points = edge_points[0].x
            x_max_edge_points = edge_points[-1].x
        else:
            x_min_edge_points = DEFAULT_EDGE_PT_X
            x_max_edge_points = DEFAULT_EDGE_PT_X

        # Find bounds of shadows
        df_bounds_shadows = (self.line_registry
                             .loc[(self.line_registry['line_type'] == 'ground')
                                  & self.line_registry.shaded]
                             .pvgeometry.bounds)
        shadow_indices = df_bounds_shadows.index

        # Take the outermost shadow points to create the remaining illuminated
        # ground areas that are not between shadows
        # On the left side:
        min_x_ground = min(MIN_X_GROUND,
                           df_bounds_shadows.loc[shadow_indices[0], 'minx']
                           - DELTA_MAX_MIN_GROUND_WHEN_TOO_SMALL_BIG)
        geometry_left = LineString([
            (min(x_min_edge_points, min_x_ground), Y_GROUND),
            (df_bounds_shadows.loc[shadow_indices[0], 'minx'],
             df_bounds_shadows.loc[shadow_indices[0], 'miny']),
        ])
        ill_gnd_left = LinePVArray(geometry=geometry_left,
                                   line_type='ground',
                                   shaded=False)
        # On the right side
        max_x_ground = max(MAX_X_GROUND,
                           df_bounds_shadows.loc[shadow_indices[-1], 'maxx']
                           + DELTA_MAX_MIN_GROUND_WHEN_TOO_SMALL_BIG)
        geometry_right = LineString([
            (df_bounds_shadows.loc[shadow_indices[-1], 'maxx'],
             df_bounds_shadows.loc[shadow_indices[-1], 'maxy']),
            (max(x_max_edge_points, max_x_ground), Y_GROUND)
        ])
        ill_gnd_right = LinePVArray(geometry=geometry_right,
                                    line_type='ground',
                                    shaded=False)

        self.illum_ground_indices.append(
            self.line_registry.pvgeometry.add(
                [ill_gnd_left, ill_gnd_right]))

    def calculate_interrow_direct_shading(self, sun_on_front_surface):
        """
        Calculate inter-row direct shading and  break up PV row objects into
        shaded and unshaded parts.

        :param bool sun_on_front_surface: flag check if sun is incident on
            front surface
        :return: None; updating :attr:`line_registry` with additional entries
        """
        # Find the direction of shading
        # Direct shading calculation must be specific to the PVRow class
        # Shading is said "forward" if the shadow of the pvrow is on the
        # right side of the pvrow
        shading_is_forward = (self.pvrows[0].shadow['geometry']
                              .bounds[0] >=
                              self.pvrows[0].left_point.x)
        # Determine if front or back surface has direct shading
        if sun_on_front_surface:
            side_shaded = 'front'
        else:
            side_shaded = 'back'

        if shading_is_forward:
            for idx_pvrow in range(1, self.n_pvrows):
                # idx_pvrow is the index of the shaded pvrow
                pvrow = self.pvrows[idx_pvrow - 1]
                adjacent_pvrow = self.pvrows[idx_pvrow]
                # Shadows from left to right: find vector of shadow
                top_point_vector = pvrow.highest_point
                x1_shadow, x2_shadow = pvrow.get_shadow_bounds(
                    self.solar_2d_vector)
                ground_point = Point(x2_shadow, Y_GROUND)
                linestring_shadow = LineString([top_point_vector,
                                                ground_point])
                # FIXME: we do not want to create a line_registry object
                self.surface_registry.pvgeometry.split_pvrow_geometry(
                    idx_pvrow,
                    linestring_shadow,
                    adjacent_pvrow.highest_point,
                    side_shaded
                )
        else:
            for idx_pvrow in range(self.n_pvrows - 2, -1, -1):
                # idx_pvrow is the index of the shaded pvrow
                pvrow = self.pvrows[idx_pvrow + 1]
                adjacent_pvrow = self.pvrows[idx_pvrow]
                # Shadows from right to left: find vector of shadow
                top_point_vector = pvrow.highest_point
                x1_shadow, x2_shadow = (pvrow
                                        .get_shadow_bounds(
                                            self.solar_2d_vector)
                                        )
                ground_point = Point(x1_shadow, Y_GROUND)
                linestring_shadow = LineString(
                    [top_point_vector, ground_point])
                # FIXME: we do not want to create a line_registry object
                self.surface_registry.pvgeometry.split_pvrow_geometry(
                    idx_pvrow,
                    linestring_shadow,
                    adjacent_pvrow.highest_point,
                    side_shaded
                )

# ------- Surface creation
    def create_surface_registry(self):
        """
        Create ``surface_registry`` attribute from :attr:`line_registry`.
        One of the big differences is that the ``surface_registry`` is able to
        distinguish the two sides of a PV row object. For instance it will
        make sure to record that only one side of a PV row can have direct
        shading, or that only one side may be discretized. The names of the two
        sides are 'front' and 'back'.

        :return: None; creating ``surface_registry`` attribute
        """

        front_surface_registry = copy.copy(self.line_registry)
        front_surface_registry.loc[:, 'surface_side'] = 'front'
        # Create pvrow back surfaces
        back_surface_registry = front_surface_registry.loc[
            front_surface_registry.line_type == 'pvrow'].copy()
        back_surface_registry.loc[:, 'surface_side'] = 'back'
        # Merge two registries together
        self.surface_registry = (
            front_surface_registry
            .append(back_surface_registry)
            .assign(line_registry_index=lambda x: x.index.astype(int))
            .reset_index(drop=True)
        )

        # Discretize surfaces specified by user
        self.discretize_surfaces()

    def discretize_surfaces(self):
        """
        Discretize PV row surfaces using the inputs provided in the class
        constructor. New entries will be added to the ``surface_registry``.

        :return: None; updating :attr:`surface_registry`
        """

        for cut in self.cut:
            pvrow_index = cut[0]
            n_segments = cut[1]
            side = cut[2]
            self.pvrows[pvrow_index].calculate_cut_points(n_segments)
            self.surface_registry.pvgeometry.cut_pvrow_geometry(
                self.pvrows[pvrow_index].cut_points, pvrow_index, side,
                count_segments=True)

# ------- View matrix creation
    def create_view_matrix(self):
        """
        Create the ``view_matrix`` and the ``args_matrix``, which records which
        surface sees which surface, as well as the "type" of view that it is,
        and potential "obstructing" objects.
        The logic here can be a little complex, and there may be ways to
        simplify it.

        :return: ``view_matrix``, ``args_matrix``; both :class:`numpy.array`
            objects and containing the "type" of views of each finite surface
            to the others, and additional arguments like "obstructing" objects
        """

        # view matrix will contain the view relationships between each surface
        view_matrix = np.zeros((self.surface_registry.shape[0] + 1,
                                self.surface_registry.shape[0] + 1), dtype=int)
        # args matrix will contain the obstructng objects of views for instance
        args_matrix = np.zeros((self.surface_registry.shape[0] + 1,
                                self.surface_registry.shape[0] + 1),
                               dtype=object)
        args_matrix[:] = None

        # All surface indices need to be grouped and tracked for simplification
        indices_front_pvrows = self.surface_registry.loc[
            (self.surface_registry.line_type == 'pvrow')
            & (self.surface_registry.surface_side == 'front')].index.values
        indices_back_pvrows = self.surface_registry.loc[
            (self.surface_registry.line_type == 'pvrow')
            & (self.surface_registry.surface_side == 'back')].index.values
        indices_ground = self.surface_registry.loc[
            self.surface_registry.line_type == 'ground'
        ].index.values
        index_sky_dome = np.array([view_matrix.shape[0] - 1])

        # The ground will always see the sky
        # Use broadcasting for assigning values to subarrays of view matrix
        # Could also use np.ix_
        view_matrix[indices_ground[:, np.newaxis],
                    index_sky_dome] = VIEW_DICT["ground_sky"]

        # The pvrow front surface is always either flat or pointing to the
        # left by design
        pvrow_is_flat = (self.surface_tilt == 0.)
        if pvrow_is_flat:
            # Only back surface can see the ground
            view_matrix[indices_back_pvrows[:, np.newaxis],
                        indices_ground] = VIEW_DICT["back_gnd"]
            view_matrix[indices_ground[:, np.newaxis],
                        indices_back_pvrows] = VIEW_DICT["gnd_back"]
            # The front side only sees the sky
            view_matrix[indices_front_pvrows[:, np.newaxis],
                        index_sky_dome] = VIEW_DICT["front_sky"]
        else:
            # Find ground centroid values
            ground_registry = self.surface_registry.loc[indices_ground,
                                                        :].copy()
            # Need to create a registry to use geometry functions like "bounds"
            centroids = Registry(self.surface_registry.pvgeometry.centroid,
                                 columns=['geometry'])
            ground_registry[
                'x_centroid'] = (centroids.pvgeometry.bounds.minx)

            # All pvrow surfaces see the sky dome
            view_matrix[indices_back_pvrows[:, np.newaxis],
                        index_sky_dome] = VIEW_DICT["back_sky"]
            view_matrix[indices_front_pvrows[:, np.newaxis],
                        index_sky_dome] = VIEW_DICT["front_sky"]

            # Initialize last indices for interrow views
            last_indices_back_pvrow = None

            # PVRow neighbors for each PVRow
            pvrows_list = self.pvrows + [None]

            for idx, pvrow in enumerate(self.pvrows):
                # Get indices specific to pvrow
                indices_back_pvrow = self.surface_registry.loc[
                    (self.surface_registry.pvrow_index == idx)
                    & (self.surface_registry.surface_side == 'back')
                ].index.values
                indices_front_pvrow = self.surface_registry.loc[
                    (self.surface_registry.pvrow_index == idx)
                    & (self.surface_registry.surface_side == 'front')
                ].index.values

                # --- Find the lines that front and back see on the ground
                # Get edge point of this pvrow
                edge_pt = self.surface_registry.loc[indices_front_pvrow[0],
                                                    'edge_point']
                # Find the ground lines laying on the right or left side of pt
                indices_ground_right_of_edge_pt = ground_registry.loc[
                    edge_pt.x < ground_registry.x_centroid
                ].index.values
                indices_ground_left_of_edge_pt = ground_registry.loc[
                    edge_pt.x > ground_registry.x_centroid
                ].index.values

                # Initialize variables for obstructions and Hottel's method
                right_neighbor_pvrow = pvrows_list[idx + 1]
                left_neighbor_pvrow = pvrows_list[idx - 1]
                indices_ground_seen_by_front = None
                indices_ground_seen_by_back = None

                # The projection of normal of front surface onto ground points
                # to the left (by design of PVRow objects)
                indices_ground_seen_by_front = (
                    indices_ground_left_of_edge_pt)
                indices_ground_seen_by_back = (
                    indices_ground_right_of_edge_pt
                )
                # Finding any obstructing pv rows
                front_obstruction_pvrow = left_neighbor_pvrow
                back_obstruction_pvrow = right_neighbor_pvrow
                # Save the PV row neighbor index values
                if right_neighbor_pvrow is not None:
                    self.surface_registry.loc[
                        (self.surface_registry.pvrow_index == pvrow.index)
                        & (self.surface_registry.surface_side == 'back'),
                        'index_pvrow_neighbor'
                    ] = right_neighbor_pvrow.index
                if left_neighbor_pvrow is not None:
                    self.surface_registry.loc[
                        (self.surface_registry.pvrow_index == pvrow.index)
                        & (self.surface_registry.surface_side == 'front'),
                        'index_pvrow_neighbor'
                    ] = left_neighbor_pvrow.index

                # Front and back sides see different lines on the ground
                if indices_ground_seen_by_back is not None:
                    # Define the views in the matrix
                    view_matrix[indices_back_pvrow[:, np.newaxis],
                                indices_ground_seen_by_back] = (
                        VIEW_DICT["back_gnd_obst"])
                    view_matrix[indices_ground_seen_by_back[:, np.newaxis],
                                indices_back_pvrow] = (
                        VIEW_DICT["gnd_back_obst"])
                    # Define the obstructing pv rows in the matrix
                    args_matrix[indices_back_pvrow[:, np.newaxis],
                                indices_ground_seen_by_back] = (
                        back_obstruction_pvrow)
                    args_matrix[indices_ground_seen_by_back[:, np.newaxis],
                                indices_back_pvrow] = (
                        back_obstruction_pvrow)
                if indices_ground_seen_by_front is not None:
                    view_matrix[indices_front_pvrow[:, np.newaxis],
                                indices_ground_seen_by_front] = (
                        VIEW_DICT["front_gnd_obst"])
                    view_matrix[indices_ground_seen_by_front[:, np.newaxis],
                                indices_front_pvrow] = (
                        VIEW_DICT["gnd_front_obst"])
                    # Define the obstructing pv rows in the matrix
                    args_matrix[indices_front_pvrow[:, np.newaxis],
                                indices_ground_seen_by_front] = (
                        front_obstruction_pvrow)
                    args_matrix[indices_ground_seen_by_front[:, np.newaxis],
                                indices_front_pvrow] = (
                        front_obstruction_pvrow)

                # --- Find the views between neighbor pv rows
                if last_indices_back_pvrow is not None:
                    # pvrow to pvrow view
                    view_matrix[last_indices_back_pvrow[:, np.newaxis],
                                indices_front_pvrow] = (
                        VIEW_DICT["pvrows"])
                    view_matrix[indices_front_pvrow[:, np.newaxis],
                                last_indices_back_pvrow] = (
                        VIEW_DICT["pvrows"])

                # Save last indices for next pvrow interaction
                last_indices_back_pvrow = indices_back_pvrow

        return view_matrix, args_matrix
