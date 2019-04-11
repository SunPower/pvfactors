"""Module containing irradiance models used with pv array geometries"""
from pvlib.tools import cosd
from pvlib.irradiance import aoi as aoi_function
import numpy as np


class BaseModel(object):
    """Base class for irradiance models"""

    def __init__(self):
        pass

    def fit(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError


class IsotropicOrdered(BaseModel):
    """Diffuse isotropic sky model for
    :py:class:`~pvfactors.geometry.OrderedPVArray`"""

    params = ['rho', 'direct']

    def __init__(self):
        self.dni_ground = None
        self.dni_front_pvrow = None
        self.dni_back_pvrow = None

    def fit(self, DNI, DHI, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth):
        """Use vectorization to calculate values used for irradiance model"""
        # Make sure getting array-like values
        if np.isscalar(DNI):
            DNI = np.array([DNI])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])

        # DNI seen by ground illuminated surfaces
        self.dni_ground = DNI * cosd(solar_zenith)

        # Calculate AOI on front pvrow using pvlib implementation
        aoi_front_pvrow = aoi_function(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)

        # DNI seen by pvrow illuminated surfaces
        front_is_illum = aoi_front_pvrow <= 90
        self.dni_front_pvrow = np.where(front_is_illum,
                                        DNI * cosd(aoi_front_pvrow), 0.)
        self.dni_back_pvrow = np.where(~front_is_illum,
                                       DNI * cosd(180. - aoi_front_pvrow), 0.)

    def transform(self, pvarray, idx=0):
        """Apply calculated irradiance values to PV array"""

        for seg in pvarray.ground.list_segments:
            seg._illum_collection.set_param('direct', self.dni_ground[idx])
            seg._shaded_collection.set_param('direct', 0.)

        for pvrow in pvarray.pvrows:
            # Front
            for seg in pvrow.front.list_segments:
                seg._illum_collection.set_param('direct',
                                                self.dni_front_pvrow[idx])
                seg._shaded_collection.set_param('direct', 0.)
            # Back
            for seg in pvrow.back.list_segments:
                seg._illum_collection.set_param('direct',
                                                self.dni_back_pvrow[idx])
                seg._shaded_collection.set_param('direct', 0.)


class HybridPerezOrdered(BaseModel):
    """Model is based off Perez diffuse light model, but
    applied to pvfactors :py:class:`~pvfactors.geometry.OrderedPVArray`"""

    params = ['rho', 'direct', 'isotropic', 'circumsolar', 'horizon']
    cats = ['ground', 'front_pvrow', 'back_pvrow']

    def __init__(self):
        self.direct = dict.fromkeys(self.cats)
        self.circumsolar = dict.fromkeys(self.cats)
        self.horizon = dict.fromkeys(self.cats)

    def fit(self, DNI, DHI, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth):
        """Use vectorization to calculate values used for irradiance model"""
        # Make sure getting array-like values
        if np.isscalar(DNI):
            DNI = np.array([DNI])
            DHI = np.array([DHI])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])

        n = len(DNI)

        # Calculate terms from Perez model
        luminance_circumsolar, luminance_isotropic, poa_horizon, \
            poa_circumsolar_front, poa_circumsolar_back = \
            self.calculate_luminance_poa_components(
                DNI, DHI, solar_zenith, solar_azimuth, surface_tilt,
                surface_azimuth)

        # Ground surfaces
        self.direct['ground'] = DNI * cosd(solar_zenith)
        self.circumsolar['ground'] = luminance_circumsolar
        self.horizon['ground'] = np.zeros(n)

        # Calculate AOI on front pvrow using pvlib implementation
        aoi_front_pvrow = aoi_function(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
        aoi_back_pvrow = 180. - aoi_front_pvrow

        # PV row surfaces
        front_is_illum = aoi_front_pvrow <= 90
        self.direct['front_pvrow'] = np.where(
            front_is_illum, DNI * cosd(aoi_front_pvrow), 0.)
        self.direct['back_pvrow'] = np.where(
            ~front_is_illum, DNI * cosd(aoi_back_pvrow), 0.)
        self.circumsolar['front_pvrow'] = np.where(
            front_is_illum, poa_circumsolar_front, 0.)
        self.circumsolar['back_pvrow'] = np.where(
            ~front_is_illum, poa_circumsolar_back, 0.)
        self.horizon['front_pvrow'] = poa_horizon
        self.horizon['back_pvrow'] = poa_horizon

    def transform(self, pvarray, idx=0):
        """Apply calculated irradiance values to PV array"""

        # TODO: calculate horizon band shading on the back surface
        # self.apply_back_horizon_shading()

        for seg in pvarray.ground.list_segments:
            seg._illum_collection.set_param('direct', self.dni_ground[idx])
            seg._shaded_collection.set_param('direct', 0.)

        for pvrow in pvarray.pvrows:
            # Front
            for seg in pvrow.front.list_segments:
                seg._illum_collection.set_param('direct',
                                                self.dni_front_pvrow[idx])
                seg._shaded_collection.set_param('direct', 0.)
            # Back
            for seg in pvrow.back.list_segments:
                seg._illum_collection.set_param('direct',
                                                self.dni_back_pvrow[idx])
                seg._shaded_collection.set_param('direct', 0.)

        # Sum up the necessary parameters to form the irradiance terms
        irradiance_terms = None

        return irradiance_terms

    @staticmethod
    def calculate_luminance_poa_components(
            DNI, DHI, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth):

        luminance_isotropic = np.nan
        luminance_circumsolar = np.nan
        poa_horizon = np.nan
        poa_circumsolar_front = np.nan
        poa_circumsolar_back = np.nan

        # # Will be used for back surface adjustments: from Perez model
        # vf_circumsolar_backsurface = \
        # cosd(aoi_back_pvrow) / cosd(solar_zenith)
        # poa_circumsolar_back = \
        #     luminance_circumsolar * vf_circumsolar_backsurface

        # # TODO: return only >0 values for poa_horizon
        # poa_horizon = np.abs(poa_horizon)

        return luminance_isotropic, luminance_circumsolar, poa_horizon, \
            poa_circumsolar_front, poa_circumsolar_back
