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

    params = ['direct']

    def __init__(self):
        self.dni_ground = None
        self.dni_front_pvrow = None
        self.dni_back_pvrow = None

    def fit(self, DNI, solar_zenith, solar_azimuth, surface_tilt,
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

    params = ['direct', 'isotropic', 'circumsolar', 'horizon']

    def __init__(self, DNI=[], DHI=[]):
        pass

    def apply_irradiance(self, pvarray):
        pass

    @staticmethod
    def get_luminance(DNI, DHI, solar_zenith, solar_azimuth,
                      surface_tilt, surface_azimuth):
        pass
