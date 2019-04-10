"""Module containing irradiance models used with pv array geometries"""
from pvlib.tools import cosd, sind
from pvlib.irradiance import aoi as aoi_function


class BaseModel(object):
    """Base class for irradiance models"""

    def __init__(self):
        pass

    def apply_irradiance(self, pvarray, DNI, DHI):
        raise NotImplementedError


class IsotropicOrdered(BaseModel):
    """Diffuse isotropic sky model for
    :py:class:`~pvfactors.geometry.OrderedPVArray`"""

    params = ['direct']

    def __init__(self):
        pass

    @staticmethod
    def apply_irradiance(pvarray, DNI):

        if not pvarray._surfaces_indexed:
            pvarray.index_all_surfaces()

        # --- Fit
        zenith = pvarray.solar_zenith
        tilt = pvarray.surface_tilt
        solar_azimuth = pvarray.solar_azimuth
        surface_azimuth = pvarray.surface_azimuth

        # Beam components
        dni_ground = DNI * cosd(zenith)
        dni_front_pvrow = 0
        dni_back_pvrow = 0

        aoi_front_pvrow = aoi_function(tilt, surface_azimuth, zenith,
                                       solar_azimuth)
        if pvarray.illum_side == 'front':
            dni_front_pvrow = DNI * cosd(aoi_front_pvrow)
        else:
            dni_back_pvrow = DNI * cosd(180. - aoi_front_pvrow)

        # --- Apply
        for seg in pvarray.ground.list_segments:
            seg._illum_collection.set_param('direct', dni_ground)
            seg._shaded_collection.set_param('direct', 0.)

        for pvrow in pvarray.pvrows:
            # Front
            for seg in pvrow.front.list_segments:
                seg._illum_collection.set_param('direct', dni_front_pvrow)
                seg._shaded_collection.set_param('direct', 0.)
            # Back
            for seg in pvrow.back.list_segments:
                seg._illum_collection.set_param('direct', dni_back_pvrow)
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
