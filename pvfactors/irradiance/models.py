"""Module containing irradiance models used with pv array geometries"""
from pvlib.tools import cosd
from pvlib.irradiance import aoi as aoi_function
import numpy as np
from pvfactors.irradiance.utils import \
    perez_diffuse_luminance, calculate_horizon_band_shading
from pvfactors.irradiance.base import BaseModel
from pvfactors.config import DEFAULT_HORIZON_BAND_ANGLE


class IsotropicOrdered(BaseModel):
    """Diffuse isotropic sky model for
    :py:class:`~pvfactors.geometry.OrderedPVArray`"""

    params = ['rho', 'direct']
    cats = ['ground', 'front_pvrow', 'back_pvrow']
    irradiance_comp = ['direct']

    def __init__(self):
        self.direct = dict.fromkeys(self.cats)
        self.isotropic_luminance = None

    def fit(self, timestamps, DNI, DHI, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth):
        """Use vectorization to calculate values used for irradiance model"""
        # Make sure getting array-like values
        if np.isscalar(DNI):
            timestamps = [timestamps]
            DNI = np.array([DNI])
            DHI = np.array([DHI])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])

        # Save diffuse light
        self.isotropic_luminance = DHI

        # DNI seen by ground illuminated surfaces
        self.direct['ground'] = DNI * cosd(solar_zenith)

        # Calculate AOI on front pvrow using pvlib implementation
        aoi_front_pvrow = aoi_function(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
        aoi_back_pvrow = 180. - aoi_front_pvrow

        # DNI seen by pvrow illuminated surfaces
        front_is_illum = aoi_front_pvrow <= 90
        self.direct['front_pvrow'] = np.where(
            front_is_illum, DNI * cosd(aoi_front_pvrow), 0.)
        self.direct['back_pvrow'] = np.where(
            ~front_is_illum, DNI * cosd(aoi_back_pvrow), 0.)

    def transform(self, pvarray, idx=0):
        """Apply calculated irradiance values to PV array"""

        for seg in pvarray.ground.list_segments:
            seg._illum_collection.update_params(
                {'direct': self.direct['ground'][idx]})
            seg._shaded_collection.update_params({'direct': 0.})

        for pvrow in pvarray.pvrows:
            # Front
            for seg in pvrow.front.list_segments:
                seg._illum_collection.update_params(
                    {'direct': self.direct['front_pvrow'][idx]})
                seg._shaded_collection.update_params({'direct': 0.})
            # Back
            for seg in pvrow.back.list_segments:
                seg._illum_collection.update_params(
                    {'direct': self.direct['back_pvrow'][idx]})
                seg._shaded_collection.update_params({'direct': 0.})

        # Sum up the necessary parameters to form the irradiance vector
        irradiance_vec = self.get_irradiance_vector(pvarray)
        irradiance_vec.append(self.isotropic_luminance[idx])

        return np.array(irradiance_vec)


class HybridPerezOrdered(BaseModel):
    """Model is based off Perez diffuse light model, but
    applied to pvfactors :py:class:`~pvfactors.geometry.OrderedPVArray`"""

    params = ['rho', 'direct', 'isotropic', 'circumsolar', 'horizon']
    cats = ['ground', 'front_pvrow', 'back_pvrow']
    irradiance_comp = ['direct', 'circumsolar', 'horizon']

    def __init__(self, horizon_band_angle=DEFAULT_HORIZON_BAND_ANGLE):
        self.direct = dict.fromkeys(self.cats)
        self.circumsolar = dict.fromkeys(self.cats)
        self.horizon = dict.fromkeys(self.cats)
        self.isotropic_luminance = None
        self.horizon_band_angle = horizon_band_angle

    def fit(self, timestamps, DNI, DHI, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth):
        """Use vectorization to calculate values used for irradiance model"""
        # Make sure getting array-like values
        if np.isscalar(DNI):
            timestamps = [timestamps]
            DNI = np.array([DNI])
            DHI = np.array([DHI])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])

        # Size of timeseries values
        n = len(DNI)

        # Calculate terms from Perez model
        luminance_circumsolar, luminance_isotropic, poa_horizon, \
            poa_circumsolar_front, poa_circumsolar_back, \
            aoi_front_pvrow, aoi_back_pvrow = \
            self.calculate_luminance_poa_components(
                timestamps, DNI, DHI, solar_zenith, solar_azimuth,
                surface_tilt, surface_azimuth)

        # Save isotropic luminance
        self.isotropic_luminance = luminance_isotropic

        # Ground surfaces
        self.direct['ground'] = DNI * cosd(solar_zenith)
        self.circumsolar['ground'] = luminance_circumsolar
        self.horizon['ground'] = np.zeros(n)

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
            seg._illum_collection.update_params(
                {'direct': self.direct['ground'][idx],
                 'circumsolar': self.circumsolar['ground'][idx],
                 'horizon': 0.})
            seg._shaded_collection.update_params(
                {'direct': 0.,
                 'circumsolar': 0.,
                 'horizon': 0.})

        pvrows = pvarray.pvrows
        for idx_pvrow, pvrow in enumerate(pvarray.pvrows):
            # Front
            for seg in pvrow.front.list_segments:
                seg._illum_collection.update_params(
                    {'direct': self.direct['front_pvrow'][idx],
                     'circumsolar': self.circumsolar['front_pvrow'][idx],
                     'horizon': self.horizon['front_pvrow'][idx]})
                seg._shaded_collection.update_params(
                    {'direct': 0.,
                     'circumsolar': 0.,
                     'horizon': self.horizon['front_pvrow'][idx]})
            # Back: apply back surface horizon shading
            for seg in pvrow.back.list_segments:
                # Illum
                idx_neighbor = pvarray.back_neighbors[idx_pvrow]
                for surf in seg._illum_collection.list_surfaces:
                    hor_shd_pct = self.calculate_hor_shd_pct(
                        surf, idx_neighbor, pvrows)
                    surf.update_params(
                        {'direct': self.direct['back_pvrow'][idx],
                         'circumsolar': self.circumsolar['back_pvrow'][idx],
                         'horizon': self.horizon['back_pvrow'][idx] *
                         (1. - hor_shd_pct / 100.),
                         'horizon_shd_pct': hor_shd_pct})
                # Shaded
                for surf in seg._shaded_collection.list_surfaces:
                    hor_shd_pct = self.calculate_hor_shd_pct(
                        surf, idx_neighbor, pvrows)
                    surf.update_params(
                        {'direct': 0.,
                         'circumsolar': 0.,
                         'horizon': self.horizon['back_pvrow'][idx] *
                         (1. - hor_shd_pct / 100.),
                         'horizon_shd_pct': hor_shd_pct})

        # Sum up the necessary parameters to form the irradiance vector
        irradiance_vec = self.get_irradiance_vector(pvarray)
        irradiance_vec.append(self.isotropic_luminance[idx])

        return np.array(irradiance_vec)

    def calculate_hor_shd_pct(self, surface, idx_neighbor, pvrows):
        """Calculate horizon band shading percentage"""

        horizon_shading_pct = 0.
        if idx_neighbor is not None:
            centroid = surface.centroid
            neighbor_point = pvrows[idx_neighbor].highest_point
            shading_angle = np.abs(np.arctan(
                (neighbor_point.y - centroid.y) /
                (neighbor_point.x - centroid.x))
            ) * 180. / np.pi
            horizon_shading_pct = calculate_horizon_band_shading(
                shading_angle, self.horizon_band_angle)

        return horizon_shading_pct

    @staticmethod
    def calculate_luminance_poa_components(
            timestamps, DNI, DHI, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth):

        # Calculations from utils function
        df_inputs = perez_diffuse_luminance(
            timestamps, surface_tilt, surface_azimuth, solar_zenith,
            solar_azimuth, DNI, DHI)

        luminance_isotropic = df_inputs.luminance_isotropic.values
        luminance_circumsolar = df_inputs.luminance_circumsolar.values
        poa_horizon = df_inputs.poa_horizon.values
        poa_circumsolar_front = df_inputs.poa_circumsolar.values

        # Calculate AOI on front pvrow using pvlib implementation
        aoi_front_pvrow = aoi_function(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
        aoi_back_pvrow = 180. - aoi_front_pvrow

        # Will be used for back surface adjustments: from Perez model
        # FIXME: pvlib clips the angle values to calculate vf -> adjust here
        vf_circumsolar_backsurface = \
            cosd(aoi_back_pvrow) / cosd(solar_zenith)
        poa_circumsolar_back = \
            luminance_circumsolar * vf_circumsolar_backsurface

        # TODO: return only >0 values for poa_horizon
        poa_horizon = np.abs(poa_horizon)

        return luminance_isotropic, luminance_circumsolar, poa_horizon, \
            poa_circumsolar_front, poa_circumsolar_back, \
            aoi_front_pvrow, aoi_back_pvrow
