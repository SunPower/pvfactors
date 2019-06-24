"""Module containing irradiance models used with pv array geometries"""

from pvlib.tools import cosd
from pvlib.irradiance import aoi as aoi_function
from pvlib.irradiance import get_total_irradiance
import numpy as np
from pvfactors.irradiance.utils import \
    perez_diffuse_luminance, calculate_horizon_band_shading, \
    calculate_circumsolar_shading
from pvfactors.irradiance.base import BaseModel
from pvfactors.config import \
    DEFAULT_HORIZON_BAND_ANGLE, SKY_REFLECTIVITY_DUMMY, \
    DEFAULT_CIRCUMSOLAR_ANGLE


class IsotropicOrdered(BaseModel):
    """Diffuse isotropic sky model for
    :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray`. It will
    calculate the appropriate values for an isotropic sky dome and apply
    it to the PV array."""

    params = ['rho', 'inv_rho', 'direct', 'isotropic', 'reflection']
    cats = ['ground', 'front_pvrow', 'back_pvrow']
    irradiance_comp = ['direct']

    def __init__(self):
        """Initialize irradiance model values that will be saved later on."""
        self.direct = dict.fromkeys(self.cats)
        self.total_perez = dict.fromkeys(self.cats)
        self.isotropic_luminance = None
        self.rho_front = None
        self.rho_back = None
        self.albedo = None
        self.GHI = None
        self.DHI = None

    def fit(self, timestamps, DNI, DHI, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, rho_front, rho_back, albedo,
            GHI=None):
        """Use vectorization to calculate values used for the isotropic
        irradiance model.

        Parameters
        ----------
        timestamps : array-like
            List of timestamps of the simulation.
        DNI : array-like
            Direct normal irradiance values [W/m2]
        DHI : array-like
            Diffuse horizontal irradiance values [W/m2]
        solar_zenith : array-like
            Solar zenith angles [deg]
        solar_azimuth : array-like
            Solar azimuth angles [deg]
        surface_tilt : array-like
            Surface tilt angles, from 0 to 180 [deg]
        surface_azimuth : array-like
            Surface azimuth angles [deg]
        rho_front : float
            Reflectivity of the front side of the PV rows
        rho_back : float
            Reflectivity of the back side of the PV rows
        albedo : array-like
            Albedo values (or ground reflectivity)
        GHI : array-like, optional
            Global horizontal irradiance [W/m2], if not provided, will be
            calculated from DNI and DHI (Default = None)
        """
        # Make sure getting array-like values
        if np.isscalar(DNI):
            timestamps = [timestamps]
            DNI = np.array([DNI])
            DHI = np.array([DHI])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])
            if GHI is not None:
                GHI = np.array([GHI])
        # Length of arrays
        n = len(DNI)
        # Make sure that albedo is a vector
        if np.isscalar(albedo):
            albedo = albedo * np.ones(n)

        # Save and calculate total POA values from Perez model
        if GHI is None:
            # Calculate GHI if not specified
            GHI = DNI * cosd(solar_zenith) + DHI
        self.GHI = GHI
        self.DHI = DHI
        perez_front_pvrow = get_total_irradiance(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            DNI, GHI, DHI, albedo=albedo)

        # Save diffuse light
        self.isotropic_luminance = DHI
        self.rho_front = rho_front
        self.rho_back = rho_back
        self.albedo = albedo

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
        self.total_perez['front_pvrow'] = perez_front_pvrow['poa_global']

    def transform(self, pvarray, idx=0):
        """Apply calculated irradiance values to PV array.

        Parameters
        ----------
        pvarray : PV array object
            PV array on which the calculated irradiance values will be applied
        idx : int, optional
            Index of the irradiance values to apply to the PV array (in the
            whole timeseries values)

        Returns
        -------
        irradiance_vec : numpy array
            List of summed up non-reflective irradiance values for all surfaces
            and sky
        rho_vec : numpy array
            List of reflectivity values for all surfaces and sky
        invrho_vec : numpy array
            List of inverse reflectivity for all surfaces and sky
        total_perez_vec : numpy array
            List of total perez transposed irradiance values for all surfaces
            and sky
        """

        for seg in pvarray.ground.list_segments:
            seg.illum_collection.update_params(
                {'direct': self.direct['ground'][idx],
                 'rho': self.albedo[idx],
                 'inv_rho': 1. / self.albedo[idx],
                 'total_perez': self.GHI[idx]})
            seg.shaded_collection.update_params(
                {'direct': 0.,
                 'rho': self.albedo[idx],
                 'inv_rho': 1. / self.albedo[idx],
                 'total_perez': self.DHI[idx]})

        for pvrow in pvarray.pvrows:
            # Front
            for seg in pvrow.front.list_segments:
                seg.illum_collection.update_params(
                    {'direct': self.direct['front_pvrow'][idx],
                     'rho': self.rho_front,
                     'inv_rho': 1. / self.rho_front,
                     'total_perez': self.total_perez['front_pvrow'][idx]})
                seg.shaded_collection.update_params(
                    {'direct': 0.,
                     'rho': self.rho_front,
                     'inv_rho': 1. / self.rho_front,
                     'total_perez': self.total_perez['front_pvrow'][idx] -
                     self.direct['front_pvrow'][idx]})
            # Back
            for seg in pvrow.back.list_segments:
                seg.illum_collection.update_params(
                    {'direct': self.direct['back_pvrow'][idx],
                     'rho': self.rho_back,
                     'inv_rho': 1. / self.rho_back,
                     'total_perez': 0.})
                seg.shaded_collection.update_params(
                    {'direct': 0.,
                     'rho': self.rho_back,
                     'inv_rho': 1. / self.rho_back,
                     'total_perez': 0.})

        # Sum up the necessary parameters to form the irradiance vector
        irradiance_vec, rho_vec, inv_rho_vec, total_perez_vec = \
            self.get_modeling_vectors(pvarray)
        # Add sky values
        irradiance_vec.append(self.isotropic_luminance[idx])
        total_perez_vec.append(self.isotropic_luminance[idx])
        rho_vec.append(SKY_REFLECTIVITY_DUMMY)
        inv_rho_vec.append(SKY_REFLECTIVITY_DUMMY)

        return np.array(irradiance_vec), np.array(rho_vec), \
            np.array(inv_rho_vec), np.array(total_perez_vec)


class HybridPerezOrdered(BaseModel):
    """Model is based off Perez diffuse light model, and
    applied to pvfactors :py:class:`~pvfactors.geometry.OrderedPVArray`
    objects.
    The model applies direct, circumsolar, and horizon irradiance to the PV
    array surfaces.
    """

    params = ['rho', 'inv_rho', 'direct', 'isotropic', 'circumsolar',
              'horizon', 'reflection']
    cats = ['ground', 'front_pvrow', 'back_pvrow']
    irradiance_comp = ['direct', 'circumsolar', 'horizon']

    def __init__(self, horizon_band_angle=DEFAULT_HORIZON_BAND_ANGLE,
                 circumsolar_angle=DEFAULT_CIRCUMSOLAR_ANGLE,
                 circumsolar_model='uniform_disk'):
        """Initialize irradiance model values that will be saved later on.

        Parameters
        ----------
        horizon_band_angle : float, optional
            Width of the horizon band in [deg] (Default =
            DEFAULT_HORIZON_BAND_ANGLE)
        circumsolar_angle : float, optional
            Diameter of the circumsolar area in [deg] (Default =
            DEFAULT_CIRCUMSOLAR_ANGLE)
        circumsolar_model : str
            Circumsolar shading model to use (Default = 'uniform_disk')
        """
        self.direct = dict.fromkeys(self.cats)
        self.circumsolar = dict.fromkeys(self.cats)
        self.horizon = dict.fromkeys(self.cats)
        self.total_perez = dict.fromkeys(self.cats)
        self.isotropic_luminance = None
        self.horizon_band_angle = horizon_band_angle
        self.circumsolar_angle = circumsolar_angle
        self.circumsolar_model = circumsolar_model
        self.rho_front = None
        self.rho_back = None
        self.albedo = None
        self.GHI = None
        self.DNI = None

    def fit(self, timestamps, DNI, DHI, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, rho_front, rho_back, albedo,
            GHI=None):
        """Use vectorization to calculate values used for the hybrid Perez
        irradiance model.

        Parameters
        ----------
        timestamps : array-like
            List of timestamps of the simulation.
        DNI : array-like
            Direct normal irradiance values [W/m2]
        DHI : array-like
            Diffuse horizontal irradiance values [W/m2]
        solar_zenith : array-like
            Solar zenith angles [deg]
        solar_azimuth : array-like
            Solar azimuth angles [deg]
        surface_tilt : array-like
            Surface tilt angles, from 0 to 180 [deg]
        surface_azimuth : array-like
            Surface azimuth angles [deg]
        rho_front : float
            Reflectivity of the front side of the PV rows
        rho_back : float
            Reflectivity of the back side of the PV rows
        albedo : array-like
            Albedo values (or ground reflectivity)
        GHI : array-like, optional
            Global horizontal irradiance [W/m2], if not provided, will be
            calculated from DNI and DHI (Default = None)
        """
        # Make sure getting array-like values
        if np.isscalar(DNI):
            timestamps = [timestamps]
            DNI = np.array([DNI])
            DHI = np.array([DHI])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])
            if GHI is not None:
                GHI = np.array([GHI])
        # Length of arrays
        n = len(DNI)
        # Make sure that albedo is a vector
        if np.isscalar(albedo):
            albedo = albedo * np.ones(n)

        # Calculate terms from Perez model
        luminance_isotropic, luminance_circumsolar, poa_horizon, \
            poa_circumsolar_front, poa_circumsolar_back, \
            aoi_front_pvrow, aoi_back_pvrow = \
            self.calculate_luminance_poa_components(
                timestamps, DNI, DHI, solar_zenith, solar_azimuth,
                surface_tilt, surface_azimuth)

        # Save and calculate total POA values from Perez model
        if GHI is None:
            # Calculate GHI if not specified
            GHI = DNI * cosd(solar_zenith) + DHI
        self.GHI = GHI
        self.DHI = DHI
        perez_front_pvrow = get_total_irradiance(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            DNI, GHI, DHI, albedo=albedo)
        total_perez_front_pvrow = perez_front_pvrow['poa_global']

        # Save isotropic luminance
        self.isotropic_luminance = luminance_isotropic
        self.rho_front = rho_front
        self.rho_back = rho_back
        self.albedo = albedo

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
        self.total_perez['front_pvrow'] = total_perez_front_pvrow

    def transform(self, pvarray, idx=0):
        """Apply calculated irradiance values to PV array, as well as
        horizon band shading on pvrow back sides.

        Parameters
        ----------
        pvarray : PV array object
            PV array on which the calculated irradiance values will be applied
        idx : int, optional
            Index of the irradiance values to apply to the PV array (in the
            whole timeseries values)

        Returns
        -------
        irradiance_vec : numpy array
            List of summed up non-reflective irradiance values for all surfaces
            and sky
        rho_vec : numpy array
            List of reflectivity values for all surfaces and sky
        invrho_vec : numpy array
            List of inverse reflectivity for all surfaces and sky
        total_perez_vec : numpy array
            List of total perez transposed irradiance values for all surfaces
            and sky
        """

        for seg in pvarray.ground.list_segments:
            seg.illum_collection.update_params(
                {'direct': self.direct['ground'][idx],
                 'circumsolar': self.circumsolar['ground'][idx],
                 'horizon': 0.,
                 'rho': self.albedo[idx],
                 'inv_rho': 1. / self.albedo[idx],
                 'total_perez': self.GHI[idx]})
            seg.shaded_collection.update_params(
                {'direct': 0.,
                 'circumsolar': 0.,
                 'horizon': 0.,
                 'rho': self.albedo[idx],
                 'inv_rho': 1. / self.albedo[idx],
                 'total_perez': self.DHI[idx]})

        pvrows = pvarray.pvrows
        for idx_pvrow, pvrow in enumerate(pvarray.pvrows):
            # Front
            for seg in pvrow.front.list_segments:
                seg.illum_collection.update_params(
                    {'direct': self.direct['front_pvrow'][idx],
                     'circumsolar': self.circumsolar['front_pvrow'][idx],
                     'horizon': self.horizon['front_pvrow'][idx],
                     'rho': self.rho_front,
                     'inv_rho': 1. / self.rho_front,
                     'total_perez': self.total_perez['front_pvrow'][idx]})
                seg.shaded_collection.update_params(
                    {'direct': 0.,
                     'circumsolar': 0.,
                     'horizon': self.horizon['front_pvrow'][idx],
                     'rho': self.rho_front,
                     'inv_rho': 1. / self.rho_front,
                     'total_perez': self.total_perez['front_pvrow'][idx] -
                     self.direct['front_pvrow'][idx]})
            # Back: apply back surface horizon shading
            for seg in pvrow.back.list_segments:
                # Illum
                idx_neighbor = pvarray.back_neighbors[idx_pvrow]
                for surf in seg.illum_collection.list_surfaces:
                    hor_shd_pct = self.calculate_horizon_shading_pct(
                        surf, idx_neighbor, pvrows)
                    surf.update_params(
                        {'direct': self.direct['back_pvrow'][idx],
                         'circumsolar': self.circumsolar['back_pvrow'][idx],
                         'horizon': self.horizon['back_pvrow'][idx] *
                         (1. - hor_shd_pct / 100.),
                         'horizon_shd_pct': hor_shd_pct,
                         'rho': self.rho_back,
                         'inv_rho': 1. / self.rho_back,
                         'total_perez': 0.})
                # Shaded
                for surf in seg.shaded_collection.list_surfaces:
                    hor_shd_pct = self.calculate_horizon_shading_pct(
                        surf, idx_neighbor, pvrows)
                    surf.update_params(
                        {'direct': 0.,
                         'circumsolar': 0.,
                         'horizon': self.horizon['back_pvrow'][idx] *
                         (1. - hor_shd_pct / 100.),
                         'horizon_shd_pct': hor_shd_pct,
                         'rho': self.rho_back,
                         'inv_rho': 1. / self.rho_back,
                         'total_perez': 0.})

        # Sum up the necessary parameters to form the irradiance vector
        irradiance_vec, rho_vec, inv_rho_vec, total_perez_vec = \
            self.get_modeling_vectors(pvarray)
        # Add sky values
        irradiance_vec.append(self.isotropic_luminance[idx])
        total_perez_vec.append(self.isotropic_luminance[idx])
        rho_vec.append(SKY_REFLECTIVITY_DUMMY)
        inv_rho_vec.append(SKY_REFLECTIVITY_DUMMY)

        return np.array(irradiance_vec), np.array(rho_vec), \
            np.array(inv_rho_vec), np.array(total_perez_vec)

    def calculate_horizon_shading_pct(self, surface, idx_neighbor, pvrows):
        """Calculate horizon band shading percentage on surfaces of the ordered
        PV array.
        TODO: This needs to be merged with circumsolar shading for performance

        Parameters
        ----------
        surface : :py:class:`~pvfactors.geometry.base.PVSurface` object
            PV surface for which some horizon band shading will occur
        idx_neighbor : int
            Index of the neighboring PV row (can be ``None``)
        pvrows : list of :py:class:`~pvfactors.geometry.pvrow.PVRow` objects
            List of PV rows on which ``idx_neighbor`` will be used

        Returns
        -------
        horizon_shading_pct : float
            Percentage of horizon band irradiance shaded (from 0 to 100)
        """

        # TODO: should be applied to all pvrow surfaces

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

    def calculate_circumsolar_shading_pct(self, surface, idx_neighbor, pvrows,
                                          solar_2d_vector):
        """Model method to calculate circumsolar shading on surfaces of
        the ordered PV array.
        TODO: This needs to be merged with horizon shading for performance

        Parameters
        ----------
        surface : :py:class:`~pvfactors.geometry.base.PVSurface` object
            PV surface for which some horizon band shading will occur
        idx_neighbor : int
            Index of the neighboring PV row (can be ``None``)
        pvrows : list of :py:class:`~pvfactors.geometry.pvrow.PVRow` objects
            List of PV rows on which ``idx_neighbor`` will be used
        solar_2d_vector : list
            Solar vector in the 2D PV array representation

        Returns
        -------
        circ_shading_pct : float
            Percentage of circumsolar irradiance shaded (from 0 to 100)
        """

        # TODO: should be applied to all pvrow surfaces

        if idx_neighbor is not None:
            # Calculate the solar and circumsolar elevation angles in 2D plane
            solar_2d_elevation = np.abs(
                np.arctan(solar_2d_vector[1] / solar_2d_vector[0])
            ) * 180. / np.pi
            lower_angle_circumsolar = (solar_2d_elevation -
                                       self.circumsolar_angle / 2.)
            centroid = surface.centroid
            neighbor_point = pvrows[idx_neighbor].highest_point
            shading_angle = np.abs(np.arctan(
                (neighbor_point.y - centroid.y) /
                (neighbor_point.x - centroid.x))) * 180. / np.pi
            percentage_circ_angle_covered = \
                (shading_angle - lower_angle_circumsolar) \
                / self.circumsolar_angle * 100.
            circ_shading_pct = calculate_circumsolar_shading(
                percentage_circ_angle_covered, model=self.circumsolar_model)

        return circ_shading_pct

    @staticmethod
    def calculate_luminance_poa_components(
            timestamps, DNI, DHI, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth):
        """Calculate Perez-like luminance and plane-of-array irradiance values.

        Parameters
        ----------
        timestamps : array-like
            List of timestamps of the simulation.
        DNI : array-like
            Direct normal irradiance values [W/m2]
        DHI : array-like
            Diffuse horizontal irradiance values [W/m2]
        solar_zenith : array-like
            Solar zenith angles [deg]
        solar_azimuth : array-like
            Solar azimuth angles [deg]
        surface_tilt : array-like
            Surface tilt angles, from 0 to 180 [deg]
        surface_azimuth : array-like
            Surface azimuth angles [deg]

        Returns
        -------
        luminance_isotropic : numpy array
            Luminance values of the isotropic sky dome area
        luminance_circumsolar : numpy array
            Luminance values of the circumsolar area
        poa_horizon : numpy array
            Plane-of-array irradiance coming from horizon band and incident on
            the sides of the PV rows [W/m2]
        poa_circumsolar_front : numpy array
            Plane-of-array irradiance coming from the circumsolar area and
            incident on the front side of the PV rows [W/m2]
        poa_circumsolar_back : numpy array
            Plane-of-array irradiance coming from the circumsolar area and
            incident on the back side of the PV rows [W/m2]
        aoi_front_pvrow : numpy array
            Angle of incidence of direct light on front side of PV rows [deg]
        aoi_back_pvrow : numpy array
            Angle of incidence of direct light on back side of PV rows [deg]
        """

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
        vf_circumsolar_backsurface = cosd(aoi_back_pvrow) / cosd(solar_zenith)
        poa_circumsolar_back = \
            luminance_circumsolar * vf_circumsolar_backsurface

        # Return only >0 values for poa_horizon
        poa_horizon = np.abs(poa_horizon)

        return luminance_isotropic, luminance_circumsolar, poa_horizon, \
            poa_circumsolar_front, poa_circumsolar_back, \
            aoi_front_pvrow, aoi_back_pvrow
