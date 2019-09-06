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

    def __init__(self, rho_front=0.01, rho_back=0.03):
        """Initialize irradiance model values that will be saved later on.

        Parameters
        ----------
        rho_front : float, optional
            Reflectivity of the front side of the PV rows (default = 0.01)
        rho_back : float, optional
            Reflectivity of the back side of the PV rows (default = 0.03)
        """
        self.direct = dict.fromkeys(self.cats)
        self.total_perez = dict.fromkeys(self.cats)
        self.isotropic_luminance = None
        self.rho_front = rho_front
        self.rho_back = rho_back
        self.albedo = None
        self.GHI = None
        self.DHI = None

    def fit(self, timestamps, DNI, DHI, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, albedo,
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
            GHI = None if GHI is None else np.array([GHI])
        # Length of arrays
        n = len(DNI)
        # Make sure that albedo is a vector
        albedo = albedo * np.ones(n) if np.isscalar(albedo) else albedo

        # Save and calculate total POA values from Perez model
        GHI = DNI * cosd(solar_zenith) + DHI if GHI is None else GHI
        self.GHI = GHI
        self.DHI = DHI
        self.n_steps = n
        perez_front_pvrow = get_total_irradiance(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            DNI, GHI, DHI, albedo=albedo)

        # Save diffuse light
        self.isotropic_luminance = DHI
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

    def transform(self, pvarray):
        """Apply calculated irradiance values to PV array timeseries
        geometries: assign values as parameters to timeseries surfaces.

        Parameters
        ----------
        pvarray : PV array object
            PV array on which the calculated irradiance values will be applied
        """

        # Prepare variables
        n_steps = self.n_steps
        rho_front = self.rho_front * np.ones(n_steps)
        inv_rho_front = 1. / rho_front
        rho_back = self.rho_back * np.ones(n_steps)
        inv_rho_back = 1. / rho_back

        # Transform timeseries ground
        pvarray.ts_ground.illum_params.update(
            {'direct': self.direct['ground'],
             'rho': self.albedo,
             'inv_rho': 1. / self.albedo,
             'total_perez': self.gnd_illum})
        pvarray.ts_ground.shaded_params.update(
            {'direct': np.zeros(n_steps),
             'rho': self.albedo,
             'inv_rho': 1. / self.albedo,
             'total_perez': self.gnd_shaded})

        for ts_pvrow in pvarray.ts_pvrows:
            # Front
            for ts_seg in ts_pvrow.front.list_segments:
                ts_seg.illum.update_params(
                    {'direct': self.direct['front_pvrow'],
                     'rho': rho_front,
                     'inv_rho': inv_rho_front,
                     'total_perez': self.pvrow_illum})
                ts_seg.shaded.update_params(
                    {'direct': np.zeros(n_steps),
                     'rho': rho_front,
                     'inv_rho': inv_rho_front,
                     'total_perez': self.pvrow_shaded})
            # Back
            for ts_seg in ts_pvrow.back.list_segments:
                ts_seg.illum.update_params(
                    {'direct': self.direct['back_pvrow'],
                     'rho': rho_back,
                     'inv_rho': inv_rho_back,
                     'total_perez': np.zeros(n_steps)})
                ts_seg.shaded.update_params(
                    {'direct': np.zeros(n_steps),
                     'rho': rho_back,
                     'inv_rho': inv_rho_back,
                     'total_perez': np.zeros(n_steps)})

    def get_full_modeling_vectors(self, pvarray, idx):
        """Get the modeling vectors used in matrix calculations of mathematical
        model.

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

    @property
    def gnd_shaded(self):
        """Total timeseries irradiance incident on ground shaded areas"""
        return self.DHI

    @property
    def gnd_illum(self):
        """Total timeseries irradiance incident on ground illuminated areas"""
        return self.GHI

    @property
    def pvrow_shaded(self):
        """Total timeseries irradiance incident on PV row's front illuminated
        areas and calculated by Perez transposition"""
        return self.pvrow_illum - self.direct['front_pvrow']

    @property
    def pvrow_illum(self):
        """Total timeseries irradiance incident on PV row's front shaded
        areas and calculated by Perez transposition"""
        return self.total_perez['front_pvrow']

    @property
    def sky_luminance(self):
        """Total timeseries isotropic luminance of sky"""
        return self.isotropic_luminance


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
                 circumsolar_model='uniform_disk', rho_front=0.01,
                 rho_back=0.03):
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
        rho_front : float, optional
            Reflectivity of the front side of the PV rows (default = 0.01)
        rho_back : float, optional
            Reflectivity of the back side of the PV rows (default = 0.03)
        """
        self.direct = dict.fromkeys(self.cats)
        self.circumsolar = dict.fromkeys(self.cats)
        self.horizon = dict.fromkeys(self.cats)
        self.total_perez = dict.fromkeys(self.cats)
        self.isotropic_luminance = None
        self.horizon_band_angle = horizon_band_angle
        self.circumsolar_angle = circumsolar_angle
        self.circumsolar_model = circumsolar_model
        self.rho_front = rho_front
        self.rho_back = rho_back
        self.albedo = None
        self.GHI = None
        self.DNI = None
        self.n_steps = None

    def fit(self, timestamps, DNI, DHI, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, albedo,
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
            GHI = None if GHI is None else np.array([GHI])
        # Length of arrays
        n = len(DNI)
        # Make sure that albedo is a vector
        albedo = albedo * np.ones(n) if np.isscalar(albedo) else albedo

        # Calculate terms from Perez model
        luminance_isotropic, luminance_circumsolar, poa_horizon, \
            poa_circumsolar_front, poa_circumsolar_back, \
            aoi_front_pvrow, aoi_back_pvrow = \
            self._calculate_luminance_poa_components(
                timestamps, DNI, DHI, solar_zenith, solar_azimuth,
                surface_tilt, surface_azimuth)

        # Save and calculate total POA values from Perez model
        GHI = DNI * cosd(solar_zenith) + DHI if GHI is None else GHI
        self.GHI = GHI
        self.DHI = DHI
        self.n_steps = n
        perez_front_pvrow = get_total_irradiance(
            surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
            DNI, GHI, DHI, albedo=albedo)
        total_perez_front_pvrow = perez_front_pvrow['poa_global']

        # Save isotropic luminance
        self.isotropic_luminance = luminance_isotropic
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
        self.total_perez['front_illum_pvrow'] = total_perez_front_pvrow
        self.total_perez['front_shaded_pvrow'] = (
            total_perez_front_pvrow - self.direct['front_pvrow'])
        self.total_perez['ground_shaded'] = DHI
        self.total_perez['ground_illum'] = GHI
        self.total_perez['sky'] = luminance_isotropic

    def transform(self, pvarray):
        """Apply calculated irradiance values to PV array timeseries
        geometries: assign values as parameters to timeseries surfaces.

        Parameters
        ----------
        pvarray : PV array object
            PV array on which the calculated irradiance values will be applied
        """

        # Prepare variables
        n_steps = self.n_steps
        ts_pvrows = pvarray.ts_pvrows
        tilted_to_left = pvarray.rotation_vec > 0
        rho_front = self.rho_front * np.ones(n_steps)
        inv_rho_front = 1. / rho_front
        rho_back = self.rho_back * np.ones(n_steps)
        inv_rho_back = 1. / rho_back

        # Transform timeseries ground
        pvarray.ts_ground.illum_params.update({
            'direct': self.direct['ground'],
            'circumsolar': self.circumsolar['ground'],
            'horizon': np.zeros(n_steps),
            'rho': self.albedo,
            'inv_rho': 1. / self.albedo,
            'total_perez': self.gnd_illum})
        pvarray.ts_ground.shaded_params.update({
            'direct': np.zeros(n_steps),
            'circumsolar': np.zeros(n_steps),
            'horizon': np.zeros(n_steps),
            'rho': self.albedo,
            'inv_rho': 1. / self.albedo,
            'total_perez': self.gnd_shaded})

        # Transform timeseries PV rows
        for idx_pvrow, ts_pvrow in enumerate(ts_pvrows):
            # Front
            for ts_seg in ts_pvrow.front.list_segments:
                ts_seg.illum.update_params({
                    'direct': self.direct['front_pvrow'],
                    'circumsolar': self.circumsolar['front_pvrow'],
                    'horizon': self.horizon['front_pvrow'],
                    'rho': rho_front,
                    'inv_rho': inv_rho_front,
                    'total_perez': self.pvrow_illum})
                ts_seg.shaded.update_params({
                    'direct': np.zeros(n_steps),
                    'circumsolar': np.zeros(n_steps),
                    'horizon': self.horizon['front_pvrow'],
                    'rho': rho_front,
                    'inv_rho': inv_rho_front,
                    'total_perez': self.pvrow_shaded})
            # Back: apply back surface horizon shading
            for ts_seg in ts_pvrow.back.list_segments:
                # Illum
                centroid_illum = ts_seg.illum.centroid
                hor_shd_pct_illum = self._calculate_horizon_shading_pct_ts(
                    ts_pvrows, centroid_illum, idx_pvrow, tilted_to_left,
                    is_back_side=True)
                ts_seg.illum.update_params({
                    'direct': self.direct['back_pvrow'],
                    'circumsolar': self.circumsolar['back_pvrow'],
                    'horizon': self.horizon['back_pvrow'] *
                    (1. - hor_shd_pct_illum / 100.),
                    'horizon_unshaded': self.horizon['back_pvrow'],
                    'horizon_shd_pct': hor_shd_pct_illum,
                    'rho': rho_back,
                    'inv_rho': inv_rho_back,
                    'total_perez': np.zeros(n_steps)})
                # Back
                centroid_shaded = ts_seg.shaded.centroid
                hor_shd_pct_shaded = self._calculate_horizon_shading_pct_ts(
                    ts_pvrows, centroid_shaded, idx_pvrow, tilted_to_left,
                    is_back_side=True)
                ts_seg.shaded.update_params({
                    'direct': np.zeros(n_steps),
                    'circumsolar': np.zeros(n_steps),
                    'horizon': self.horizon['back_pvrow'] *
                    (1. - hor_shd_pct_shaded / 100.),
                    'horizon_unshaded': self.horizon['back_pvrow'],
                    'horizon_shd_pct': hor_shd_pct_shaded,
                    'rho': rho_back,
                    'inv_rho': inv_rho_back,
                    'total_perez': np.zeros(n_steps)})

    def get_full_modeling_vectors(self, pvarray, idx):
        """Get the modeling vectors used in matrix calculations of mathematical
        model.

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

    @property
    def gnd_shaded(self):
        """Total timeseries irradiance incident on ground shaded areas"""
        return self.total_perez['ground_shaded']

    @property
    def gnd_illum(self):
        """Total timeseries irradiance incident on ground illuminated areas"""
        return self.total_perez['ground_illum']

    @property
    def pvrow_shaded(self):
        """Total timeseries irradiance incident on PV row's front illuminated
        areas and calculated by Perez transposition"""
        return self.total_perez['front_shaded_pvrow']

    @property
    def pvrow_illum(self):
        """Total timeseries irradiance incident on PV row's front shaded
        areas and calculated by Perez transposition"""
        return self.total_perez['front_illum_pvrow']

    @property
    def sky_luminance(self):
        """Total timeseries isotropic luminance of sky"""
        return self.isotropic_luminance

    def _calculate_horizon_shading_pct_ts(self, ts_pvrows, ts_point_coords,
                                          pvrow_idx, tilted_to_left,
                                          is_back_side=True):
        """Calculate horizon band shading percentage on surfaces of the ordered
        PV array, in a vectorized way.

        Parameters
        ----------
        ts_pvrows : list of :py:class:`~pvfactors.geometry.timeseries.TsPVRow`
            List of timeseries PV rows in the PV array
        ts_point_coords : :py:class:`~pvfactors.geometry.timeseries.TsPointCoords`
            Timeseries coordinates of point that suffers horizon band shading
        pvrow_idx : int
            Index of PV row on which the above point is located
        tilted_to_left : list of bool
            Flags indicating when the PV rows are strictly tilted to the left
        is_back_side : bool
            Flag indicating if point is located on back side of PV row

        Returns
        -------
        horizon_shading_pct : np.ndarray
            Percentage vector of horizon band irradiance shading
            (from 0 to 100)
        """
        n_pvrows = len(ts_pvrows)
        if pvrow_idx == 0:
            shading_pct_left = np.zeros_like(tilted_to_left)
        else:
            high_pt_left = ts_pvrows[
                pvrow_idx - 1].full_pvrow_coords.highest_point
            shading_angle_left = np.rad2deg(np.abs(np.arctan(
                (high_pt_left.y - ts_point_coords.y)
                / (high_pt_left.x - ts_point_coords.x))))
            shading_pct_left = calculate_horizon_band_shading(
                shading_angle_left, self.horizon_band_angle)

        if pvrow_idx == (n_pvrows - 1):
            shading_pct_right = np.zeros_like(tilted_to_left)
        else:
            high_pt_right = ts_pvrows[
                pvrow_idx + 1].full_pvrow_coords.highest_point
            shading_angle_right = np.rad2deg(np.abs(np.arctan(
                (high_pt_right.y - ts_point_coords.y)
                / (high_pt_right.x - ts_point_coords.x))))
            shading_pct_right = calculate_horizon_band_shading(
                shading_angle_right, self.horizon_band_angle)

        if is_back_side:
            shading_pct = np.where(tilted_to_left, shading_pct_right,
                                   shading_pct_left)
        else:
            shading_pct = np.where(tilted_to_left, shading_pct_left,
                                   shading_pct_right)

        return shading_pct

    def _calculate_circumsolar_shading_pct(self, surface, idx_neighbor, pvrows,
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
            percentage_circ_angle_covered = (shading_angle - lower_angle_circumsolar) \
                / self.circumsolar_angle * 100.
            circ_shading_pct = calculate_circumsolar_shading(
                percentage_circ_angle_covered, model=self.circumsolar_model)

        return circ_shading_pct

    @staticmethod
    def _calculate_luminance_poa_components(
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
        poa_circumsolar_back = luminance_circumsolar * vf_circumsolar_backsurface

        # Return only >0 values for poa_horizon
        poa_horizon = np.abs(poa_horizon)

        return luminance_isotropic, luminance_circumsolar, poa_horizon, \
            poa_circumsolar_front, poa_circumsolar_back, \
            aoi_front_pvrow, aoi_back_pvrow
