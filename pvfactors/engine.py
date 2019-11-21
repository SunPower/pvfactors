"""This module contains the engine class that will run the complete
timeseries simulations."""

import numpy as np
from pvfactors.viewfactors import VFCalculator
from pvfactors.irradiance import HybridPerezOrdered
from pvfactors.config import DEFAULT_RHO_FRONT, DEFAULT_RHO_BACK


class PVEngine(object):
    """Class putting all of the calculations together into simple
    workflows.
    """

    def __init__(self, pvarray, vf_calculator=None, irradiance_model=None,
                 fast_mode_pvrow_index=None, fast_mode_segment_index=None):
        """Create pv engine class, and initialize timeseries parameters.

        Parameters
        ----------
        pvarray : BasePVArray (or child) object
            The initialized PV array object that will be used for calculations
        vf_calculator : vf calculator object, optional
            Calculator that will be used to calculate the view factor matrices,
            will use :py:class:`~pvfactors.viewfactors.calculator.VFCalculator`
            if None (Default = None)
        irradiance_model : irradiance model object, optional
            The irradiance model that will be applied to the PV array,
            will use
            :py:class:`~pvfactors.irradiance.models.HybridPerezOrdered`
            if None (Default = None)
        fast_mode_pvrow_index : int, optional
            If a pvrow index is passed, then the PVEngine fast mode
            will be activated and the engine calculation will be done only
            for the back surface of the pvrow with the corresponding
            index (Default = None)
        fast_mode_segment_index : int, optional
            If a segment index is passed, then the PVEngine fast mode
            will calculate back surface irradiance only for the
            selected segment of the selected back surface (Default = None)
        """
        # Initialize the attributes of the PV engine
        self.vf_calculator = vf_calculator or VFCalculator()
        self.irradiance = irradiance_model or HybridPerezOrdered()
        self.pvarray = pvarray
        # Save fast mode indices
        self.fast_mode_pvrow_index = fast_mode_pvrow_index
        self.fast_mode_segment_index = fast_mode_segment_index

        # These values will be updated at fitting time
        self.n_points = None
        self.skip_step = None

    @classmethod
    def with_rho_initialization(
            cls, pvarray, vf_calculator, irradiance_model,
            fast_mode_pvrow_index=None, fast_mode_segment_index=None):
        """Before creating the PV engine object, update the front and
        back reflectivity scalars using the faoi functions, if those values
        weren't passed originally

        Parameters
        ----------
        pvarray : :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray`
            The initialized PV array object that will be used for calculations
        vf_calculator : \
        :py:class:`~pvfactors.viewfactors.calculator.VFCalculator`
            Calculator that will be used to calculate the view factor matrices,
            and AOI losses
        irradiance_model : irradiance model object
            The irradiance model that will be applied to the PV array,
            for instance
            :py:class:`~pvfactors.irradiance.models.HybridPerezOrdered`
        fast_mode_pvrow_index : int, optional
            If a pvrow index is passed, then the PVEngine fast mode
            will be activated and the engine calculation will be done only
            for the back surface of the pvrow with the corresponding
            index (Default = None)
        fast_mode_segment_index : int, optional
            If a segment index is passed, then the PVEngine fast mode
            will calculate back surface irradiance only for the
            selected segment of the selected back surface (Default = None)

        Returns
        -------
        PV engine
            PV engine where the rho values have been initialized
        """
        # Calculate global average reflectivity using VF calculator
        is_back = False
        rho_front_calculated = (vf_calculator.vf_aoi_methods
                                .rho_from_faoi_fn(is_back))
        is_back = True
        rho_back_calculated = (vf_calculator.vf_aoi_methods
                               .rho_from_faoi_fn(is_back))
        # Initialize rho values for irradiance model
        irradiance_model.rho_front = irradiance_model.initialize_rho(
            irradiance_model.rho_front, rho_front_calculated,
            DEFAULT_RHO_FRONT)
        irradiance_model.rho_back = irradiance_model.initialize_rho(
            irradiance_model.rho_back, rho_back_calculated,
            DEFAULT_RHO_BACK)

        return cls(pvarray, vf_calculator=vf_calculator,
                   irradiance_model=irradiance_model,
                   fast_mode_pvrow_index=fast_mode_pvrow_index,
                   fast_mode_segment_index=fast_mode_segment_index)

    def fit(self, timestamps, DNI, DHI, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, albedo, ghi=None):
        """Fit the timeseries data to the engine. More specifically,
        save all the parameters that needs to be saved, and fit the PV array
        and irradiance models to the data (i.e. perform all the intermediate
        vector-based calculations).
        Note that all angles follow the pvlib-python angle convention: North -
        0 deg, East - 90 deg, etc.

        Parameters
        ----------
        timestamps : array-like or timestamp-like
            List of timestamps of the simulation.
        DNI : array-like or float
            Direct normal irradiance values [W/m2]
        DHI : array-like or float
            Diffuse horizontal irradiance values [W/m2]
        solar_zenith : array-like or float
            Solar zenith angles [deg]
        solar_azimuth : array-like or float
            Solar azimuth angles [deg]
        surface_tilt : array-like or float
            Surface tilt angles, from 0 to 180 [deg]
        surface_azimuth : array-like or float
            Surface azimuth angles [deg]
        albedo : array-like or float
            Albedo values (ground reflectivity)
        ghi : array-like, optional
            Global horizontal irradiance [W/m2], if not provided, will be
            calculated from DNI and DHI if needed (Default = None)
        """
        # Format inputs to numpy arrays if it looks like floats where inputted
        if np.isscalar(DNI):
            timestamps = [timestamps]
            DNI = np.array([DNI])
            DHI = np.array([DHI])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])
            ghi = None if ghi is None else np.array([ghi])

        # Format albedo
        self.n_points = len(DNI)
        albedo = (albedo * np.ones(self.n_points) if np.isscalar(albedo)
                  else albedo)

        # Fit PV array
        self.pvarray.fit(solar_zenith, solar_azimuth, surface_tilt,
                         surface_azimuth)

        # Fit irradiance model
        self.irradiance.fit(timestamps, DNI, DHI, solar_zenith, solar_azimuth,
                            surface_tilt, surface_azimuth, albedo, ghi=ghi)

        # Add timeseries irradiance results to pvarray
        self.irradiance.transform(self.pvarray)

        # Fit VF calculator
        self.vf_calculator.fit(self.n_points)

        # Skip timesteps when:
        #    - solar zenith > 90, ie the sun is down
        #    - DNI or DHI is negative, which does not make sense
        #    - DNI and DHI are both zero
        self.skip_step = (solar_zenith > 90) | (DNI < 0) | (DHI < 0) \
            | ((DNI == 0) & (DHI == 0))

    def run_full_mode(self, fn_build_report=None):
        """Run all simulation timesteps using the full mode, which calculates
        the equilibrium of reflections in the system, and returns a report that
        will be built by the function passed by the user.

        Parameters
        ----------
        fn_build_report : function, optional
            Function that will build the report of the simulation
            (Default value = None)

        Returns
        -------
        report
            Saved results from the simulation, as specified by user's report
            function. If no function is passed, nothing will be returned.
        """
        # Get pvarray
        pvarray = self.pvarray

        # Get the irradiance modeling matrices
        # shape = n_surfaces, n_timesteps
        irradiance_mat, rho_mat, invrho_mat, _ = \
            self.irradiance.get_full_ts_modeling_vectors(pvarray)

        # --- Calculate view factors
        # shape = n_surfaces, n_surfaces, n_timesteps
        ts_vf_matrix = self.vf_calculator.build_ts_vf_matrix(pvarray)
        pvarray.ts_vf_matrix = ts_vf_matrix
        # Reshape for broadcasting and inverting
        # shape = n_timesteps, n_surfaces, n_surfaces
        ts_vf_matrix_reshaped = np.moveaxis(ts_vf_matrix, -1, 0)

        # --- Solve mathematical problem
        # Build matrix of inverse reflectivities
        # shape = n_surfaces, n_surfaces
        invrho_mat = np.diag(invrho_mat[:, 0])
        # Subtract matrices: will rely on broadcasting
        # shape = n_timesteps, n_surfaces, n_surfaces
        a_mat = invrho_mat - ts_vf_matrix_reshaped
        # Calculate inverse, requires specific shape
        # shape = n_timesteps, n_surfaces, n_surfaces
        inv_a_mat = np.linalg.inv(a_mat)
        # Use einstein sum to get final timeseries radiosities
        # shape = n_surfaces, n_timesteps
        q0 = np.einsum('ijk,ki->ji', inv_a_mat, irradiance_mat)
        # Calculate incident irradiance: will rely on broadcasting
        # shape = n_surfaces, n_timesteps
        qinc = np.dot(invrho_mat, q0)

        # --- Derive other irradiance terms
        # shape = n_surfaces, n_timesteps
        isotropic_mat = ts_vf_matrix[:-1, -1, :] * irradiance_mat[-1, :]
        reflection_mat = qinc[:-1, :] - irradiance_mat[:-1, :] - isotropic_mat

        # --- Calculate AOI losses and absorbed irradiance
        rho_mat = np.tile(rho_mat[:, 0], (rho_mat.shape[0], 1)).T
        # shape [n_surfaces + 1, n_surfaces + 1, n_timestamps]
        vf_aoi_matrix = (self.vf_calculator
                         .build_ts_vf_aoi_matrix(pvarray, rho_mat))
        pvarray.ts_vf_aoi_matrix = vf_aoi_matrix
        # shape [n_surfaces, n_timestamps]
        irradiance_abs_mat = (
            self.irradiance.get_summed_components(pvarray, absorbed=True))
        # Calculate absorbed irradiance
        qabs = (np.einsum('ijk,jk->ik', vf_aoi_matrix, q0)[:-1, :]
                + irradiance_abs_mat)

        # --- Update surfaces with values: the lists are ordered by index
        for idx_surf, ts_surface in enumerate(pvarray.all_ts_surfaces):
            ts_surface.update_params(
                {'q0': q0[idx_surf, :],
                 'qinc': qinc[idx_surf, :],
                 'isotropic': isotropic_mat[idx_surf, :],
                 'reflection': reflection_mat[idx_surf, :],
                 'qabs': qabs[idx_surf, :]})

        # Return report if function was passed
        report = None if fn_build_report is None else fn_build_report(pvarray)
        return report

    def run_fast_mode(self, fn_build_report=None, pvrow_index=None,
                      segment_index=None):
        """Run all simulation timesteps using the fast mode for the back
        surface of a PV row, and assuming that the incident irradiance on all
        other surfaces is known (all but back surfaces).
        The function will return a report that will be built by the function
        passed by the user.

        Parameters
        ----------
        fn_build_report : function, optional
            Function that will build the report of the simulation
            (Default value = None)
        pvrow_index : int, optional
            Index of the PV row for which we want to calculate back surface
            irradiance; if used, this will override the
            ``fast_mode_pvrow_index`` passed at engine initialization
            (Default = None)
        segment_index : int, optional
            Index of the segment on the PV row's back side for which we want
            to calculate the incident irradiance; if used, this will override
            the ``fast_mode_segment_index`` passed at engine initialization
            (Default = None)

        Returns
        -------
        report
            Results from the simulation, as specified by user's report
            function. If no function is passed, nothing will be returned.
        """

        # Prepare variables
        pvrow_idx = (self.fast_mode_pvrow_index if pvrow_index is None
                     else pvrow_index)
        segment_idx = (self.fast_mode_segment_index if segment_index is None
                       else segment_index)
        ts_pvrow = self.pvarray.ts_pvrows[pvrow_idx]

        # Run calculations
        if segment_idx is None:
            # Run calculation for all segments of back surface
            for ts_segment in ts_pvrow.back.list_segments:
                self._calculate_back_ts_segment_qinc(ts_segment, pvrow_idx)
        else:
            # Run calculation for selected segment of back surface
            ts_segment = ts_pvrow.back.list_segments[segment_idx]
            self._calculate_back_ts_segment_qinc(ts_segment, pvrow_idx)

        # Create report
        report = (None if fn_build_report is None
                  else fn_build_report(self.pvarray))

        return report

    def _calculate_back_ts_segment_qinc(self, ts_segment, pvrow_idx):
        """Calculate the incident irradiance on a timeseries segment's surfaces
        for the back side of a PV row, using the fast mode, so assuming that
        the incident irradiance on all other surfaces is known.
        Nothing is returned by the function, but the segment's surfaces'
        parameter values are updated.

        Parameters
        ----------
        ts_segment : :py:class:`~pvfactors.geometry.timeseries.TsDualSegment`
            Timeseries segment for which we want to calculate the incident
            irradiance
        pvrow_idx : int
            Index of the PV row on which the segment is located
        """

        # Get all timeseries surfaces in segment
        list_ts_surfaces = ts_segment.all_ts_surfaces

        # Get irradiance vectors for calculation
        albedo = self.irradiance.albedo
        rho_front = self.irradiance.rho_front
        irr_gnd_shaded = self.irradiance.gnd_shaded
        irr_gnd_illum = self.irradiance.gnd_illum
        irr_pvrow_shaded = self.irradiance.pvrow_shaded
        irr_pvrow_illum = self.irradiance.pvrow_illum
        irr_sky = self.irradiance.sky_luminance

        for ts_surface in list_ts_surfaces:
            # Calculate view factors for timeseries surface
            vf = self.vf_calculator.get_vf_ts_pvrow_element(
                pvrow_idx, ts_surface, self.pvarray.ts_pvrows,
                self.pvarray.ts_ground, self.pvarray.rotation_vec,
                self.pvarray.width)

            # Update sky terms of timeseries surface
            self.irradiance.update_ts_surface_sky_term(ts_surface)

            # Calculate incident irradiance on illuminated surface
            gnd_shadow_refl = vf['to_gnd_shaded'] * albedo * irr_gnd_shaded
            gnd_illum_refl = vf['to_gnd_illum'] * albedo * irr_gnd_illum
            pvrow_shadow_refl = (vf['to_pvrow_shaded'] * rho_front
                                 * irr_pvrow_shaded)
            pvrow_illum_refl = (vf['to_pvrow_illum'] * rho_front
                                * irr_pvrow_illum)
            reflections = (gnd_shadow_refl + gnd_illum_refl + pvrow_shadow_refl
                           + pvrow_illum_refl)
            isotropic = vf['to_sky'] * irr_sky
            qinc = (gnd_shadow_refl + gnd_illum_refl + pvrow_shadow_refl
                    + pvrow_illum_refl + isotropic
                    + ts_surface.get_param('sky_term'))

            # Update parameters of timeseries surface object
            ts_surface.update_params(
                {'qinc': qinc,
                 'reflection_gnd_shaded': gnd_shadow_refl,
                 'reflection_gnd_illum': gnd_illum_refl,
                 'reflection_pvrow_shaded': pvrow_shadow_refl,
                 'reflection_pvrow_illum': pvrow_illum_refl,
                 'isotropic': isotropic,
                 'reflection': reflections,
                 'view_factors': vf})
