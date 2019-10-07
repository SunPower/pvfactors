"""This module contains the engine classes that will run the complete
timeseries simulations."""

import numpy as np
from pvfactors.viewfactors import VFCalculator
from pvfactors.irradiance import HybridPerezOrdered
from scipy import linalg
from tqdm import tqdm


class PVEngine(object):
    """Class putting all of the calculations together, and able to run it
    as a timeseries when the pvarrays can be build from dictionary parameters
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
        self.vf_calculator = (VFCalculator() if vf_calculator is None
                              else vf_calculator)
        self.irradiance = (HybridPerezOrdered() if irradiance_model is None
                           else irradiance_model)
        self.pvarray = pvarray
        # Save fast mode indices
        self.fast_mode_pvrow_index = fast_mode_pvrow_index
        self.fast_mode_segment_index = fast_mode_segment_index

        # These values will be updated at fitting time
        self.n_points = None
        self.skip_step = None

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

        # Skip timesteps when:
        #    - solar zenith > 90, ie the sun is down
        #    - DNI or DHI is negative, which does not make sense
        #    - DNI and DHI are both zero
        self.skip_step = (solar_zenith > 90) | (DNI < 0) | (DHI < 0) \
            | ((DNI == 0) & (DHI == 0))

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

    def run_full_mode(self, fn_build_report=None):
        """Run all simulation timesteps using the full mode, which calculates
        the equilibrium of reflections in the system, and return a report that
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

        report = None
        for idx in tqdm(range(self.n_points)):
            pvarray = self.run_full_mode_timestep(idx)
            report = (None if fn_build_report is None
                      else fn_build_report(report, pvarray))

        return report

    def run_full_mode_timestep(self, idx):
        """Run simulation for a single timestep index and using the full mode,
        which calculates the equilibrium of reflections in the system.

        Timestep will be skipped when:
            - solar zenith > 90, ie the sun is down
            - DNI or DHI is negative, which does not make sense
            - DNI and DHI are both zero

        Parameters
        ----------
        idx : int
            Index for which to run simulation

        Returns
        -------
        pvarray : PV array object or None
            PV array object after all the calculations are performed and
            applied to it, ``None`` if the timestep is skipped

        """

        if self.skip_step[idx]:
            pvarray = None
        else:
            # To be returned at the end
            pvarray = self.pvarray

            # Transform pvarray to time step idx to create geometries
            pvarray.transform(idx)

            # Get the irradiance modeling vectors used in final calculations
            irradiance_vec, _, invrho_vec, _ = \
                self.irradiance.get_full_modeling_vectors(pvarray, idx)

            # Prepare inputs to view factor calculator
            geom_dict = pvarray.dict_surfaces
            view_matrix, obstr_matrix = pvarray.view_obstr_matrices

            # Calculate view factors
            vf_matrix = self.vf_calculator.get_vf_matrix(
                geom_dict, view_matrix, obstr_matrix, pvarray.pvrows)
            pvarray.vf_matrix = vf_matrix

            # Calculate radiosities by solving system of equations
            invrho_mat = np.diag(invrho_vec)
            a_mat = invrho_mat - vf_matrix
            q0 = linalg.solve(a_mat, irradiance_vec)
            qinc = np.dot(invrho_mat, q0)

            # Derive other irradiance terms
            isotropic_vec = vf_matrix[:-1, -1] * irradiance_vec[-1]
            reflection_vec = qinc[:-1] \
                - irradiance_vec[:-1] - isotropic_vec

            # Update surfaces with values
            for idx_surf, surface in geom_dict.items():
                surface.update_params(
                    {'q0': q0[idx_surf],
                     'qinc': qinc[idx_surf],
                     'isotropic': isotropic_vec[idx_surf],
                     'reflection': reflection_vec[idx_surf]})

        return pvarray

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
        list_ts_surfaces = [ts_segment.illum, ts_segment.shaded]

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
