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

    def __init__(self, pvarray, vf_calculator=VFCalculator(),
                 irradiance_model=HybridPerezOrdered(),
                 fast_mode_pvrow_index=None):
        """Create pv engine class, and initialize timeseries parameters.

        Parameters
        ----------
        pvarray : BasePVArray (or child) object
            The initialized PV array object that will be used for calculations
        vf_calculator : vf calculator object, optional
            Calculator that will be used to calculate the view factor matrices
            (Default =
            :py:class:`~pvfactors.viewfactors.calculator.VFCalculator` object)
        irradiance_model : irradiance model object, optional
            The irradiance model that will be applied to the PV array
            (Default =
            :py:class:`~pvfactors.irradiance.models.HybridPerezOrdered`
            object)
        fast_mode_pvrow_index : int, optional
            If a pvrow index is passed, then the PVEngine fast mode
            will be activated and the engine calculation will be done only
            for the back surface of the pvrow with the corresponding
            index (Default = None)

        """
        self.vf_calculator = vf_calculator
        self.irradiance = irradiance_model
        self.pvarray = pvarray
        self.is_fast_mode = False if fast_mode_pvrow_index is None else True
        self.fast_mode_pvrow_index = fast_mode_pvrow_index

        # These values will be updated at fitting time
        self.n_points = None
        self.skip_step = None

    def fit(self, timestamps, DNI, DHI, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, albedo):
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

        # Format albedo
        self.n_points = len(DNI)
        if np.isscalar(albedo):
            albedo = albedo * np.ones(self.n_points)

        # Fit irradiance model
        self.irradiance.fit(timestamps, DNI, DHI, solar_zenith, solar_azimuth,
                            surface_tilt, surface_azimuth, albedo)

        # Fit PV array
        self.pvarray.fit(solar_zenith, solar_azimuth, surface_tilt,
                         surface_azimuth)

        # Skip timesteps when:
        #    - solar zenith > 90, ie the sun is down
        #    - DNI or DHI is negative, which does not make sense
        #    - DNI and DHI are both zero
        self.skip_step = (solar_zenith > 90) | (DNI < 0) | (DHI < 0) \
            | ((DNI == 0) & (DHI == 0))

    def run_timestep(self, idx):
        """Run simulation for a single timestep index.

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

            # Transform pvarray to time step
            pvarray.transform(idx)

            # Apply irradiance terms to pvarray
            irradiance_vec, rho_vec, invrho_vec, total_perez_vec = \
                self.irradiance.transform(pvarray, idx=idx)

            # Prepare inputs to view factor calculator
            geom_dict = pvarray.dict_surfaces
            view_matrix, obstr_matrix = pvarray.view_obstr_matrices

            if self.is_fast_mode:
                # Indices of the surfaces of the back of the selected pvrows
                list_surface_indices = pvarray.pvrows[
                    self.fast_mode_pvrow_index].back.surface_indices

                # Calculate view factors using a subset of view_matrix to
                # gain in calculation speed
                vf_matrix_subset = self.vf_calculator.get_vf_matrix_subset(
                    geom_dict, view_matrix, obstr_matrix, pvarray.pvrows,
                    list_surface_indices)
                pvarray.vf_matrix = vf_matrix_subset

                irradiance_vec_subset = irradiance_vec[list_surface_indices]
                # In fast mode, will not care to calculate q0
                qinc = vf_matrix_subset.dot(rho_vec * total_perez_vec) \
                    + irradiance_vec_subset

                # Calculate other terms
                isotropic_vec = vf_matrix_subset[:, -1] * total_perez_vec[-1]
                reflection_vec = qinc - irradiance_vec_subset \
                    - isotropic_vec

                # Update selected surfaces with values
                for i, surf_idx in enumerate(list_surface_indices):
                    surface = geom_dict[surf_idx]
                    surface.update_params({'qinc': qinc[i],
                                           'isotropic': isotropic_vec[i],
                                           'reflection': reflection_vec[i]})

            else:
                # Calculate view factors
                vf_matrix = self.vf_calculator.get_vf_matrix(
                    geom_dict, view_matrix, obstr_matrix, pvarray.pvrows)
                pvarray.vf_matrix = vf_matrix

                # Calculate radiosities
                invrho_mat = np.diag(invrho_vec)
                a_mat = invrho_mat - vf_matrix
                q0 = linalg.solve(a_mat, irradiance_vec)
                qinc = np.dot(invrho_mat, q0)

                # Calculate other terms
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

    def run_all_timesteps(self, fn_build_report=None):
        """Run all simulation timesteps and return a report that will be built
        by the function passed by the user.

        Parameters
        ----------
        fn_build_report : function, optional
            Function that will build the report of the simulation
            (Default value = None)

        Returns
        -------
        report
            Saved results from the simulation, as specified by user's report
            function

        """

        report = None
        for idx in tqdm(range(self.n_points)):
            pvarray = self.run_timestep(idx)
            if fn_build_report is not None:
                report = fn_build_report(report, pvarray)

        return report
