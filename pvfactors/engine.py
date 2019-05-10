"""This module contains the engine classes that will run the complete
timeseries simulations."""

import numpy as np
from pvfactors.geometry import OrderedPVArray
from pvfactors.viewfactors import VFCalculator
from pvfactors.irradiance import HybridPerezOrdered
from scipy import linalg
from tqdm import tqdm


class PVEngine(object):
    """Class putting all of the calculations together, and able to run it
    as a timeseries when the pvarrays can be build from dictionary parameters
    """

    def __init__(self, params, vf_calculator=VFCalculator(),
                 irradiance_model=HybridPerezOrdered(),
                 cls_pvarray=OrderedPVArray):
        """Create pv engine class, and initialize timeseries parameters.

        Parameters
        ----------
        params : dict
            The parameters defining the PV array
        vf_calculator : vf calculator object, optional
            Calculator that will be used to calculate the view factor matrices
            (Default =
            :py:class:`~pvfactors.viewfactors.calculator.VFCalculator` object)
        irradiance_model : irradiance model object, optional
            The irradiance model that will be applied to the PV array
            (Default =
            :py:class:`~pvfactors.irradiance.models.HybridPerezOrdered`
            object)
        cls_pvarray : class of PV array, optional
            Class that will be used to build the PV array
            (Default =
            :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` class)

        """
        self.params = params
        self.vf_calculator = vf_calculator
        self.irradiance = irradiance_model
        self.cls_pvarray = cls_pvarray

        # Required timeseries values
        self.solar_zenith = None
        self.solar_azimuth = None
        self.surface_tilt = None
        self.surface_azimuth = None
        self.n_points = None

    def fit(self, timestamps, DNI, DHI, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, albedo):
        """Fit the timeseries data to the engine. More specifically,
        save all the parameters that needs to be saved, and perform the
        irradiance transformations required by the irradiance model.
        Note that all angles follow the pvlib-python angle convention: North -
        0 deg, East - 90 deg, etc.

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

        """
        # Save
        if np.isscalar(DNI):
            timestamps = [timestamps]
            DNI = np.array([DNI])
            DHI = np.array([DHI])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])
        self.n_points = len(DNI)
        if np.isscalar(albedo):
            albedo = albedo * np.ones(self.n_points)

        # Save timeseries values
        self.solar_zenith = solar_zenith
        self.solar_azimuth = solar_azimuth
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth

        # Fit irradiance model
        self.irradiance.fit(timestamps, DNI, DHI, solar_zenith, solar_azimuth,
                            surface_tilt, surface_azimuth,
                            self.params['rho_front_pvrow'],
                            self.params['rho_back_pvrow'], albedo)

    def run_timestep(self, idx):
        """Run simulation for a single timestep index.

        Parameters
        ----------
        idx : int
            Index for which to run simulation

        Returns
        -------
        pvarray : PV array object
            PV array object after all the calculations are performed and
            applied to it

        """

        # Check that the sun is up, if not do not run calculations
        solar_zenith = self.solar_zenith[idx]
        sun_is_up = solar_zenith <= 90

        if sun_is_up:
            # Update parameters
            self.params.update(
                {'solar_zenith': solar_zenith,
                 'solar_azimuth': self.solar_azimuth[idx],
                 'surface_tilt': self.surface_tilt[idx],
                 'surface_azimuth': self.surface_azimuth[idx]})

            # Create pv array
            pvarray = self.cls_pvarray.from_dict(
                self.params, surface_params=self.irradiance.params)
            pvarray.cast_shadows()
            pvarray.cuts_for_pvrow_view()

            # Calculate view factors
            geom_dict = pvarray.dict_surfaces
            vf_matrix = self.vf_calculator.get_vf_matrix(
                geom_dict, pvarray.view_matrix, pvarray.obstr_matrix,
                pvarray.pvrows)
            pvarray.vf_matrix = vf_matrix

            # Apply irradiance terms to pvarray
            irradiance_vec, invrho_vec = \
                self.irradiance.transform(pvarray, idx=idx)

            # Calculate radiosities
            invrho_mat = np.diag(invrho_vec)
            a_mat = invrho_mat - vf_matrix
            q0 = linalg.solve(a_mat, irradiance_vec)
            qinc = np.dot(invrho_mat, q0)

            # Calculate other terms
            isotropic_vec = vf_matrix[:-1, -1] * irradiance_vec[-1]
            reflection_vec = qinc[:-1] - irradiance_vec[:-1] - isotropic_vec

            # Update surfaces with values
            for idx, surface in geom_dict.items():
                surface.update_params({'q0': q0[idx], 'qinc': qinc[idx],
                                       'isotropic': isotropic_vec[idx],
                                       'reflection': reflection_vec[idx]})
        else:
            pvarray = None

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
