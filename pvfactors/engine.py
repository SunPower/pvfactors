"""Engine that will run complete pvfactors calculations"""
import numpy as np
from pvfactors.geometry import OrderedPVArray
from pvfactors.viewfactors import VFCalculator
from pvfactors.irradiance import IsotropicOrdered


class PVEngine(object):
    """Class putting all of the calculations together, and able to run it
    as a timeseries when the pvarrays can be build from dictionary parameters
    """

    def __init__(self, params, vf_calculator=VFCalculator(),
                 irradiance_model=IsotropicOrdered(),
                 cls_pvarray=OrderedPVArray):
        """Create pv engine class, and initialize timeseries parameters"""
        self.params = params
        self.vf_calculator = vf_calculator
        self.irradiance = irradiance_model
        self.cls_pvarray = cls_pvarray

        # Required parameters
        self.solar_zenith = None
        self.solar_azimuth = None
        self.surface_tilt = None
        self.surface_azimuth = None

    def fit(self, DNI, DHI, solar_zenith, solar_azimuth, surface_tilt,
            surface_azimuth):
        """Save timeseries angle data and fit the irradiance model"""
        # Save
        if np.isscalar(DNI):
            DNI = np.array([DNI])
            DHI = np.array([DHI])
            solar_zenith = np.array([solar_zenith])
            solar_azimuth = np.array([solar_azimuth])
            surface_tilt = np.array([surface_tilt])
            surface_azimuth = np.array([surface_azimuth])
        self.solar_zenith = solar_zenith
        self.solar_azimuth = solar_azimuth
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth

        # Fit irradiance model
        self.irradiance.fit(DNI, DHI, solar_zenith, solar_azimuth,
                            surface_tilt, surface_azimuth)

    def run_timestep(self, idx):
        """Run timestep"""
        # Update parameters
        self.params.update(
            {'solar_zenith': self.solar_zenith[idx],
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

        # Apply irradiance terms to pvarray
        self.irradiance.transform(pvarray, idx=idx)

        return pvarray, vf_matrix
