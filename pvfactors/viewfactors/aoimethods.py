"""Module containing AOI loss calculation methods"""

import pvlib
from pvlib.tools import cosd
import numpy as np


class AOIMethods:
    """Class containing methods related to calculating AOI losses for
    :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` objects."""

    def __init__(self, faoi_fn, n_integral_sections=300):
        """Instantiate class with faoi function

        Parameters
        ----------
        faoi_fn : function
            Function which takes a list (or numpy array) of incidence angles
            measured from the surface horizontal
            (with values from 0 to 180 deg) and returns the fAOI values
        n_integral_sections : int, optional
            Number of integral divisions of the 0 to 180 deg interval
            to use for the fAOI loss integral (default = 300)
        """
        self.faoi_fn = faoi_fn
        self.n_integral_sections = n_integral_sections
        # The following will be updated at fitting time
        self.interval = None
        self.aoi_angles_low = None
        self.aoi_angles_high = None
        self.integrand_values = None

    def fit(self, n_timestamps):
        """Fit the AOI methods to timeseries inputs: create all the necessary
        integration attributes.

        Parameters
        ----------
        n_timestamps : int
            Number of simulation timestamps
        """
        # Will use x values at the middle of the integral sections
        aoi_angles = np.linspace(0., 180., num=self.n_integral_sections + 1)
        # Assumes that at least 2 aoi angle values, otherwise what's the point
        self.interval = aoi_angles[1] - aoi_angles[0]
        # Get integral intervals' low, high, and middle points
        aoi_angles_low = aoi_angles[:-1]
        aoi_angles_high = aoi_angles_low + self.interval
        aoi_angles_middle = aoi_angles_low + self.interval / 2.
        # Calculate faoi values using middle points of integral intervals
        faoi_values = self.faoi_fn(aoi_angles_middle)
        # Calculate small view factor values for each section
        vf_values = self._vf(aoi_angles_low, aoi_angles_high)
        # Multiply to get integrand
        integrand_values = faoi_values * vf_values
        # Replicate these values for all timestamps such that shapes
        # becomes: [n_timestamps, n_integral_sections]map
        self.aoi_angles_low = np.tile(aoi_angles_low, (n_timestamps, 1))
        self.aoi_angles_high = np.tile(aoi_angles_high, (n_timestamps, 1))
        self.integrand_values = np.tile(integrand_values, (n_timestamps, 1))

    def _calculate_vf_aoi_pvrow_to_gnd(self, ts_surface):
        pass

    def _calculate_vf_aoi_wedge_level(self, low_angles, high_angles):
        """Calculate faoi modified view factors for a wedge defined by
        low and high angles.

        Parameters
        ----------
        low_angles : np.ndarray
            Low AOI angles (between 0 and 180 deg), length = n_timestamps
        high_angles : np.ndarray
            High AOI angles (between 0 and 180 deg), length = n_timestamps.
            Should be bigger than ``low_angles``

        Returns
        -------
        np.ndarray
            faoi modified view factors for wedge
            shape = (n_timestamps, )
        """
        # Calculate integrand: all d_vf_aoi values
        faoi_integrand = self._calculate_vfaoi_integrand(low_angles,
                                                         high_angles)
        # Total vf_aoi will be sum of all smaller d_vf_aoi values
        total_vf_aoi = faoi_integrand.sum(axis=1)

        return total_vf_aoi

    def _calculate_vfaoi_integrand(self, low_angles, high_angles):
        """
        Calculate the timeseries view factors with aoi loss integrand
        given the low and high angles that define the surface.

        Parameters
        ----------
        low_angles : np.ndarray
            Low AOI angles (between 0 and 180 deg), length = n_timestamps
        high_angles : np.ndarray
            High AOI angles (between 0 and 180 deg), length = n_timestamps.
            Should be bigger than ``low_angles``

        Returns
        -------
        np.ndarray
            vf_aoi integrand values for all timestamps
            shape = (n_timestamps, n_integral_sections)
        """

        # Turn into dimension: [n_timestamps, n_integral_sections]
        low_angles_mat = np.tile(low_angles, (self.n_integral_sections, 1)).T
        high_angles_mat = np.tile(high_angles, (self.n_integral_sections, 1)).T

        # Filter out integrand values outside of range
        count_integral_section = ((low_angles_mat <= self.aoi_angles_high) &
                                  (high_angles_mat > self.aoi_angles_low))
        faoi_integrand = np.where(count_integral_section,
                                  self.integrand_values, 0.)

        return faoi_integrand

    @staticmethod
    def _vf(aoi_1, aoi_2):
        """Calculate view factor from infinitesimal surface to infinite band.

        See illustration: http://www.thermalradiation.net/sectionb/B-71.html
        Here we're using angles measured from the horizontal

        Parameters
        ----------
        aoi_1 : np.ndarray
            Lower angles defining the infinite band
        aoi_2 : np.ndarray
            Higher angles defining the infinite band

        Returns
        -------
        np.ndarray
            View factors from infinitesimal surface to infinite strip

        """
        return 0.5 * np.abs(cosd(aoi_1) - cosd(aoi_2))


def faoi_fn_from_pvlib_sandia(pvmodule_name):
    """Generate a faoi function from a pvlib sandia PV module name

    Parameters
    ----------
    pvmodule_name : str
        Name of PV module in pvlib Sandia module database

    Returns
    -------
    faoi_function
        Function that returns positive loss values for numeric inputs
        between 0 and 180 degrees.
    """
    # Get Sandia module database from pvlib
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    # Grab pv module sandia coeffs from database
    pvmodule = sandia_modules[pvmodule_name]

    def fn(angles):
        """fAOI loss funtion: calculate how much light is absorbed at given
        incidence angles

        Parameters
        ----------
        angles : np.ndarray or list
            Angles measured from surface horizontal, from 0 de 180 deg

        Returns
        -------
        np.ndarray
            fAOI values
        """
        angles = np.array(angles) if isinstance(angles, list) else angles
        # Transform the inputs for the SAPM function
        angles = np.where(angles >= 90, angles - 90, 90. - angles)
        # Use pvlib sapm aoi loss method
        return pvlib.pvsystem.sapm_aoi_loss(angles, pvmodule, upper=1.)

    return fn
