"""Module containing AOI loss calculation methods"""

import pvlib
import numpy as np


class TsAOIMethods:
    """Class containing methods related to calculating AOI losses for
    :py:class:`~pvfactors.geometry.pvarray.OrderedPVArray` objects."""

    def __init__(self, faoi_fn, n_timestamps, n_integral_sections=180):
        """Instantiate class with faoi function

        Parameters
        ----------
        faoi_fn : function
            Function which takes a list (or numpy array) of incidence angles
            measured from the surface horizontal
            (with values from 0 to 180 deg) and returns the fAOI values
        n_timestamps : int
            Number of simulation timestamps
        n_integral_sections : int, optional
            Number of integral divisions of the 0 to 180 deg interval
            to use for the fAOI loss integral (default = 180)
        """
        self.faoi_fn = faoi_fn
        self.n_integral_sections = n_integral_sections
        # Will use x values at the middle of the integral sections
        aoi_angles = np.linspace(0., 180., num=n_integral_sections + 1)
        # Assumes that at least 2 aoi angle values, otherwise what's the point
        self.interval = aoi_angles[1] - aoi_angles[0]
        # Get integral intervals' low, high, and middle points
        aoi_angles_low = aoi_angles[:-1]
        aoi_angles_high = aoi_angles_low + self.interval
        aoi_angles_middle = aoi_angles_low + self.interval / 2.
        # Calculate faoi values using middle points of integral intervals
        faoi_values = faoi_fn(aoi_angles_middle)
        # Replicate these values for all timestamps such that shapes
        # becomes: [n_timestamps, n_integral_sections]
        self.aoi_angles_low = np.tile(aoi_angles_low, (n_timestamps, 1))
        self.aoi_angles_high = np.tile(aoi_angles_high, (n_timestamps, 1))
        self.faoi_values = np.tile(faoi_values, (n_timestamps, 1))

    def _calculate_faoi_values(self, low_angles, high_angles):
        pass

    def _calculate_faoi_integrand(self, low_angles, high_angles):
        """
        Calculate the timeseries fAOI loss integrand given the low and high
        angles to use in the fAOI function.

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
            fAOI integrand values for all timestamps
            shape = (n_timestamps, n_integral_sections)
        """

        # Turn into dimension: [n_timestamps, n_integral_sections]
        low_angles_mat = np.tile(low_angles, (self.n_integral_sections, 1)).T
        high_angles_mat = np.tile(high_angles, (self.n_integral_sections, 1)).T

        # Filter out integrand values outside of range
        count_integral_section = ((low_angles_mat <= self.aoi_angles_high) &
                                  (high_angles_mat > self.aoi_angles_low))
        faoi_integrand = np.where(count_integral_section, self.faoi_values, 0.)

        return faoi_integrand


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
