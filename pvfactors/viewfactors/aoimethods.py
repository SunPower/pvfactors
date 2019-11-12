"""Module containing AOI loss calculation methods"""

import pvlib
import numpy as np


class TsAOIMethods:

    def __init__(self, faoi_fn):
        self.faoi_fn = faoi_fn


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

        """
        angles = np.array(angles) if isinstance(angles, list) else angles
        # Transform the inputs for the SAPM function
        angles = np.where(angles >= 90, angles - 90,
                          90. - angles)
        # Use pvlib sapm aoi loss method
        return pvlib.pvsystem.sapm_aoi_loss(angles, pvmodule, upper=1.)

    return fn
