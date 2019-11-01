"""Module with Base classes for irradiance models"""
import numpy as np


class BaseModel(object):
    """Base class for irradiance models"""

    params = None
    cats = None
    irradiance_comp = None

    def __init__(self, *args, **kwargs):
        """Not implemented"""
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        """Not implemented"""
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        """Not implemented"""
        raise NotImplementedError

    def get_full_modeling_vectors(self, *args, **kwargs):
        """Not implemented"""
        raise NotImplementedError

    @property
    def gnd_shaded(self):
        """Not implemented"""
        raise NotImplementedError

    @property
    def gnd_illum(self):
        """Not implemented"""
        raise NotImplementedError

    @property
    def pvrow_shaded(self):
        """Not implemented"""
        raise NotImplementedError

    @property
    def pvrow_illum(self):
        """Not implemented"""
        raise NotImplementedError

    @property
    def sky_luminance(self):
        """Not implemented"""
        raise NotImplementedError

    def get_ts_modeling_vectors(self, pvarray):
        """Get matrices of summed up irradiance values from a PV array, as well
        as the inverse reflectivity values (the latter need to be named
        "inv_rho"), and the total perez irradiance values.

        Parameters
        ----------
        pvarray : PV array object
            PV array with the irradiance and reflectivity values

        Returns
        -------
        irradiance_mat : list
            Matrix of summed up non-reflective irradiance values for all
            timeseries surfaces. Dimension = [n_surfaces, n_timesteps]
        rho_mat : list
            Matrix of reflectivity values for all timeseries surfaces
            Dimension = [n_surfaces, n_timesteps]
        invrho_mat : list
            List of inverse reflectivity for all timeseries surfaces
            Dimension = [n_surfaces, n_timesteps]
        total_perez_mat : list
            List of total perez transposed irradiance values for all timeseries
            surfaces
            Dimension = [n_surfaces, n_timesteps]
        """

        # TODO: this can probably be speeded up
        irradiance_mat = []
        rho_mat = []
        invrho_mat = []
        total_perez_mat = []
        # In principle, the list all ts surfaces should be ordered
        # with the ts surfaces' indices. ie first element has index 0, etc
        for ts_surface in pvarray.all_ts_surfaces:
            value = np.zeros(pvarray.n_states, dtype=float)
            for component in self.irradiance_comp:
                value += ts_surface.get_param(component)
            irradiance_mat.append(value)
            invrho_mat.append(ts_surface.get_param('inv_rho'))
            rho_mat.append(ts_surface.get_param('rho'))
            total_perez_mat.append(ts_surface.get_param('total_perez'))

        return irradiance_mat, rho_mat, invrho_mat, total_perez_mat

    def update_ts_surface_sky_term(self, ts_surface, name_sky_term='sky_term'):
        """Update the 'sky_term' parameter of a timeseries surface.

        Parameters
        ----------
        ts_surface :  :py:class:`~pvfactors.geometry.timeseries.TsSurface`
            Timeseries surface whose 'sky_term' parameter value we want to
            update
        name_sky_term : str, optional
            Name of the sky term parameter (Default = 'sky_term')
        """
        value = 0.
        for component in self.irradiance_comp:
            value += ts_surface.get_param(component)
        ts_surface.update_params({name_sky_term: value})
