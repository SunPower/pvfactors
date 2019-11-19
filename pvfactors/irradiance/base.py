"""Module with Base classes for irradiance models"""
import numpy as np


class BaseModel(object):
    """Base class for irradiance models"""

    params = None
    cats = None
    irradiance_comp = None

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

    def get_summed_components(self, pvarray, absorbed=True):
        """Get sum of irradiance components for irradiance model,
        either absorbed or only incident.

        Parameters
        ----------
        pvarray : PV array object
            PV array with the irradiance and reflectivity values
        absorbed : bool, optional
            Flag to decide whether to use the absorbed components are not
            (default = True)

        Returns
        -------
        irradiance_mat : list
            Matrix of summed up non-reflective irradiance values for all
            timeseries surfaces. Dimension = [n_surfaces, n_timesteps]"""
        # Initialize
        list_components = (self.irradiance_comp_absorbed if absorbed
                           else self.irradiance_comp)
        # Build list
        irradiance_mat = []
        for ts_surface in pvarray.all_ts_surfaces:
            value = np.zeros(pvarray.n_states, dtype=float)
            for component in list_components:
                value += ts_surface.get_param(component)
            irradiance_mat.append(value)
        return np.array(irradiance_mat)

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

    def initialize_rho(self, rho_scalar, rho_calculated, default_value):
        """Initialize reflectivity value:
        - if a scalar value is passed, use it
        - otherwise try to use calculated value
        - else use default value

        Parameters
        ----------
        rho_scalar : float
            Global average reflectivity value that is supposed to be used
        rho_calculated : float
            Reflectivity value calculated
        default_value : float
            Default value to use if everything fails

        Returns
        -------
        rho_scalar : float
            Global average reflectivity
        """
        if np.isscalar(rho_scalar):
            rho_scalar = rho_scalar
        elif np.isscalar(rho_calculated):
            rho_scalar = rho_calculated
        else:
            rho_scalar = default_value
        return rho_scalar
