"""Module with Base classes for irradiance models"""


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

    def transform_ts(self, *args, **kwargs):
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
    def sky(self):
        """Not implemented"""
        raise NotImplementedError

    def get_modeling_vectors(self, pvarray):
        """Get vector of summed up irradiance values from a PV array, as well
        as the inverse reflectivity values (the latter need to be named
        "inv_rho"), and the total perez irradiance values.

        Parameters
        ----------
        pvarray : PV array object
            PV array with the irradiance and reflectivity values

        Returns
        -------
        irradiance_vec : list
            List of summed up non-reflective irradiance values for all surfaces
            and sky
        rho_vec : list
            List of reflectivity values for all surfaces and sky
        invrho_vec : list
            List of inverse reflectivity for all surfaces and sky
        total_perez_vec : list
            List of total perez transposed irradiance values for all surfaces
            and sky
        """

        # TODO: this can probably be speeded up
        irradiance_vec = []
        rho_vec = []
        invrho_vec = []
        total_perez_vec = []
        for _, surface in pvarray.dict_surfaces.items():
            value = 0.
            for component in self.irradiance_comp:
                value += surface.get_param(component)
            irradiance_vec.append(value)
            invrho_vec.append(surface.get_param('inv_rho'))
            rho_vec.append(surface.get_param('rho'))
            total_perez_vec.append(surface.get_param('total_perez'))

        return irradiance_vec, rho_vec, invrho_vec, total_perez_vec

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
