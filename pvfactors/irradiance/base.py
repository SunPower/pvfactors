"""Module with Base classes for irradiance models"""


class BaseModel(object):
    """Base class for irradiance models"""

    params = None
    cats = None
    irradiance_comp = None

    def __init__(self):
        """Not implemented"""
        raise NotImplementedError

    def fit(self):
        """Not implemented"""
        raise NotImplementedError

    def transform(self):
        """Not implemented"""
        raise NotImplementedError

    def get_irradiance_invrho_vector(self, pvarray):
        """Get vector of summed up irradiance values from a PV array, as well
        as the inverse reflectivity values (the latter need to be named
        "inv_rho").

        Parameters
        ----------
        pvarray : PV array object
            PV array with the irradiance and reflectivity values

        Returns
        -------
        irradiance_vec : list
            List of summed up irradiance values
        invrho_vec : list
            List of inverse reflectivity values

        """

        # TODO: this can probably be speeded up
        irradiance_vec = []
        invrho_vec = []
        for idx, surface in pvarray.dict_surfaces.items():
            value = 0.
            for component in self.irradiance_comp:
                value += surface.get_param(component)
            irradiance_vec.append(value)
            invrho_vec.append(surface.get_param('inv_rho'))

        return irradiance_vec, invrho_vec
