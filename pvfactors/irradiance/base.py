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
