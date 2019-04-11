"""Module with Base classes for irradiance models"""


class BaseModel(object):
    """Base class for irradiance models"""

    params = None
    cats = None
    irradiance_comp = None

    def __init__(self):
        pass

    def fit(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def get_irradiance_invrho_vector(self, pvarray):
        """Get vector of summed up irradiance values, and inverse reflectivity
        values"""

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
