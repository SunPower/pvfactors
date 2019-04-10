from pvfactors.engine import PVEngine
from pvfactors.geometry import OrderedPVArray
import numpy as np


def test_pvengine_float_inputs(params):
    """Test that PV engine works for float inputs"""
    eng = PVEngine(params)

    # Irradiance inputs
    DNI = 1000.
    DHI = None

    # Fit engine
    eng.fit(DNI, DHI, params['solar_zenith'], params['solar_azimuth'],
            params['surface_tilt'], params['surface_azimuth'])
    # Checks
    np.testing.assert_almost_equal(eng.irradiance.dni_front_pvrow, DNI)

    # Run timestep
    pvarray, vf_matrix = eng.run_timestep(0)
    # Checks
    assert isinstance(pvarray, OrderedPVArray)
    assert vf_matrix.shape[0] == pvarray.n_surfaces + 1
